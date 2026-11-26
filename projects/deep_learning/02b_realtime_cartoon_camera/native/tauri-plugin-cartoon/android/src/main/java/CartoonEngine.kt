package cn.coursenotes.cartoon

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import app.tauri.plugin.JSObject
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo
import java.io.ByteArrayOutputStream
import java.nio.FloatBuffer
import kotlin.math.*

/** Same frozen CNN and E/S/Q rules as the desktop; mobile uses raster rendering. */
class CartoonEngine(context: Context) : AutoCloseable {
    private val config = JSONObject(context.assets.open("cartoon_config.json").bufferedReader().use { it.readText() })
    private val env = OrtEnvironment.getEnvironment()
    private val options = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(min(4, Runtime.getRuntime().availableProcessors()))
        setInterOpNumThreads(1)
    }
    private val session: OrtSession
    private val face: FaceLandmarker
    private var timestamp = 0L
    private var gain = 1f
    private var dark = false
    private var threshold: Float? = null
    private var revision = -1

    init {
        check(OpenCVLoader.initLocal()) { "OpenCV 初始化失败" }
        Core.setNumThreads(2)
        session = env.createSession(context.assets.open("mobilenetv3_features.onnx").use { it.readBytes() }, options)
        try {
            OnnxTensor.createTensor(env, FloatBuffer.wrap(FloatArray(3*256*256)), longArrayOf(1,3,256,256)).use { input ->
                session.run(mapOf("image" to input)).use { outputs -> require(outputs.size() == 3) }
            }
            face = FaceLandmarker.createFromOptions(context, FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath("face_landmarker.task").build())
                .setRunningMode(RunningMode.VIDEO).setNumFaces(1).build())
        } catch (error: Exception) { session.close(); options.close(); throw error }
    }

    fun reset() { gain = 1f; dark = false; threshold = null }
    override fun close() {
        try { face.close() } finally { try { session.close() } finally { options.close() } }
    }

    private class Mats : AutoCloseable {
        private val values = mutableListOf<Mat>()
        fun keep(mat: Mat = Mat()): Mat { values.add(mat); return mat }
        override fun close() { values.asReversed().forEach { it.release() } }
    }

    fun render(bytes: ByteArray, meta: JSObject): Pair<JSObject, ByteArray> {
        val start = System.nanoTime()
        val capture = meta.getString("kind") == "capture"
        val size = if (capture) 384 else 256
        val style = meta.getJSONObject("style")
        val density = style.getDouble("lineDensity").toFloat()
        val paletteName = style.getString("palette")
        require(density in 0f..1f && paletteName in setOf("natural_warm", "pastel"))
        val currentRevision = style.getInt("revision")
        if (currentRevision != revision) { threshold = null; revision = currentRevision }
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
        require(bounds.outWidth > 0 && bounds.outHeight > 0 && bounds.outWidth.toLong()*bounds.outHeight <= 16_000_000)
        val source = requireNotNull(BitmapFactory.decodeByteArray(bytes, 0, bytes.size)) { "无法解码图片" }
        try {
            Mats().use { pool ->
                val rgba = pool.keep(); Utils.bitmapToMat(source, rgba)
                val original = pool.keep(); Imgproc.cvtColor(rgba, original, Imgproc.COLOR_RGBA2RGB)
                val scale = size.toDouble()/max(source.width, source.height)
                val width = max(1, (source.width*scale).roundToInt())
                val height = max(1, (source.height*scale).roundToInt())
                val left = (size-width)/2; val top = (size-height)/2
                val resized = pool.keep(); Imgproc.resize(original, resized, Size(width.toDouble(), height.toDouble()), 0.0, 0.0, Imgproc.INTER_LANCZOS4)
                val padded = pool.keep(); Core.copyMakeBorder(resized, padded, top, size-height-top, left, size-width-left, Core.BORDER_REPLICATE)
                val rgb = pool.keep(); Imgproc.bilateralFilter(padded, rgb, 5, 18.0, 2.0)
                val gray8 = pool.keep(); Imgproc.cvtColor(rgb, gray8, Imgproc.COLOR_RGB2GRAY)
                val luminance = ByteArray(size*size); gray8.get(0,0,luminance)
                val valid = BooleanArray(size*size) { i -> i%size in left until left+width && i/size in top until top+height }
                val median = CartoonMath.quantile(FloatArray(width*height) { i -> (luminance[(i/width+top)*size+i%width+left].toInt() and 255).toFloat() }, .5f)
                if (capture) gain = if (median < 60f) min(1.35f,65f/max(1f,median)) else 1f
                else {
                    dark = median < if (dark) 65f else 55f
                    val target = if (dark) min(1.35f,65f/max(1f,median)) else 1f
                    gain += .25f*(target-gain)
                }
                rgb.convertTo(rgb, -1, gain.toDouble()); Imgproc.cvtColor(rgb, gray8, Imgproc.COLOR_RGB2GRAY)
                val gray = pool.keep(); gray8.convertTo(gray, CvType.CV_32F, 1.0/255)
                val gx = pool.keep(); val gy = pool.keep()
                Imgproc.Sobel(gray, gx, CvType.CV_32F,1,0,3,1.0/8,0.0)
                Imgproc.Sobel(gray, gy, CvType.CV_32F,0,1,3,1.0/8,0.0)
                val magnitude = pool.keep(); val angle = pool.keep(); Core.cartToPolar(gx, gy, magnitude, angle)
                val magnitudes = FloatArray(size*size); magnitude.get(0,0,magnitudes)
                val directions = FloatArray(size*size); angle.get(0,0,directions)
                val interior = BooleanArray(size*size) { i -> i%size in left+2 until left+width-2 && i/size in top+2 until top+height-2 }
                val edge = CartoonMath.normalize(magnitudes, interior)
                val (featureMask, cheekMask, faceStatus) = geometry(rgb, size, pool)
                val prepareMs = (System.nanoTime()-start)/1e6
                val cnnStart = System.nanoTime()
                val pixels = ByteArray(size*size*3); rgb.get(0,0,pixels)
                val model = config.getJSONObject("model")
                val mean = model.getJSONArray("mean"); val std = model.getJSONArray("std")
                val inputArray = FloatArray(3*size*size)
                for (c in 0..2) for (i in 0 until size*size)
                    inputArray[c*size*size+i] = (((pixels[i*3+c].toInt() and 255)/255.0-mean.getDouble(c))/std.getDouble(c)).toFloat()
                val support = FloatArray(size*size)
                OnnxTensor.createTensor(env, FloatBuffer.wrap(inputArray), longArrayOf(1,3,size.toLong(),size.toLong())).use { input ->
                    session.run(mapOf("image" to input)).use { output ->
                        for (layer in 0..2) {
                            val tensor = output[layer] as OnnxTensor
                            val shape = tensor.info.shape
                            val channels = shape[1].toInt(); val fh = shape[2].toInt(); val fw = shape[3].toInt()
                            val features = FloatArray(channels*fh*fw); tensor.floatBuffer.get(features)
                            val difference = CartoonMath.featureDifference(features, channels, fh, fw)
                            val featureValid = BooleanArray(fh*fw) { i ->
                                val x = i%fw; val y = i/fw
                                x>0 && y>0 && x<fw-1 && y<fh-1 &&
                                    (x-1)*size/fw >= left && (x+1)*size/fw < left+width &&
                                    (y-1)*size/fh >= top && (y+1)*size/fh < top+height
                            }
                            val response = CartoonMath.normalize(difference, featureValid)
                            val small = pool.keep(Mat(fh, fw, CvType.CV_32F)); small.put(0,0,response)
                            val large = pool.keep(); Imgproc.resize(small, large, Size(size.toDouble(),size.toDouble()))
                            val array = FloatArray(size*size); large.get(0,0,array)
                            val weight = floatArrayOf(.5f,.3f,.2f)[layer]
                            for (i in support.indices) if (valid[i]) support[i] += weight*array[i]
                        }
                    }
                }
                val cnnMs = (System.nanoTime()-cnnStart)/1e6
                val renderStart = System.nanoTime()
                val score = FloatArray(size*size) { i -> edge[i]*(.25f+.75f*support[i]) }
                val reference = nms(edge,directions,size,pool)
                val refValues = reference.filter { it>.04f }.toFloatArray()
                val target = max(.1f, (if (refValues.isEmpty()) 1f else CartoonMath.quantile(refValues,.85f-.3f*density))*(.48f-.2f*density))
                val high = if (capture || threshold == null) target else threshold!!+.25f*(target-threshold!!)
                if (!capture) threshold = high
                val selected = nms(score,directions,size,pool)
                val weak = ByteArray(size*size); val strong = BooleanArray(size*size)
                for (i in weak.indices) {
                    val multiplier = if (featureMask[i] && interior[i]) .8f else if (cheekMask[i] && interior[i]) 1.3f else 1f
                    if (selected[i] >= high*.45f*multiplier) weak[i] = 1
                    strong[i] = selected[i] >= high*multiplier
                }
                val mask = pool.keep(Mat(size,size,CvType.CV_8U)); mask.put(0,0,weak)
                val labels = pool.keep(); val stats = pool.keep(); val centers = pool.keep()
                val count = Imgproc.connectedComponentsWithStats(mask,labels,stats,centers,8,CvType.CV_32S)
                val labelData = IntArray(size*size); labels.get(0,0,labelData)
                val statData = IntArray(count*5); stats.get(0,0,statData)
                val accepted = BooleanArray(count); val protectedComponents = BooleanArray(count)
                for (i in labelData.indices) {
                    if (strong[i]) accepted[labelData[i]] = true
                    if (featureMask[i] && interior[i] && weak[i].toInt()!=0) protectedComponents[labelData[i]] = true
                }
                accepted[0] = false
                val minArea = max(2, (7.0*(size/384.0).pow(2)).roundToInt())
                for (label in 1 until count) accepted[label] = accepted[label] && (statData[label*5+4]>=minArea || protectedComponents[label])
                val ink = CartoonMath.thin(BooleanArray(size*size) { accepted[labelData[it]] && valid[it] },size)
                val colors = colors(rgb,paletteName,pool)
                val inkColor = config.getJSONObject("style").getJSONArray("ink_rgb")
                for (i in ink.indices) if (ink[i]) for (c in 0..2) colors[i*3+c] = inkColor.getInt(c).toByte()
                val output = pool.keep(Mat(size,size,CvType.CV_8UC3)); output.put(0,0,colors)
                val cropped = pool.keep(output.submat(Rect(left,top,width,height)))
                if (capture && style.optBoolean("mirrored")) Core.flip(cropped,cropped,1)
                val outputRgba = pool.keep(); Imgproc.cvtColor(cropped,outputRgba,Imgproc.COLOR_RGB2RGBA)
                val bitmap = Bitmap.createBitmap(width,height,Bitmap.Config.ARGB_8888)
                val encoded = try {
                    Utils.matToBitmap(outputRgba,bitmap)
                    ByteArrayOutputStream().use { stream ->
                        bitmap.compress(if(capture) Bitmap.CompressFormat.PNG else Bitmap.CompressFormat.JPEG,88,stream)
                        stream.toByteArray()
                    }
                } finally { bitmap.recycle() }
                val result = JSObject(meta.toString()).put("mime",if(capture) "image/png" else "image/jpeg")
                    .put("processingMs",(System.nanoTime()-start)/1e6).put("faceStatus",faceStatus)
                    .put("width",width).put("height",height).put("renderer","android_raster")
                    .put("timings",JSObject().put("prepareFace",prepareMs).put("cnnStructure",cnnMs).put("renderEncode",(System.nanoTime()-renderStart)/1e6))
                return Pair(result,encoded)
            }
        } finally { source.recycle() }
    }

    private fun geometry(rgb: Mat,size: Int,pool: Mats): Triple<BooleanArray,BooleanArray,String> {
        val features = pool.keep(Mat.zeros(size,size,CvType.CV_8U)); val cheeks = pool.keep(Mat.zeros(size,size,CvType.CV_8U))
        val rgba = pool.keep(); Imgproc.cvtColor(rgb,rgba,Imgproc.COLOR_RGB2RGBA)
        val bitmap = Bitmap.createBitmap(size,size,Bitmap.Config.ARGB_8888)
        var status = "not_detected"
        try {
            Utils.matToBitmap(rgba,bitmap)
            val image = BitmapImageBuilder(bitmap).build()
            try {
                timestamp = max(timestamp+1,SystemClock.uptimeMillis())
                val result = face.detectForVideo(image,timestamp)
                if (result.faceLandmarks().isNotEmpty()) {
                    val points = result.faceLandmarks()[0].map { Point((it.x()*size).toDouble(),(it.y()*size).toDouble()) }
                    val connections = config.getJSONObject("geometry_connections")
                    for (key in listOf("LEFT_EYE","RIGHT_EYE","LEFT_EYEBROW","RIGHT_EYEBROW","LIPS")) {
                        val pairs = connections.getJSONArray(key)
                        for (i in 0 until pairs.length()) {
                            val pair = pairs.getJSONArray(i)
                            Imgproc.line(features,points[pair.getInt(0)],points[pair.getInt(1)],Scalar(1.0),max(1,(7f*size/384).roundToInt()))
                        }
                    }
                    for (indices in listOf(intArrayOf(50,101,205,206,216,212,214,192),intArrayOf(280,330,425,426,436,432,434,416))) {
                        val polygon = MatOfPoint(*indices.map { points[it] }.toTypedArray())
                        val hullIndices = MatOfInt(); Imgproc.convexHull(polygon,hullIndices)
                        val hull = MatOfPoint(*hullIndices.toArray().map { polygon.toArray()[it] }.toTypedArray())
                        Imgproc.fillConvexPoly(cheeks,hull,Scalar(1.0)); polygon.release(); hullIndices.release(); hull.release()
                    }
                    status = "detected"
                }
            } finally { image.close() }
        } catch (error: Exception) { status = "detection_error: ${error.javaClass.simpleName}" }
        finally { bitmap.recycle() }
        val a = ByteArray(size*size); val b = ByteArray(size*size); features.get(0,0,a); cheeks.get(0,0,b)
        return Triple(BooleanArray(a.size) { a[it].toInt()!=0 },BooleanArray(b.size) { b[it].toInt()!=0 && a[it].toInt()==0 },status)
    }

    private fun nms(values: FloatArray,angle: FloatArray,size: Int,pool: Mats): FloatArray {
        val score = pool.keep(Mat(size,size,CvType.CV_32F)); score.put(0,0,values)
        val x = FloatArray(values.size); val y = FloatArray(values.size)
        val mapX = pool.keep(Mat(size,size,CvType.CV_32F)); val mapY = pool.keep(Mat(size,size,CvType.CV_32F))
        for (i in x.indices) { x[i] = i%size+cos(angle[i]); y[i] = i/size+sin(angle[i]) }
        mapX.put(0,0,x); mapY.put(0,0,y)
        val plus = pool.keep(); Imgproc.remap(score,plus,mapX,mapY,Imgproc.INTER_LINEAR,Core.BORDER_CONSTANT)
        for (i in x.indices) { x[i] = i%size-cos(angle[i]); y[i] = i/size-sin(angle[i]) }
        mapX.put(0,0,x); mapY.put(0,0,y)
        val minus = pool.keep(); Imgproc.remap(score,minus,mapX,mapY,Imgproc.INTER_LINEAR,Core.BORDER_CONSTANT)
        val a = FloatArray(values.size); val b = FloatArray(values.size); plus.get(0,0,a); minus.get(0,0,b)
        return FloatArray(values.size) { if(values[it]>=a[it] && values[it]>b[it]) values[it] else 0f }
    }

    private fun colors(rgb: Mat,paletteName: String,pool: Mats): ByteArray {
        val bgr = pool.keep(); Imgproc.cvtColor(rgb,bgr,Imgproc.COLOR_RGB2BGR)
        val smoothBgr = pool.keep(); Photo.edgePreservingFilter(bgr,smoothBgr,1,55f,.30f)
        val smooth = pool.keep(); Imgproc.cvtColor(smoothBgr,smooth,Imgproc.COLOR_BGR2RGB)
        val floatRgb = pool.keep(); smooth.convertTo(floatRgb,CvType.CV_32FC3,1.0/255)
        val lab = pool.keep(); Imgproc.cvtColor(floatRgb,lab,Imgproc.COLOR_RGB2Lab)
        val rawPalette = config.getJSONObject("style").getJSONArray("palette_rgb")
        val count = rawPalette.length()
        val palette = ByteArray(count*3) { i ->
            val value = rawPalette.getJSONArray(i/3).getInt(i%3)
            (if(paletteName=="pastel") (value*.85+intArrayOf(255,245,233)[i%3]*.15).toInt() else value).toByte()
        }
        val paletteMat = pool.keep(Mat(1,count,CvType.CV_32FC3)); paletteMat.put(0,0,FloatArray(palette.size) { (palette[it].toInt() and 255)/255f })
        val paletteLab = pool.keep(); Imgproc.cvtColor(paletteMat,paletteLab,Imgproc.COLOR_RGB2Lab)
        val pal = FloatArray(count*3); paletteLab.get(0,0,pal)
        val pixels = FloatArray(rgb.rows()*rgb.cols()*3); lab.get(0,0,pixels)
        val labels = IntArray(pixels.size/3)
        for(i in labels.indices) {
            var distance = Float.POSITIVE_INFINITY
            for(k in 0 until count) {
                var d=0f; for(c in 0..2) { val delta=pixels[i*3+c]-pal[k*3+c]; d+=delta*delta }
                if(d<distance) { distance=d; labels[i]=k }
            }
        }
        val out=ByteArray(pixels.size); val width=rgb.cols(); val height=rgb.rows(); val votes=IntArray(count)
        for(i in labels.indices) {
            votes.fill(0)
            for(dy in -1..1) for(dx in -1..1) {
                val x=(i%width+dx).coerceIn(0,width-1); val y=(i/width+dy).coerceIn(0,height-1)
                votes[labels[y*width+x]]++
            }
            var winner=labels[i]
            for(k in 0 until count) if(votes[k]>=5) { winner=k; break }
            for(c in 0..2) out[i*3+c]=palette[winner*3+c]
        }
        return out
    }
}
