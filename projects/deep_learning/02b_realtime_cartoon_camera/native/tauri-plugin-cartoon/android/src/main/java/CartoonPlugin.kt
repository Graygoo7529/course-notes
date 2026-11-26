package cn.coursenotes.cartoon

import android.app.Activity
import android.content.ContentValues
import android.provider.MediaStore
import android.util.Base64
import androidx.appcompat.app.AppCompatActivity
import app.tauri.annotation.Command
import app.tauri.annotation.InvokeArg
import app.tauri.annotation.TauriPlugin
import app.tauri.plugin.Invoke
import app.tauri.plugin.JSObject
import app.tauri.plugin.Plugin
import java.util.concurrent.Executors

@InvokeArg
class CameraArgs {
    var sessionId: String = ""
    var image: String = ""
    var meta: String = "{}"
    var filename: String = ""
}

@TauriPlugin
class CartoonPlugin(private val activity: Activity) : Plugin(activity) {
    private val executor = Executors.newSingleThreadExecutor()
    private var engine: CartoonEngine? = null
    private var session = ""
    private var lastFrame = -1L
    override fun onDestroy(activity: AppCompatActivity) {
        executor.execute { engine?.close(); engine = null; session = "" }
        executor.shutdown()
        super.onDestroy(activity)
    }
    private fun execute(invoke: Invoke, action: (CameraArgs) -> JSObject) {
        val args = invoke.parseArgs(CameraArgs::class.java)
        executor.execute {
            try { invoke.resolve(action(args)) }
            catch (error: Throwable) { invoke.reject(error.message ?: error.javaClass.simpleName) }
        }
    }

    @Command
    fun initialize(invoke: Invoke) = execute(invoke) { args ->
        engine?.close()
        engine = null
        val start = System.nanoTime()
        val next = CartoonEngine(activity)
        engine = next; session = args.sessionId; lastFrame = -1
        JSObject().put("backend", "ONNX Runtime Android CPU").put("model", "MobileNetV3-Large")
            .put("inputSize", 256).put("device", "CPU").put("coldStartMs", (System.nanoTime()-start)/1e6)
    }

    @Command
    fun processFrame(invoke: Invoke) = execute(invoke) { args ->
        require(args.sessionId == session && session.isNotEmpty()) { "会话已经结束" }
        val meta = JSObject(args.meta)
        val number = meta.getLong("frameId")
        require(number > lastFrame) { "过期的帧" }; lastFrame = number
        require(args.image.length <= 32 * 1024 * 1024) { "图片过大" }
        val bytes = Base64.decode(args.image, Base64.NO_WRAP)
        val rendered = requireNotNull(engine).render(bytes, meta)
        JSObject().put("meta", rendered.first).put("image", Base64.encodeToString(rendered.second, Base64.NO_WRAP))
    }

    @Command
    fun reset(invoke: Invoke) = execute(invoke) { args ->
        if (args.sessionId == session) engine?.reset()
        JSObject()
    }

    @Command
    fun stop(invoke: Invoke) = execute(invoke) { args ->
        if (args.sessionId == session) { engine?.close(); engine = null; session = "" }
        JSObject()
    }

    @Command
    fun savePng(invoke: Invoke) = execute(invoke) { args ->
        require(args.image.length <= 32 * 1024 * 1024) { "图片过大" }
        val bytes = Base64.decode(args.image, Base64.NO_WRAP)
        require(bytes.take(8) == listOf(137,80,78,71,13,10,26,10).map { it.toByte() }) { "不是 PNG" }
        require(args.filename.matches(Regex("cartoon-[0-9]+\\.png"))) { "无效文件名" }
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, args.filename)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/CartoonCamera")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val resolver = activity.contentResolver
        val uri = requireNotNull(resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values))
        try {
            requireNotNull(resolver.openOutputStream(uri)).use { it.write(bytes) }
            values.clear(); values.put(MediaStore.Images.Media.IS_PENDING, 0)
            resolver.update(uri, values, null, null)
        } catch (error: Exception) { resolver.delete(uri, null, null); throw error }
        JSObject().put("path", "相册 / CartoonCamera / ${args.filename}")
    }
}
