package cn.coursenotes.cartoon

import kotlin.math.*

/** Platform independent kernels corresponding to the reference NumPy/PyTorch path. */
object CartoonMath {
    fun quantile(values: FloatArray,q: Float): Float {
        if(values.isEmpty()) return 0f
        val sorted=values.copyOf(); sorted.sort()
        val position=q.coerceIn(0f,1f)*(sorted.size-1)
        val low=floor(position).toInt(); val high=ceil(position).toInt()
        return sorted[low]+(position-low)*(sorted[high]-sorted[low])
    }
    fun normalize(values: FloatArray,valid: BooleanArray): FloatArray {
        val samples=values.filterIndexed { i,_ -> valid[i] }.toFloatArray()
        if(samples.isEmpty()) return FloatArray(values.size)
        val lo=quantile(samples,.1f); val hi=quantile(samples,.95f)
        if(hi-lo<1e-6f) return FloatArray(values.size)
        return FloatArray(values.size) { if(valid[it]) ((values[it]-lo)/(hi-lo)).coerceIn(0f,1f) else 0f }
    }
    fun featureDifference(features: FloatArray,channels: Int,height: Int,width: Int): FloatArray {
        val plane=height*width; val unit=features.copyOf()
        for(i in 0 until plane) {
            var norm=0f; for(c in 0 until channels) norm+=features[c*plane+i]*features[c*plane+i]
            norm=max(1e-6f,sqrt(norm))
            for(c in 0 until channels) unit[c*plane+i]/=norm
        }
        return FloatArray(plane) { i ->
            val x=i%width; val y=i/width
            val a=y*width+min(x,width-2); val b=min(y,height-2)*width+x
            var sum=0f
            for(c in 0 until channels) {
                val offset=c*plane
                val dx=unit[offset+a+1]-unit[offset+a]; val dy=unit[offset+b+width]-unit[offset+b]
                sum+=dx*dx+dy*dy
            }
            sqrt(sum)
        }
    }
    fun thin(input: BooleanArray,width: Int): BooleanArray {
        val image=input.copyOf(); val height=image.size/width
        val remove=IntArray(image.size)
        var changed=true; var iterations=0
        while(changed && iterations++<64) {
            changed=false
            for(step in 0..1) {
                var count=0
                for(y in 1 until height-1) for(x in 1 until width-1) {
                    val i=y*width+x; if(!image[i]) continue
                    val p=booleanArrayOf(image[i-width],image[i-width+1],image[i+1],image[i+width+1],
                        image[i+width],image[i+width-1],image[i-1],image[i-width-1])
                    val neighbors=p.count { it }; if(neighbors !in 2..6) continue
                    var transitions=0; for(j in 0..7) if(!p[j] && p[(j+1)%8]) transitions++
                    if(transitions!=1) continue
                    val a=if(step==0) p[0]&&p[2]&&p[4] else p[0]&&p[2]&&p[6]
                    val b=if(step==0) p[2]&&p[4]&&p[6] else p[0]&&p[4]&&p[6]
                    if(!a && !b) remove[count++]=i
                }
                for(k in 0 until count) image[remove[k]]=false
                if(count>0) changed=true
            }
        }
        return image
    }
}
