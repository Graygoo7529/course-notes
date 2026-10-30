package cn.coursenotes.cartoon

import org.junit.Assert.*
import org.junit.Test

class CartoonMathTest {
    @Test fun featureDifferenceMatchesReferencePyTorch() {
        val lines = javaClass.getResourceAsStream("/feature_difference.txt")!!.bufferedReader().readLines()
        for (i in lines.indices step 3) {
            val shape = lines[i].split(",").map { it.toInt() }
            val input = lines[i+1].split(",").map { it.toFloat() }.toFloatArray()
            val expected = lines[i+2].split(",").map { it.toFloat() }.toFloatArray()
            assertArrayEquals(expected, CartoonMath.featureDifference(input, shape[0], shape[1], shape[2]), 2e-6f)
        }
    }

    @Test fun normalizationIgnoresPaddingAndHandlesFlatImages() {
        val values = floatArrayOf(-1000f, 0f, 1f, 2f, 3f, 1000f)
        val valid = booleanArrayOf(false, true, true, true, true, false)
        // NumPy quantile([0,1,2,3], [.1,.95]) = [.3, 2.85].
        assertArrayEquals(floatArrayOf(0f, 0f, .7f/2.55f, 1.7f/2.55f, 1f, 0f),
            CartoonMath.normalize(values, valid), 1e-6f)
        assertArrayEquals(FloatArray(6), CartoonMath.normalize(FloatArray(6) { 2f }, valid), 0f)
    }

    @Test fun thinningPreservesConnectedStrokeAndIsIdempotent() {
        val width = 15
        val input = BooleanArray(width*width) { it%width in 5..9 && it/width in 2..12 }
        val thin = CartoonMath.thin(input, width)
        assertTrue(thin.count { it } in 5 until input.count { it })
        assertTrue(thin.indices.all { !thin[it] || input[it] })
        assertArrayEquals(thin, CartoonMath.thin(thin, width))
        val reached = mutableSetOf(thin.indexOfFirst { it })
        val queue = java.util.ArrayDeque<Int>(); queue.addAll(reached)
        while (queue.isNotEmpty()) {
            val i = queue.removeFirst()
            for (dy in -1..1) for (dx in -1..1) {
                val j = i+dy*width+dx
                if (j in thin.indices && thin[j] && reached.add(j)) queue.add(j)
            }
        }
        assertEquals(thin.count { it }, reached.size)
    }
}
