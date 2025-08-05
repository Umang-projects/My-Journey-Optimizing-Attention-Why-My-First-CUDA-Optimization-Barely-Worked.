# My-Journey-Optimizing-Attention-Why-My-First-CUDA-Optimization-Barely-Worked.
How a 3x kernel speedup resulted in a tiny 6% overall gain, and the profiler that revealed why.
categories: [CUDA, Deep Learning]
tags: [optimization, performance, gpu, profiling, bottleneck, attention, transformer, tiling, shared memory]

---

## 🔍 Attention Latency Comparison

| Version | Matmul Kernel Time | Overall End-to-End Time | Speedup |
| :-------------------------- | :------------------ | :---------------------- | :------- |
| **Naive (global memory)** | \~3.25 ms | **38.2 ms** | 1.00x |
| **Tiled (shared memory)** | \~1.14 ms | **35.9 ms** | **~1.06x** |

> 😮 Despite a 3× faster matmul kernel, the overall gain was only ~6%!

---

## 🤔 What Was the Goal?

I wanted to write a custom CUDA implementation of the attention mechanism used in transformers.

The pipeline I implemented looks like this:

```
Q × Kᵗ → Scale → Softmax → Multiply with V
```

The idea was simple:

- First write a **naive** version using only global memory.
- Then write an **optimized** version using shared memory and tiling.
- Compare performance and understand what’s really slowing things down.

Let’s walk through what I actually built in code.

---

## 🔧 Step 1: The Naive Implementation

### ✅ Kernel 1: `matmul_naive_kernel`

This kernel computes the attention scores:

```cpp
__global__ void matmul_naive_kernel(const float* q, const float* k, float* scores, int seq_len, int d_k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < seq_len && col < seq_len) {
        float sum = 0.0f;
        // Every read/write here is a slow trip to global memory (DRAM)
        for (int i = 0; i < d_k; ++i) {
            sum += q[row * d_k + i] * k[col * d_k + i];
        }
        scores[row * seq_len + col] = sum;
    }
}
```

Each thread computes a single element of the `scores` matrix by performing a dot product between a row of Q and a column of K. No shared memory. Every access is from slow global memory.

---

### ✅ Kernel 2: `scale_softmax_gemm_kernel`

This kernel does everything else:

> Scale the scores → apply softmax → multiply with V

```cpp
__global__ void scale_softmax_gemm_kernel(float* scores, const float* v, float* output, int seq_len, int d_k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < seq_len) {
        float scale_factor = 1.0f / sqrtf((float)d_k);
        float max_val = -1e20f;
        for (int i = 0; i < seq_len; ++i) {
            scores[row * seq_len + i] *= scale_factor;
            max_val = fmaxf(max_val, scores[row * seq_len + i]);
        }

        // PROBLEM: Each thread runs three full, sequential loops over SEQ_LEN
// This is very poor parallelism.
        float exp_sum = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            float val = expf(scores[row * seq_len + i] - max_val);
            scores[row * seq_len + i] = val;
            exp_sum += val;
        }

        for (int i = 0; i < seq_len; ++i) {
            scores[row * seq_len + i] /= exp_sum;
        }

        for (int j = 0; j < d_k; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < seq_len; ++i) {
                sum += scores[row * seq_len + i] * v[i * d_k + j];
            }
            output[row * d_k + j] = sum;
        }
    }
}
```

Each thread handles one full row of the scores matrix — heavy sequential computation and memory access.

---

## 🚀 Step 2: Optimized Matmul Using Tiling

### ✅ `matmul_tiled_kernel`

Shared memory tiling used to load blocks of Q and K, compute partial sums, and reduce global memory access.

```cpp
__global__ void matmul_tiled_kernel(const float* q, const float* k, float* scores, int seq_len, int d_k) {
    __shared__ float q_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float k_tile[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float sum = 0.0f;

    for (int t = 0; t < (d_k + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < seq_len && (t * TILE_WIDTH + tx) < d_k) {
            q_tile[ty][tx] = q[row * d_k + (t * TILE_WIDTH + tx)];
        } else {
            q_tile[ty][tx] = 0.0f;
        }

        if (col < seq_len && (t * TILE_WIDTH + ty) < d_k) {
            k_tile[tx][ty] = k[col * d_k + (t * TILE_WIDTH + ty)];
        } else {
            k_tile[tx][ty] = 0.0f;
        }

        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += q_tile[ty][i] * k_tile[tx][i];
        }

        __syncthreads();
    }

    if (row < seq_len && col < seq_len) {
        scores[row * seq_len + col] = sum;
    }
}
```

- Brought matmul time down from \~3.25 ms to \~1.14 ms
- But overall latency improvement was only \~6%

---

## 📊 Profiling Results

```
93.91% → scale_softmax_gemm_kernel
 4.39% → matmul_naive_kernel
 1.54% → matmul_tiled_kernel
```

Main issue: `scale_softmax_gemm_kernel` dominated runtime.

---

## ❗ Bottleneck Analysis

### 1. Poor Parallelism

- Each thread processes a full row sequentially.

### 2. Memory-Bound

- Scores written and read from global memory multiple times.


## 🧠 Learnings

- Always profile before optimizing.
- Shared memory and tiling help, but only if you target the actual bottleneck.
- CUDA optimization is less about computation and more about data movement.
---

## 💻 Code & Next Steps
- Implement fused FlashAttention-style kernel
- Explore warp-level parallelism for softmax
- Learn more.

Thanks for reading!
