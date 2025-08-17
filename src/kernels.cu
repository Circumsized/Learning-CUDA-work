#include <vector>
#include <stdexcept>
#include <cmath>
#include "../tester/utils.h"

/**
 * @brief 并行快速选择算法实现 - 查找数组中第k大元素
 * 
 * 该实现基于快速选择算法的并行化版本，通过分治策略和并行分区操作，
 * 在GPU上高效查找第k大元素。平均时间复杂度为O(n)，优于排序后选择的O(n log²n)。
 */

/**
 * @brief 分区函数 - 将数组按 pivot 分为大于、等于和小于三部分
 * 
 * @tparam T 数据类型（int或float）
 * @param data 待分区数据
 * @param low 分区起始索引
 * @param high 分区结束索引
 * @param pivot  pivot值
 * @param left 输出参数，大于pivot区域的结束索引+1
 * @param right 输出参数，小于pivot区域的开始索引-1
 */
template <typename T>
__device__ void partition(T* data, int low, int high, T pivot, int& left, int& right) {
    int i = low;
    left = low;
    right = high;
    
    // 三向切分：[low..left-1] > pivot, [left..right] == pivot, [right+1..high] < pivot
    while (i <= right) {
        if (data[i] > pivot) {
            // 当前元素大于pivot，交换到左侧区域
            T temp = data[left];
            data[left] = data[i];
            data[i] = temp;
            left++;
            i++;
        } else if (data[i] < pivot) {
            // 当前元素小于pivot，交换到右侧区域
            T temp = data[i];
            data[i] = data[right];
            data[right] = temp;
            right--;
        } else {
            // 当前元素等于pivot，留在中间区域
            i++;
        }
    }
}

/**
 * @brief 并行快速选择核函数
 * 
 * @tparam T 数据类型（int或float）
 * @param data 输入数组
 * @param n 数组大小
 * @param k 目标排名（0-based）
 * @param result 存储结果的数组
 * @param active 标记当前块是否处于活动状态
 */
template <typename T>
__global__ void quickSelectKernel(T* data, int n, int k, T* result, bool* active) {
    // 每个块处理一个子问题
    int block = blockIdx.x;
    if (!active[block]) return;
    
    // 块内共享内存存储当前子问题的范围
    __shared__ int shared_low, shared_high;
    __shared__ bool found;
    __shared__ T pivot;
    __shared__ int left, right;
    
    // 块内第一个线程初始化共享内存
    if (threadIdx.x == 0) {
        shared_low = block * (n / gridDim.x);
        shared_high = (block == gridDim.x - 1) ? n - 1 : (block + 1) * (n / gridDim.x) - 1;
        found = false;
    }
    __syncthreads();
    
    // 当块处于活动状态且未找到结果时继续处理
    while (active[block] && !found) {
        // 块内第一个线程选择pivot并执行初始分区
        if (threadIdx.x == 0) {
            // 选择当前子区域的中间元素作为pivot
            pivot = data[(shared_low + shared_high) / 2];
            partition(data, shared_low, shared_high, pivot, left, right);
            
            // 检查目标k是否在当前块的范围内
            if (k >= left && k <= right) {
                // 找到目标元素
                *result = data[k];
                found = true;
                
                // 通知所有块停止工作
                for (int b = 0; b < gridDim.x; b++) {
                    active[b] = false;
                }
            } else if (k < left) {
                // 目标在左侧区域，缩小搜索范围
                shared_high = left - 1;
            } else {
                // 目标在右侧区域，缩小搜索范围
                shared_low = right + 1;
            }
            
            // 检查当前子区域是否为空
            if (shared_low > shared_high) {
                active[block] = false;
            }
        }
        __syncthreads();
    }
}

/**
 * @brief Find the k-th largest element in a vector using CUDA.
 *
 * @tparam T Type of elements in the input vector (should support `int` and `float`).
 * @param h_input Host-side input vector.
 * @param k 1-based index of the element to find (e.g., `k=1` returns the largest element).
 * @return T The k-th largest element in `h_input`.
 * @note Must use CUDA kernels for all compute-intensive steps; no significant CPU allowed.
 * @note Library functions that can directly complete a significant part of the work are NOT allowed.
 * @note For invalid cases, return T(-100).
 * @note Handles device memory management (allocate/copy/free) internally. Errors should be thrown.
 */
template <typename T>
T kthLargest(const std::vector<T>& h_input, size_t k) {
    int n = h_input.size();
    
    // 处理无效输入
    if (n == 0 || k <= 0 || k > n) {
        return T(-100);
    }
    
    // 转换为0-based索引
    int k_index = k - 1;
    
    T* d_data;
    T* d_result;
    bool* d_active;
    cudaError_t err;
    
    // 分配设备内存
    err = cudaMalloc(&d_data, n * sizeof(T));
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA内存分配失败 for d_data");
    }
    
    err = cudaMalloc(&d_result, sizeof(T));
    if (err != cudaSuccess) {
        cudaFree(d_data);
        throw std::runtime_error("CUDA内存分配失败 for d_result");
    }
    
    // 分配活动状态数组，用于控制块的执行
    int num_blocks = 32;  // 使用32个块并行处理
    err = cudaMalloc(&d_active, num_blocks * sizeof(bool));
    if (err != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_result);
        throw std::runtime_error("CUDA内存分配失败 for d_active");
    }
    
    // 初始化活动状态：所有块初始为活动
    std::vector<bool> h_active(num_blocks, true);
    
    // 复制数据到设备
    err = cudaMemcpy(d_data, h_input.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_result);
        cudaFree(d_active);
        throw std::runtime_error("数据复制失败 HostToDevice");
    }
    
    err = cudaMemcpy(d_active, h_active.data(), num_blocks * sizeof(bool), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_result);
        cudaFree(d_active);
        throw std::runtime_error("活动状态复制失败 HostToDevice");
    }
    
    // 启动并行快速选择核函数
    dim3 grid(num_blocks);
    dim3 block(256);  // 每个块使用256个线程
    quickSelectKernel<T><<<grid, block>>>(d_data, n, k_index, d_result, d_active);
    
    // 检查核函数启动错误
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_result);
        cudaFree(d_active);
        throw std::runtime_error("快速选择核函数启动失败: " + std::string(cudaGetErrorString(err)));
    }
    
    // 同步设备
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_result);
        cudaFree(d_active);
        throw std::runtime_error("CUDA设备同步失败: " + std::string(cudaGetErrorString(err)));
    }
    
    // 复制结果回主机
    T result;
    err = cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_result);
        cudaFree(d_active);
        throw std::runtime_error("结果复制失败 DeviceToHost");
    }
    
    // 释放设备内存
    cudaFree(d_data);
    cudaFree(d_result);
    cudaFree(d_active);
    
    return result;
}

/**
 * @brief Flash Attention算子实现
 * 
 * 该实现支持因果掩码(Causal Masking)和分组查询注意力(GQA)，
 * 行为与PyTorch的torch.nn.functional.scaled_dot_product_attention保持一致。
 * 通过共享内存优化和内存访问模式优化，实现高效的注意力计算。
 */

/**
 * @brief 数值稳定的softmax函数（设备端）
 * 
 * 实现数值稳定的softmax计算，通过减去最大值防止指数函数溢出。
 * 
 * @param input 输入数组（logits）
 * @param output 输出数组（概率）
 * @param length 数组长度
 */
__device__ void softmax_device(const float* input, float* output, int length) {
    // 找到最大值，用于数值稳定
    float max_val = input[0];
    for (int i = 1; i < length; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    // 计算指数并求和
    float sum = 0.0f;
    for (int i = 0; i < length; ++i) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    // 归一化得到概率
    float inv_sum = 1.0f / (sum + 1e-8f);  // 添加小epsilon防止除零
    for (int i = 0; i < length; ++i) {
        output[i] *= inv_sum;
    }
}

/**
 * @brief Flash Attention核函数
 * 
 * 实现Flash Attention的核心计算，包括QK点积、缩放、掩码、softmax和VO乘积。
 * 
 * @param q 查询张量 [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param k 键张量 [batch_size, src_seq_len, kv_heads, head_dim]
 * @param v 值张量 [batch_size, src_seq_len, kv_heads, head_dim]
 * @param o 输出张量 [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param batch_size 批次大小
 * @param tgt_seq_len 目标序列长度
 * @param src_seq_len 源序列长度
 * @param query_heads 查询头数量
 * @param kv_heads 键/值头数量
 * @param head_dim 头维度
 * @param is_causal 是否应用因果掩码
 * @param scale 缩放因子 (1/√head_dim)
 */
__global__ void flashAttentionKernel(
    const float* __restrict__ q, 
    const float* __restrict__ k, 
    const float* __restrict__ v, 
    float* __restrict__ o,
    int batch_size, 
    int tgt_seq_len, 
    int src_seq_len, 
    int query_heads, 
    int kv_heads, 
    int head_dim, 
    bool is_causal,
    float scale) {
    
    // 计算当前线程处理的batch、query head和target position
    int b = blockIdx.z;  // 批次索引
    int h = blockIdx.y;  // 查询头索引
    int t = threadIdx.y;  // 目标序列位置
    
    // 确保在有效范围内
    if (b >= batch_size || h >= query_heads || t >= tgt_seq_len) {
        return;
    }
    
    // GQA: 查询头到键/值头的映射 (Grouped Query Attention)
    int kvh = (h * kv_heads) / query_heads;
    if (kvh >= kv_heads) kvh = kv_heads - 1;
    
    // 共享内存用于存储scores和probs，避免重复访问全局内存
    extern __shared__ float smem[];
    float* scores = smem;          // 存储QK点积结果 (logits)
    float* probs = smem + src_seq_len;  // 存储softmax结果 (概率)
    
    // 当前query向量的起始索引
    size_t q_idx = ((size_t)b * tgt_seq_len + t) * query_heads * head_dim + h * head_dim;
    
    // 步骤1: 计算Query和Key的点积并应用缩放
    for (int s = threadIdx.x; s < src_seq_len; s += blockDim.x) {
        // 因果掩码: 对于因果注意力，掩盖未来位置 (s > t)
        if (is_causal && s > t) {
            scores[s] = -1e20f;  // 使用一个非常小的值表示负无穷
        } else {
            // 计算点积
            float dot = 0.0f;
            size_t k_idx = ((size_t)b * src_seq_len + s) * kv_heads * head_dim + kvh * head_dim;
            
            for (int d = 0; d < head_dim; ++d) {
                dot += q[q_idx + d] * k[k_idx + d];
            }
            
            // 应用缩放因子 (1/√head_dim)
            scores[s] = dot * scale;
        }
    }
    
    __syncthreads();  // 等待所有线程完成scores计算
    
    // 步骤2: 计算softmax - 仅由每个块的第一个线程执行
    if (threadIdx.x == 0) {
        softmax_device(scores, probs, src_seq_len);
    }
    
    __syncthreads();  // 等待softmax计算完成
    
    // 步骤3: 使用权重对Value进行加权求和
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < src_seq_len; ++s) {
            size_t v_idx = ((size_t)b * src_seq_len + s) * kv_heads * head_dim + kvh * head_dim + d;
            acc += probs[s] * v[v_idx];
        }
        
        // 将结果写入输出张量
        size_t o_idx = ((size_t)b * tgt_seq_len + t) * query_heads * head_dim + h * head_dim + d;
        o[o_idx] = acc;
    }
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k, const std::vector<T>& h_v, std::vector<T>& h_o, 
                   int batch_size, int target_seq_len, int src_seq_len, int query_heads, int kv_heads, int head_dim, bool is_causal) {
    // 仅支持float类型
    static_assert(std::is_same<T, float>::value, "flashAttention仅支持float类型");
    
    cudaError_t err;
    
    // 检查输入有效性
    if (head_dim <= 0 || query_heads <= 0 || kv_heads <= 0 || 
        batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0) {
        throw std::invalid_argument("无效的输入参数");
    }
    
    // 检查输入大小是否匹配
    size_t q_size = (size_t)batch_size * target_seq_len * query_heads * head_dim;
    size_t kv_size = (size_t)batch_size * src_seq_len * kv_heads * head_dim;
    
    if (h_q.size() != q_size || h_k.size() != kv_size || h_v.size() != kv_size) {
        throw std::invalid_argument("输入张量大小不匹配");
    }
    
    // 调整输出大小
    h_o.resize(q_size);
    
    // 分配设备内存
    float *d_q, *d_k, *d_v, *d_o;
    err = cudaMalloc(&d_q, q_size * sizeof(float));
    err = err ?: cudaMalloc(&d_k, kv_size * sizeof(float));
    err = err ?: cudaMalloc(&d_v, kv_size * sizeof(float));
    err = err ?: cudaMalloc(&d_o, q_size * sizeof(float));
    
    if (err != cudaSuccess) {
        if (d_q) cudaFree(d_q);
        if (d_k) cudaFree(d_k);
        if (d_v) cudaFree(d_v);
        if (d_o) cudaFree(d_o);
        throw std::runtime_error("CUDA内存分配失败: " + std::string(cudaGetErrorString(err)));
    }
    
    // 复制数据到设备
    err = cudaMemcpy(d_q, h_q.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    err = err ?: cudaMemcpy(d_k, h_k.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    err = err ?: cudaMemcpy(d_v, h_v.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    
    if (err != cudaSuccess) {
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
        throw std::runtime_error("数据复制失败 HostToDevice: " + std::string(cudaGetErrorString(err)));
    }
    
    // 计算缩放因子 (1/√head_dim)
    float scale = 1.0f / std::sqrt((float)head_dim);
    
    // 配置核函数参数
    dim3 blockDim(32, 8);  // 32x8线程块，优化内存访问模式
    dim3 gridDim(1, query_heads, batch_size);  // 网格维度: [1, query_heads, batch_size]
    
    // 共享内存大小: scores(src_seq_len) + probs(src_seq_len)
    size_t shared_mem_size = 2 * src_seq_len * sizeof(float);
    
    // 启动核函数
    flashAttentionKernel<<<gridDim, blockDim, shared_mem_size>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim, is_causal, scale
    );
    
    // 检查核函数启动错误
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
        throw std::runtime_error("FlashAttention核函数启动失败: " + std::string(cudaGetErrorString(err)));
    }
    
    // 同步设备
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
        throw std::runtime_error("CUDA设备同步失败: " + std::string(cudaGetErrorString(err)));
    }
    
    // 将结果复制回主机
    err = cudaMemcpy(h_o.data(), d_o, q_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
        throw std::runtime_error("结果复制失败 DeviceToHost: " + std::string(cudaGetErrorString(err)));
    }
    
    // 释放设备内存
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int kthLargest(const std::vector<int>&, size_t);
template float kthLargest(const std::vector<float>&, size_t);
template void flashAttention(const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, std::vector<float>&, 
                           int, int, int, int, int, int, bool);
