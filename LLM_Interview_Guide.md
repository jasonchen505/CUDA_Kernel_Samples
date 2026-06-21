# LLM 算法实习面试准备指南 — 基于 CUDA_Kernel_Samples 项目

> 本文档面向：找 LLM 算法实习的 MS 在读学生
> 核心目标：将 CUDA infra 知识转化为 LLM 算法面试中的差异化竞争力

---

## 一、项目速览与学习路径

### 1.1 项目结构

```
CUDA_Kernel_Samples/
├── elementwise/    # 逐元素算子 (add, relu, sigmoid) — 最基础
├── gemv/           # 矩阵乘向量 — warp级归约
├── reduce/         # 归约算子 (sum, max, softmax) — 面试高频
├── sgemm/          # 矩阵乘优化 (7个版本) — 核心重点
└── transpose/      # 矩阵转置 — bank conflict
```

### 1.2 推荐学习顺序

| 阶段 | 内容 | 时间 | 目标 |
|------|------|------|------|
| 1 | elementwise/add.cu | 0.5h | 理解 grid/block/线程索引 |
| 2 | reduce/sum/sum.cu | 1h | 掌握 shared memory + warp shuffle |
| 3 | reduce/softmax_matrix/softmax_matrix.cu | 1h | 理解行归约模式 |
| 4 | sgemm/kernel1 → kernel3 | 2h | 掌握 shared memory tiling + thread tile |
| 5 | sgemm/kernel7 (double buffer) | 1h | 理解延迟隐藏原理 |
| 6 | transpose/transpose.cu | 0.5h | 理解 bank conflict |

---

## 二、LLM 算法面试中 CUDA 知识的定位

### 2.1 为什么 LLM 算法岗需要懂 CUDA？

面试官的底层逻辑：

```
LLM 算法岗 ≠ 只会调 API
                ↓
需要理解：为什么这个模型慢？瓶颈在哪？怎么优化？
                ↓
需要知道：GPU 是怎么算的？访存和计算的关系？
                ↓
CUDA 知识 = 理解 LLM 推理/训练性能瓶颈的基础
```

### 2.2 面试中 CUDA 知识的切入场景

| 场景 | 面试官可能的问题 | 你需要展示的知识 |
|------|-----------------|-----------------|
| LLM 推理优化 | "vLLM 为什么快？PagedAttention 原理？" | shared memory、访存优化 |
| 模型训练加速 | "DeepSpeed ZeRO 怎么减少显存？" | GPU 内存层次、通信开销 |
| 算子性能分析 | "为什么你的自定义 kernel 慢？" | 计算访存比、bank conflict |
| 后训练/RLHF | "reward model 的 softmax 怎么实现？" | reduce 优化、warp shuffle |
| 量化部署 | "INT8 量化对 GPU 计算有什么影响？" | 向量化访存、精度-性能权衡 |

---

## 三、核心知识点详解（面试必备）

### 3.1 GPU 内存层次（必考）

```
寄存器 (Register)     ← 最快，每个线程私有，~1 cycle
    ↓
共享内存 (Shared Memory) ← block 内共享，~5 cycles，48-164 KB/SM
    ↓
L1 Cache              ← SM 级别
    ↓
全局内存 (Global Memory) ← 所有线程可见，~200-400 cycles，GB 级
    ↓
主机内存 (Host Memory)   ← CPU 端，通过 PCIe 传输
```

**面试关联点**：
- vLLM 的 PagedAttention 本质上是把 KV cache 从全局内存的"碎片化"管理变为"分页"管理
- LoRA 之所以高效，是因为低秩矩阵可以放在更快的内存层级
- FlashAttention 的核心思想：减少对全局内存（HBM）的访问次数

### 3.2 计算访存比（核心概念）

```
计算访存比 = 每秒计算量(FLOPS) / 每秒访存量(GB/s)
```

| Kernel | 计算访存比 | 说明 |
|--------|-----------|------|
| Elementwise | 1:2 | 每次计算读2个数，写1个数 |
| Naive SGEMM | 1:2 | 每次 FMA 读 A 和 B 各一个元素 |
| 优化后 SGEMM (Thread Tile 8x8) | 64:16 = 4:1 | 大幅提高计算密度 |

**面试关联点**：
- LLM 推理是 **访存密集型**（memory-bound），不是计算密集型
- 解释为什么 batch size 越大，GPU 利用率越高：更多计算复用同一份权重
- 量化（INT8/INT4）的本质：减少访存量，提高计算访存比

### 3.3 Shared Memory Tiling（SGEMM 核心优化）

```
Naive: 每个元素计算需要从 Global Memory 读 2K 次
Tiling: 每个元素计算只需要从 Global Memory 读 2K/BN 次 (BN=32 → 1/32)
```

**面试关联点**：
- FlashAttention 的 tiling 策略与此完全一致
- 面试官可能问："FlashAttention 为什么能减少显存？"
  - 答：通过 tiling，不需要存储完整的 N×N attention matrix，而是分块计算

### 3.4 Warp Shuffle（归约优化）

```cpp
// 关键代码
for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
}
```

**面试关联点**：
- Softmax 实现中的 max/sum 归约
- Cross-entropy loss 中的 log-sum-exp
- 分布式训练中的 all-reduce 操作

### 3.5 双缓冲/延迟隐藏（高级概念）

```
传统: [加载数据] → [计算] → [加载数据] → [计算]
双缓冲: [加载数据A] → [计算A + 加载数据B] → [计算B + 加载数据C] ...
```

**面试关联点**：
- CUDA Stream 的本质：让不同的 kernel 重叠执行
- 流水线并行（Pipeline Parallelism）的硬件基础
- 为什么 DeepSpeed 的通信-计算重叠能加速训练

---

## 四、LLM 算法面试高频问题 & 基于本项目的回答框架

### 4.1 Attention 机制相关

**Q: 解释 FlashAttention 的核心思想**

回答框架：
1. **问题定义**：标准 Attention 需要存储 N×N 的 attention matrix，显存 O(N²)
2. **核心思想**：借鉴 SGEMM 的 tiling 策略，分块计算 attention
3. **具体实现**：
   - 将 Q, K, V 分成小块，每块放入 SRAM（类似 shared memory）
   - 在 SRAM 中完成 softmax 和乘法，避免写回 HBM
   - 使用 online softmax（类似我们 softmax_matrix 中的 max-sum 两步法）
4. **效果**：显存从 O(N²) 降到 O(N)，速度提升 2-4 倍

**深挖点**：
- "online softmax 怎么实现？" → 对应 reduce/softmax 中的 max_kernel + sum_kernel
- "为什么不能直接用 __shfl_xor_sync 做全局 softmax？" → warp 内同步，跨 warp 需要 shared memory

### 4.2 KV Cache 相关

**Q: vLLM 的 PagedAttention 解决了什么问题？**

回答框架：
1. **传统问题**：KV cache 连续分配，导致内存碎片化和浪费
2. **PagedAttention 原理**：
   - 借鉴操作系统虚拟内存的分页机制
   - 将 KV cache 分成固定大小的 page（block）
   - 通过 page table 映射逻辑地址到物理地址
3. **与 CUDA 的关系**：
   - 类似 shared memory 的分块管理
   - 减少内存碎片 = 提高 GPU 内存利用率

### 4.3 模型量化相关

**Q: INT8 量化为什么能加速推理？**

回答框架：
1. **访存量减少**：INT8 比 FP16 少一半字节 → 访存带宽翻倍
2. **计算量不变**：GPU 的 INT8 Tensor Core 吞吐量是 FP16 的 2 倍
3. **计算访存比提高**：从 memory-bound 变得更 compute-bound
4. **本项目关联**：float4 向量化访存的原理（减少访存指令数量）

### 4.4 分布式训练相关

**Q: DeepSpeed ZeRO 的三个阶段分别做了什么？**

回答框架：
```
ZeRO-1: 优化器状态分片 (每张卡只存 1/N 的 optimizer states)
ZeRO-2: + 梯度分片 (每张卡只存 1/N 的 gradients)
ZeRO-3: + 参数分片 (每张卡只存 1/N 的 parameters)
```

**与 CUDA 的关联**：
- 通信开销：all-reduce / all-gather 操作，类似 warp shuffle 的跨线程通信
- 计算-通信重叠：类似双缓冲的思想

### 4.5 后训练（RLHF/DPO）相关

**Q: RLHF 中 reward model 的 softmax 怎么高效实现？**

回答框架：
1. **问题**：reward 分数需要 softmax 归一化
2. **实现**：
   - 类似 softmax_matrix 的行归约模式
   - 每个 warp 处理一个样本的 reward 分数
   - 使用 __shfl_down_sync 做 max 和 sum 归约
3. **优化点**：
   - 如果 reward 维度 < 32，一个 warp 足够
   - 如果 reward 维度 > 32，需要 shared memory 中转

---

## 五、面试中如何介绍这个项目

### 5.1 项目介绍模板（1-2 分钟）

> "我系统学习了 CUDA 算子优化，实现了一个从 naive 到 near-peak performance 的优化框架。
>
> 核心模块是 SGEMM 的 7 个优化版本：从 naive 的全局内存访问，到 shared memory tiling 减少访存，再到 thread tile 提高计算访存比，最后用双缓冲实现计算和访存的重叠。
>
> 这些优化思想和 LLM 推理优化高度相关：FlashAttention 的 tiling 策略、KV cache 的分页管理、量化的访存优化，本质上都是在优化计算访存比和内存层次的利用。
>
> 我不仅会写 CUDA kernel，更重要的是理解了 GPU 的性能瓶颈在哪，这帮助我在算法层面做出更好的设计决策。"

### 5.2 面试官可能的追问 & 应对

| 追问 | 应对策略 |
|------|---------|
| "你这个项目的性能达到 cuBLAS 的多少？" | "SGEMM 优化版本在 2560×2560 矩阵上达到 cuBLAS 的 99.7%，约 9300 GFLOPS" |
| "为什么不用 Triton？" | "手写 CUDA 帮助我理解底层原理，Triton 是更高层的抽象，但理解底层才能做出更好的优化决策" |
| "这对 LLM 算法有什么用？" | "理解底层瓶颈，才能在算法设计时考虑效率。比如设计 attention 机制时知道 N² 的显存开销意味着什么" |

---

## 六、LLM 算法面试深挖点（基于本项目知识）

### 6.1 Softmax 的数值稳定性

**项目代码**：`reduce/softmax/softmax.cu` 中的 v1 vs v2

```cpp
// v1: 不稳定版本 — 直接 exp，可能溢出
output[idx] = expf(input[idx]) / sum;

// v2: 稳定版本 — 减去最大值
output[idx] = expf(input[idx] - max_val) / sum;
```

**面试深挖**：
- "为什么 LLM 的 attention 计算要用 causal mask？"
  - 答：防止看到未来信息，同时 masked 位置设为 -∞，softmax 后趋近于 0
- "float16 下 softmax 有什么问题？"
  - 答：动态范围小，更容易溢出，需要 online softmax 或 logits capping

### 6.2 内存对齐与向量化

**项目代码**：`elementwise/add.cu` 中的 float4

```cpp
#define FLOAT4(value) *(float4*)(&(value))
// 要求：指针必须 16 字节对齐
```

**面试深挖**：
- "为什么 LLM 推理框架要求 tensor 对齐？"
  - 答：Tensor Core 需要特定的矩阵维度对齐（如 8 的倍数），不对齐会回退到 CUDA Core
- "FlashAttention 对 Q/K/V 的 shape 有什么要求？"
  - 答：head_dim 通常是 64/128，需要是 8 的倍数以适配 Tensor Core

### 6.3 Warp 级编程

**项目代码**：`reduce/sum/sum.cu` 中的 warp shuffle

```cpp
// Warp 内 32 个线程同步通信，无需 shared memory
val += __shfl_down_sync(0xFFFFFFFF, val, offset);
```

**面试深挖**：
- "Transformer 中 LayerNorm 怎么高效实现？"
  - 答：需要计算 mean 和 variance，类似 reduce 操作，用 warp shuffle 做归约
- "为什么 GroupNorm 比 LayerNorm 慢？"
  - 答：GroupNorm 的分组不一定是 warp 大小的倍数，导致 warp 内归约不规整

### 6.4 Bank Conflict

**项目代码**：`transpose/transpose.cu` 中的 padding 和 swizzling

```cpp
// Padding: 增加一列避免 bank conflict
__shared__ float s_mem[BLOCK_SIZE][BLOCK_SIZE + 1];

// Swizzling: XOR 打散访问模式
s_mem[threadIdx.y][threadIdx.x ^ threadIdx.y] = ...;
```

**面试深挖**：
- "FlashAttention 中有什么 bank conflict？"
  - 答：Q/K/V 在 shared memory 中的布局需要精心设计，避免读取时的 bank conflict
- "如何 profile 一个 kernel 的 bank conflict？"
  - 答：用 `ncu --metrics shared_bank_conflict` 或 `nvprof --metrics`

### 6.5 Occupancy 与寄存器压力

**项目知识**：float4 会增加寄存器使用，降低 occupancy

**面试深挖**：
- "为什么 LLM 推理的 batch size 越大，吞吐量越高？"
  - 答：更多线程 → 更高 occupancy → 更好地隐藏访存延迟
- "但 batch size 超过某个值后为什么会变慢？"
  - 答：KV cache 显存不足 → 需要 swap → 额外的 PCIe 传输开销

---

## 七、面试模拟题（基于项目知识）

### 题目 1：Softmax 实现

**问题**：请实现一个 GPU 上的 softmax kernel，输入是 N 维向量，输出是 N 维向量。

**参考答案**（基于 reduce/softmax/softmax.cu）：

```cpp
__global__ void softmax(float* input, float* output, int N) {
    __shared__ float s_max, s_sum;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    
    // Step 1: 求 max (warp shuffle)
    float val = (idx < N) ? input[idx] : -FLT_MAX;
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    if (laneId == 0) s_max = val;  // 简化版，实际需要 shared memory 中转
    __syncthreads();
    
    // Step 2: 求 sum (warp shuffle)
    val = (idx < N) ? expf(input[idx] - s_max) : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    if (laneId == 0) s_sum = val;
    __syncthreads();
    
    // Step 3: 计算 softmax
    if (idx < N) output[idx] = expf(input[idx] - s_max) / s_sum;
}
```

### 题目 2：FlashAttention 原理

**问题**：解释 FlashAttention 如何减少显存使用。

**参考答案**：
1. 标准 Attention 需要存储 N×N 的 attention matrix → O(N²) 星存
2. FlashAttention 使用 tiling：将 Q/K/V 分成小块，每块在 SRAM 中计算
3. 使用 online softmax：维护 running max 和 running sum，避免两遍扫描
4. 显存复杂度降为 O(N)，速度提升 2-4 倍

### 题目 3：LLM 推理瓶颈分析

**问题**：一个 7B 参数的 LLM，batch size = 1 时推理很慢，如何分析瓶颈？

**参考答案**：
1. **Profile**：用 nsys 或 ncu 查看 kernel 执行时间分布
2. **判断类型**：
   - 如果是 memory-bound → 优化访存（量化、减少 KV cache）
   - 如果是 compute-bound → 优化计算（kernel fusion、Tensor Core 利用率）
3. **具体分析**：
   - batch size = 1 时，矩阵乘法退化为 GEMV → memory-bound
   - 权重 14GB (FP16)，需要从 HBM 读取 → 瓶颈在访存
4. **优化方向**：
   - 量化 (INT8/INT4) → 减少访存量
   - 连续批处理 (continuous batching) → 提高 batch size
   - KV cache 优化 (PagedAttention) → 减少显存碎片

---

## 八、LLM 算法岗 vs CUDA 工程岗的面试差异

| 维度 | LLM 算法岗 | CUDA 工程岗 |
|------|-----------|------------|
| CUDA 深度 | 理解原理即可，不手写 | 需要手写优化 kernel |
| 关注点 | 算法设计、模型效果 | 性能优化、硬件利用 |
| 面试重点 | Attention 变体、训练策略 | 访存优化、bank conflict |
| 项目展示 | 强调"理解瓶颈" | 强调"性能数据" |

**你的定位**：
- 不是 CUDA 工程师，而是 **懂底层的算法工程师**
- 强调：理解 CUDA 优化 → 能设计更高效的算法 → 能和 infra 团队高效沟通

---

## 九、常见 LLM 面试问题清单（按主题分类）

### 9.1 Attention 机制
1. 标准 Attention 的时间/空间复杂度？ → O(N²)
2. FlashAttention 的核心思想？ → tiling + online softmax
3. Multi-Head Attention vs Multi-Query Attention？ → KV cache 节省
4. Causal mask 怎么实现？为什么要 mask？ → 防止看到未来信息

### 9.2 位置编码
1. RoPE 的原理？为什么比绝对位置编码好？ → 旋转矩阵，相对位置信息
2. 为什么 RoPE 不能直接外推？ → 高频分量溢出
3. NTK-aware scaling / YaRN 的思路？ → 调整频率基数

### 9.3 训练优化
1. LoRA 的原理？为什么高效？ → 低秩分解，减少可训练参数
2. DeepSpeed ZeRO 的三个阶段？ → 优化器/梯度/参数分片
3. 混合精度训练（FP16/BF16）的注意事项？ → 溢出/下溢，loss scaling

### 9.4 推理优化
1. KV Cache 的作用？显存占用怎么算？ → 避免重复计算，2×layers×hidden×seq_len
2. 连续批处理（continuous batching）是什么？ → 动态插入新请求
3. 量化方法（GPTQ/AWQ/GGUF）的区别？ → 校准数据、量化粒度

### 9.5 RLHF/DPO
1. RLHF 的三个阶段？ → SFT → Reward Model → PPO
2. DPO 为什么比 PPO 简单？ → 直接优化偏好，不需要 reward model
3. Reward hacking 是什么？怎么解决？ → 过度优化 reward signal

---

## 十、总结：CUDA 知识 → LLM 算法竞争力

```
CUDA 知识                  →  LLM 算法面试优势
─────────────────────────────────────────────────
内存层次 (Register/SM/HBM)  →  理解 KV cache 管理、FlashAttention
计算访存比                  →  理解为什么量化能加速、batch size 的影响
Tiling 策略                 →  理解 FlashAttention 的核心思想
Warp 归约                   →  理解 Softmax/LayerNorm 实现
Bank Conflict              →  理解 Attention 计算的优化空间
双缓冲/延迟隐藏             →  理解流水线并行、通信-计算重叠
```

**核心信息**：
> 你不是 CUDA 工程师，但你理解 GPU 是怎么工作的。
> 这让你能设计更高效的算法，和 infra 团队高效沟通，在面试中展示深度。

---

## 附录：项目文件快速索引

| 文件 | 面试价值 | 重点内容 |
|------|---------|---------|
| `elementwise/add.cu` | ⭐⭐ | float4 向量化、grid/block 设计 |
| `reduce/sum/sum.cu` | ⭐⭐⭐⭐ | warp shuffle、shared memory |
| `reduce/softmax/softmax.cu` | ⭐⭐⭐⭐⭐ | softmax 实现、数值稳定性 |
| `reduce/softmax_matrix/softmax_matrix.cu` | ⭐⭐⭐⭐⭐ | 行归约模式、__shfl_xor_sync |
| `sgemm/kernel1.cu` | ⭐⭐⭐ | naive 实现、性能分析 |
| `sgemm/kernel2.cu` | ⭐⭐⭐⭐ | shared memory tiling |
| `sgemm/kernel3.cu` | ⭐⭐⭐⭐ | thread tile、寄存器缓存 |
| `sgemm/kernel7.cu` | ⭐⭐⭐⭐⭐ | 双缓冲、延迟隐藏 |
| `transpose/transpose.cu` | ⭐⭐⭐ | bank conflict、padding/swizzling |
