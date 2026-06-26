# 复现学习增量笔记 — RTX 3090 实测经验

> 基于 CUDA_Kernel_Samples 项目在 8x RTX 3090 上的完整复现
> 记录对比前两轮文档新增的知识点和实测发现

---

## 一、环境配置增量知识

### 1.1 CUDA 13.1 兼容性问题

**问题**: 项目原设计基于 CUDA 12.4，在 CUDA 13.1 上编译 SGEMM 时出现模板实例化链接错误。

**根因分析**:
```
CUDA 13.1 对模板实例化有更严格的要求：
- 模板函数在头文件(.cuh)中只有声明
- 定义在源文件(.cu)中
- 默认的 whole program compilation mode 无法找到跨编译单元的模板实例化
```

**解决方案**:
```cmake
# CMakeLists.txt 中添加 -rdc=true 标志
target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)

# 或者设置 CUDA_SEPARABLE_COMPILATION
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```

**新增知识点**:
- `-rdc=true` (Relocatable Device Code) 允许跨编译单元的设备代码链接
- 模板实例化需要在链接时可见，这与 CPU 编译不同
- CUDA 13.1 比旧版本对模板实例化有更严格的检查

### 1.2 GPU 架构指定

**问题**: 编译时需要指定正确的 GPU 架构。

**解决方案**:
```bash
# 方法1: 命令行指定
nvcc -arch=sm_86 kernel.cu

# 方法2: CMakeLists.txt 指定
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "86")

# 方法3: cmake 命令行指定
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86
```

**新增知识点**:
- RTX 3090 的计算能力是 8.6 (Ampere 架构)
- `sm_86` 对应 RTX 3090/3080
- `sm_80` 对应 A100
- 不指定架构可能导致性能损失或兼容性问题

---

## 二、性能实测增量知识

### 2.1 RTX 3090 vs GTX 1050 性能对比

#### SGEMM 性能对比 (2560×2560 矩阵)

| Kernel | GTX 1050 (GFLOPS) | RTX 3090 (GFLOPS) | 提升倍数 | 占 cuBLAS 比例 |
|--------|-------------------|-------------------|---------|---------------|
| cuBLAS | 9,359 | 23,929 | 2.56x | 100% |
| Kernel 1 (Naive) | 1,012 | 2,163 | 2.14x | 9.0% |
| Kernel 2 (Shared Mem) | 1,249 | 2,023 | 1.62x | 8.5% |
| Kernel 3 (Thread Tile) | 3,671 | 3,661 | 1.00x | 15.3% |
| Kernel 4 (2D Tile) | 7,243 | 9,737 | 1.34x | 40.7% |
| Kernel 5 (Reg Cache) | 7,158 | 9,743 | 1.36x | 40.7% |
| Kernel 6 (Float4) | 7,806 | 15,965 | 2.05x | 66.7% |
| Kernel 7 (Double Buf) | 9,325 | 21,497 | 2.31x | 89.8% |

**关键发现**:

1. **cuBLAS 提升不成比例**: cuBLAS 提升 2.56x，但自定义 kernel 提升幅度差异很大
   - Kernel 1-3 提升较小 (1.0-2.1x)
   - Kernel 6-7 提升较大 (2.0-2.3x)
   - **原因**: RTX 3090 的 cuBLAS 针对 Ampere 架构深度优化，而自定义 kernel 的参数是针对 GTX 1050 调优的

2. **Kernel 2 性能反常**: 在 RTX 3090 上反而比 Kernel 1 慢
   - **可能原因**: Kernel 2 的 BLOCK_SIZE=32，在 RTX 3090 的 100KB 共享内存下没有充分利用
   - **启示**: 优化参数需要针对目标硬件调优

3. **Float4 优化效果显著**: Kernel 6 相比 Kernel 5 提升 63%
   - **原因**: RTX 3090 的显存带宽 (936 GB/s) 是 GTX 1050 (112 GB/s) 的 8.4 倍
   - **启示**: 向量化访存在高带宽硬件上效果更明显

#### Reduce/Sum 性能对比 (N=100M)

| Version | GTX 1050 (ms) | RTX 3090 (ms) | 提升倍数 |
|---------|---------------|---------------|---------|
| CPU | 383.85 | - | - |
| v0 (Global Mem) | 31.07 | 3.95 | 7.87x |
| v1 (Shared Mem) | 19.65 | 4.25 | 4.62x |
| v2 (Dynamic Shared) | 19.48 | 4.23 | 4.61x |
| v3 (Atomic) | 15.86 | 1.48 | 10.72x |
| v4 (Warp Shuffle) | 11.21 | 1.48 | 7.57x |
| v5 (Float4 + Shuffle) | 4.11 | 0.47 | 8.74x |

**关键发现**:

1. **v0 提升最大**: 7.87x，接近显存带宽提升比 (8.4x)
   - **原因**: v0 是纯访存密集型，完全受显存带宽限制
   - **启示**: Memory-bound 的 kernel 性能提升与带宽提升成正比

2. **v1/v2 提升较小**: 4.6x，远低于带宽提升比
   - **原因**: 共享内存访问延迟在 RTX 3090 上没有显著改善
   - **启示**: Shared memory 的延迟相对稳定，不是主要瓶颈

3. **v3/v4 提升较大**: 7.6-10.7x，超过带宽提升比
   - **原因**: Warp shuffle 和 atomic 操作在 Ampere 架构上有优化
   - **启示**: 新架构对特定指令有专门优化

#### Softmax_Matrix 性能对比 (M=2048, N=64)

| 版本 | GTX 1050 (ms) | RTX 3090 (ms) | 提升倍数 |
|------|---------------|---------------|---------|
| Row CPU | 2.60 | - | - |
| Row GPU | 0.062 | 0.008 | 7.75x |
| Col CPU | 5.16 | - | - |
| Col GPU | 0.168 | 0.039 | 4.31x |

**关键发现**:

1. **行 softmax 提升更大**: 7.75x vs 列 softmax 4.31x
   - **原因**: 行 softmax 的访存模式更友好（连续访问）
   - **启示**: 访存模式对性能影响很大

2. **绝对性能**: RTX 3090 上行 softmax 只需 8 微秒
   - **启示**: 对于 LLM 推理，softmax 不是瓶颈

#### Transpose 性能对比 (M=12800, N=1280)

| 版本 | GTX 1050 (ms) | RTX 3090 (ms) | 提升倍数 |
|------|---------------|---------------|---------|
| v0 (Naive) | 6.86 | 0.58 | 11.83x |
| v1 (Coalesced) | 4.31 | 0.29 | 14.86x |
| v2 (LDG) | 2.12 | 0.29 | 7.31x |
| v3 (Shared Mem) | 3.81 | 0.29 | 13.14x |
| v4 (Padding) | 2.04 | 0.21 | 9.71x |
| v5 (Swizzling) | 2.02 | 0.21 | 9.62x |

**关键发现**:

1. **v0 提升最大**: 11.83x，超过显存带宽提升比
   - **原因**: RTX 3090 的 L2 缓存更大 (6MB vs 1MB)，缓解了非合并访问的影响
   - **启示**: 大缓存可以显著改善访存模式不友好的 kernel

2. **Bank conflict 影响减小**: v3 在 RTX 3090 上没有明显性能下降
   - **原因**: Ampere 架构的共享内存带宽更高
   - **启示**: 新架构对 bank conflict 的容忍度更高

### 2.2 性能瓶颈分析

**RTX 3090 的性能特点**:

```
显存带宽: 936 GB/s (GTX 1050 的 8.4 倍)
FP32 算力: 35.6 TFLOPS (GTX 1050 的 ~20 倍)
共享内存/SM: 100 KB (GTX 1050 的 2.1 倍)
L2 缓存: 6 MB (GTX 1050 的 6 倍)
```

**性能瓶颈转移**:

| Kernel | GTX 1050 瓶颈 | RTX 3090 瓶颈 |
|--------|--------------|--------------|
| Kernel 1 | 显存带宽 | 显存带宽 |
| Kernel 2 | 共享内存带宽 | 共享内存带宽 |
| Kernel 3 | 计算 + 访存 | 计算 |
| Kernel 4-7 | 计算 | 计算 |

**启示**:
- 在 GTX 1050 上，大部分 kernel 是 memory-bound
- 在 RTX 3090 上，由于带宽提升更大，更多 kernel 变成 compute-bound
- 优化策略需要根据硬件特点调整

---

## 三、LLM 推理优化增量知识

### 3.1 Softmax 在 LLM 中的应用

**实测验证**:

```python
# 模拟 Attention 中的 Softmax
import torch

seq_len = 4096
scores = torch.randn(1, 32, seq_len, seq_len).cuda()  # [batch, heads, seq, seq]

# 标准 Softmax
attention = torch.softmax(scores, dim=-1)

# 显存占用: 1 * 32 * 4096 * 4096 * 2 bytes = 1 GB (FP16)
```

**新增知识点**:

1. **Softmax 是行操作**: Attention 的 softmax 在最后一维（seq_len）上进行
   - 对应项目中的 `softmax_row_kernel`
   - 每个 warp 处理一行

2. **Online Softmax 的必要性**: FlashAttention 需要 online softmax
   - 传统 softmax 需要两遍扫描：一遍求 max，一遍求 sum
   - Online softmax 可以一遍完成，但需要维护 running max 和 running sum
   - 对应项目中的 max_kernel + sum_kernel 分离设计

3. **数值稳定性**: 减去最大值是关键
   - 项目中的 v1 vs v2 对比验证了这一点
   - 在 FP16 下更容易溢出，需要特别注意

### 3.2 SGEMM 在 LLM 中的应用

**实测验证**:

```python
# 模拟 Linear 层
import torch

# 7B 模型的典型 Linear 层
in_features, out_features = 4096, 4096
weight = torch.randn(out_features, in_features).cuda()  # 64 MB (FP32)

# Batch size = 1: GEMV (矩阵乘向量)
input_1 = torch.randn(1, in_features).cuda()
output_1 = torch.matmul(input_1, weight.T)  # GEMV

# Batch size = 32: GEMM (矩阵乘矩阵)
input_32 = torch.randn(32, in_features).cuda()
output_32 = torch.matmul(input_32, weight.T)  # GEMM
```

**新增知识点**:

1. **GEMV vs GEMM 的性能差异**:
   - Batch size = 1 时是 GEMV，计算访存比很低
   - Batch size = 32 时是 GEMM，计算访存比提高 32 倍
   - 这就是为什么 LLM 推理的吞吐量随 batch size 增加而增加

2. **项目中的 SGEMM 优化与 LLM 的关系**:
   - Kernel 1 (Naive): 对应 naive 的 Linear 实现
   - Kernel 7 (Double Buffer): 对应优化后的 Linear 实现
   - 性能差距：21,497 vs 2,163 GFLOPS (10x)

3. **Tensor Core 的机会**:
   - 项目中的 kernel 都是 FP32，没有使用 Tensor Core
   - RTX 3090 的 Tensor Core 可以提供 2-8x 的额外加速
   - 这是进一步优化的方向

### 3.3 量化对性能的影响

**理论分析**:

| 精度 | 字节数 | 显存占用 | Tensor Core 吞吐量 |
|------|--------|---------|-------------------|
| FP32 | 4 | 100% | 1x |
| FP16 | 2 | 50% | 2x |
| INT8 | 1 | 25% | 4x |
| INT4 | 0.5 | 12.5% | 8x |

**新增知识点**:

1. **量化减少的是访存量，不是计算量**:
   - 矩阵乘法的 FLOPS 不变
   - 但访存的字节数减少
   - 对于 memory-bound 的 kernel，性能提升显著

2. **项目中 float4 优化的启示**:
   - float4 一次读取 128 位 (16 字节)
   - INT8 一次读取 128 位可以读 16 个元素
   - 这就是为什么量化可以加速推理

3. **RTX 3090 的 Tensor Core 支持**:
   - FP16: 2x FP32 吞吐量
   - BF16: 2x FP32 吞吐量
   - INT8: 4x FP32 吞吐量
   - INT4: 8x FP32 吞吐量 (稀疏)

### 3.4 KV Cache 管理

**显存占用计算**:

```python
def calculate_kv_cache_size(
    num_layers, hidden_size, num_heads, 
    seq_len, batch_size, dtype_bytes=2
):
    """
    计算 KV Cache 显存占用
    
    参数:
    - num_layers: Transformer 层数
    - hidden_size: 隐藏层维度
    - num_heads: 注意力头数
    - seq_len: 序列长度
    - batch_size: 批大小
    - dtype_bytes: 数据类型字节数 (FP16=2, FP32=4)
    """
    # 每层的 K 和 V
    head_dim = hidden_size // num_heads
    per_layer = 2 * num_heads * head_dim * seq_len * batch_size * dtype_bytes
    total = per_layer * num_layers
    return total

# 7B 模型 (LLaMA-7B)
kv_cache_7b = calculate_kv_cache_size(
    num_layers=32, hidden_size=4096, num_heads=32,
    seq_len=4096, batch_size=1, dtype_bytes=2
)
print(f"7B 模型 KV Cache: {kv_cache_7b / 1024**3:.2f} GB")  # 约 2 GB

# 70B 模型 (LLaMA-70B)
kv_cache_70b = calculate_kv_cache_size(
    num_layers=80, hidden_size=8192, num_heads=64,
    seq_len=4096, batch_size=1, dtype_bytes=2
)
print(f"70B 模型 KV Cache: {kv_cache_70b / 1024**3:.2f} GB")  # 约 20 GB
```

**新增知识点**:

1. **KV Cache 的显存占用**:
   - 7B 模型: 约 2 GB (FP16, seq_len=4096)
   - 70B 模型: 约 20 GB (FP16, seq_len=4096)
   - 这就是为什么长序列推理需要 PagedAttention

2. **PagedAttention 的原理**:
   - 类似操作系统的虚拟内存分页
   - 将 KV Cache 分成固定大小的 page
   - 通过 page table 映射逻辑地址到物理地址
   - 减少内存碎片，提高利用率

3. **连续批处理的优势**:
   - 传统批处理：所有请求必须等最长的完成
   - 连续批处理：新请求可以插入正在处理的批次
   - 提高 GPU 利用率，减少延迟

### 3.5 FlashAttention 原理验证

**理论分析**:

```
标准 Attention:
- 显存: O(N²) - 需要存储 N×N 的 attention matrix
- 计算: O(N²) - 需要计算所有 pair 的 attention score

FlashAttention:
- 显存: O(N) - 只存储输出，不存储 attention matrix
- 计算: O(N²) - 计算量不变，但通过 tiling 减少 HBM 访问
```

**新增知识点**:

1. **Tiling 策略**:
   - 将 Q/K/V 分成小块，每块放入 SRAM
   - 在 SRAM 中完成 softmax 和乘法
   - 避免将完整的 attention matrix 写回 HBM
   - 这与项目中 SGEMM 的 tiling 策略完全一致

2. **Online Softmax**:
   - 传统 softmax 需要两遍扫描
   - Online softmax 可以一遍完成
   - 需要维护 running max 和 running sum
   - 项目中的 max_kernel + sum_kernel 分离设计是基础

3. **IO 复杂度**:
   - 标准 Attention: O(N² * d) HBM 访问
   - FlashAttention: O(N² * d² / M) HBM 访问，其中 M 是 SRAM 大小
   - 当 d² < M 时，FlashAttention 更快
   - RTX 3090 的 SRAM (100 KB) 足够大

---

## 四、工程落地增量知识

### 4.1 编译优化

**编译选项**:

```bash
# 基本编译
nvcc -arch=sm_86 kernel.cu -o kernel

# 优化选项
nvcc -arch=sm_86 -O3 kernel.cu -o kernel

# 调试选项
nvcc -arch=sm_86 -G -g kernel.cu -o kernel

# 性能分析选项
nvcc -arch=sm_86 -lineinfo kernel.cu -o kernel
```

**新增知识点**:

1. **`-arch=sm_86`**: 针对 RTX 3090 优化
   - 启用 Ampere 架构的特性
   - 使用正确的指令集

2. **`-O3`**: 最高级别优化
   - 循环展开
   - 指令调度
   - 常量折叠

3. **`-lineinfo`**: 保留行号信息
   - 用于性能分析 (nsys/ncu)
   - 不影响性能

### 4.2 性能分析工具

**常用工具**:

```bash
# NVIDIA Nsight Systems (nsys) - 系统级性能分析
nsys profile --stats=true ./main 0

# NVIDIA Nsight Compute (ncu) - Kernel 级性能分析
ncu --set full ./main 0

# 简单计时
CUDA_VISIBLE_DEVICES=0 ./main 0
```

**新增知识点**:

1. **nsys vs ncu**:
   - nsys: 分析整个程序的时间线
   - ncu: 分析单个 kernel 的详细性能
   - 先用 nsys 找到瓶颈 kernel，再用 ncu 分析

2. **关键指标**:
   - 显存带宽利用率
   - 计算利用率
   - 共享内存使用率
   - Occupancy

### 4.3 多 GPU 利用

**并行测试**:

```bash
# 在不同 GPU 上并行运行
CUDA_VISIBLE_DEVICES=0 ./main 0 &
CUDA_VISIBLE_DEVICES=1 ./main 1 &
CUDA_VISIBLE_DEVICES=2 ./main 2 &
wait
```

**新增知识点**:

1. **CUDA_VISIBLE_DEVICES**: 控制 GPU 可见性
   - 只有指定的 GPU 对程序可见
   - 避免 GPU 冲突

2. **多 GPU 并行**:
   - 每个 GPU 独立运行一个进程
   - 适合并行测试多个配置
   - 不适合单个大任务（需要 NCCL 等通信库）

---

## 五、面试准备增量知识

### 5.1 项目介绍话术更新

**基于实测数据的介绍**:

> "我在 8 卡 RTX 3090 上完整复现了 CUDA 算子优化项目。
>
> 核心成果：SGEMM 优化版本达到 21,497 GFLOPS，是 cuBLAS 的 89.8%。
>
> 关键优化：
> - Shared memory tiling 减少全局内存访问
> - Thread tile 提高计算访存比
> - 双缓冲隐藏访存延迟
>
> 这些优化与 LLM 推理优化高度相关：
> - FlashAttention 的 tiling 策略与 SGEMM 一致
> - Softmax 的 warp shuffle 实现用于 Attention
> - 双缓冲思想用于流水线并行"

### 5.2 面试追问准备

**Q: 你的 kernel 和 cuBLAS 的差距在哪里？**

> "我的 kernel 达到 cuBLAS 的 89.8%，差距主要来自：
>
> 1. **Tensor Core**: cuBLAS 使用了 Tensor Core，我的 kernel 只用 CUDA Core
> 2. **参数调优**: 我的参数是针对 GTX 1050 调优的，RTX 3090 需要不同的参数
> 3. **指令级优化**: cuBLAS 有更精细的指令调度
>
> 如果要进一步优化，需要：
> - 使用 Tensor Core (wmma/mma 指令)
> - 针对 RTX 3090 调优 tile 大小
> - 使用 FP16/BF16 混合精度"

**Q: 你遇到的最大的技术挑战是什么？**

> "CUDA 13.1 的模板实例化问题。
>
> 项目原设计基于 CUDA 12.4，在 CUDA 13.1 上编译时出现链接错误。
>
> 排查过程：
> 1. 检查编译选项 → 发现 warning 信息
> 2. 搜索 CUDA 文档 → 了解模板实例化规则
> 3. 尝试 `-rdc=true` → 问题解决
>
> 这个问题让我理解了 CUDA 编译模型：
> - 默认是 whole program compilation
> - 模板实例化需要在链接时可见
> - `-rdc=true` 启用可重定位设备代码"

**Q: 这个项目对 LLM 算法有什么实际价值？**

> "这个项目帮助我理解了 LLM 推理的底层原理：
>
> 1. **Softmax**: 项目中的 softmax 实现直接用于 Attention
> 2. **SGEMM**: 项目中的矩阵乘优化用于 Linear 层
> 3. **访存优化**: 理解为什么量化能加速推理
> 4. **延迟隐藏**: 理解流水线并行的原理
>
> 实际应用：
> - 优化自定义算子
> - 理解推理框架的优化策略
> - 与 infra 团队高效沟通"

### 5.3 新增面试问题

**Q: 为什么 RTX 3090 上 cuBLAS 的性能提升不成比例？**

> "cuBLAS 在 RTX 3090 上提升 2.56x，但自定义 kernel 提升幅度差异很大。
>
> 原因：
> 1. cuBLAS 针对每个架构深度优化
> 2. 自定义 kernel 的参数是针对旧架构调优的
> 3. 新架构的性能特点不同
>
> 启示：
> - 优化需要针对目标硬件
> - 不能简单移植旧代码
> - 需要重新调优参数"

**Q: 你在复现过程中遇到的最意外的发现是什么？**

> "Kernel 2 在 RTX 3090 上反而比 Kernel 1 慢。
>
> 分析原因：
> - Kernel 2 使用 BLOCK_SIZE=32
> - 在 RTX 3090 的 100KB 共享内存下没有充分利用
> - 共享内存访问的开销超过了收益
>
> 启示：
> - 优化不是越多越好
> - 需要根据硬件特点选择优化策略
> - 小 block size 在大共享内存下可能不是最优"

---

## 六、总结：新增知识点分类

### 6.1 环境配置类

- CUDA 13.1 模板实例化问题及解决方案
- GPU 架构指定方法 (`-arch=sm_86`)
- CMake 分离编译配置

### 6.2 性能实测类

- RTX 3090 vs GTX 1050 的详细性能对比
- 性能瓶颈转移分析 (memory-bound → compute-bound)
- 不同优化策略在不同硬件上的效果差异

### 6.3 LLM 推理类

- Softmax 在 Attention 中的具体应用
- GEMV vs GEMM 的性能差异及原因
- 量化的本质是减少访存量
- KV Cache 的显存占用计算
- FlashAttention 的 tiling 策略与 SGEMM 的关系

### 6.4 工程落地类

- 编译优化选项 (`-O3`, `-lineinfo`)
- 性能分析工具 (nsys, ncu)
- 多 GPU 并行测试方法

### 6.5 面试准备类

- 基于实测数据的项目介绍话术
- 面试追问的准备（性能差距、技术挑战、实际价值）
- 新增面试问题及回答

---

## 附录：实测数据汇总

### A.1 SGEMM 性能数据 (RTX 3090)

| Kernel | 256 | 512 | 1024 | 2048 | 2560 |
|--------|-----|-----|------|------|------|
| cuBLAS | 2291 | 8041 | 18019 | 22331 | 23929 |
| v1 | - | - | - | - | 2163 |
| v2 | - | - | - | - | 2023 |
| v3 | - | - | - | - | 3661 |
| v4 | - | - | - | - | 9737 |
| v5 | - | - | - | - | 9743 |
| v6 | - | - | - | - | 15965 |
| v7 | - | - | - | - | 21497 |

### A.2 Reduce/Sum 性能数据 (RTX 3090, N=100M)

| Version | Time (ms) | vs CPU |
|---------|-----------|--------|
| CPU | 335.71 | 1.0x |
| v0 | 3.95 | 85.0x |
| v1 | 4.25 | 79.0x |
| v2 | 4.23 | 79.4x |
| v3 | 1.48 | 226.8x |
| v4 | 1.48 | 226.8x |
| v5 | 0.47 | 714.3x |

### A.3 Softmax_Matrix 性能数据 (RTX 3090, M=2048, N=64)

| Version | Time (ms) | vs CPU |
|---------|-----------|--------|
| Row CPU | 1.73 | 1.0x |
| Row GPU | 0.008 | 216.3x |
| Col CPU | 2.72 | 1.0x |
| Col GPU | 0.039 | 69.7x |

### A.4 Transpose 性能数据 (RTX 3090, M=12800, N=1280)

| Version | Time (ms) | vs v0 |
|---------|-----------|-------|
| v0 | 0.58 | 1.0x |
| v1 | 0.29 | 2.0x |
| v2 | 0.29 | 2.0x |
| v3 | 0.29 | 2.0x |
| v4 | 0.21 | 2.8x |
| v5 | 0.21 | 2.8x |
