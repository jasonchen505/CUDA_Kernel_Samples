# CUDA_Kernel_Samples 完整复现计划

> 硬件资源：8x RTX 3090 (24GB each, SM 8.6, Ampere)
> 目标：完整复现项目 + 理解LLM推理优化原理

---

## 一、项目与硬件概况

### 1.1 项目核心内容

| 模块 | 算子 | 优化版本数 | 面试价值 |
|------|------|-----------|---------|
| elementwise | add/sigmoid/relu | 2 (naive + float4) | ⭐⭐ |
| reduce | sum/max/softmax | 6+4+2+4 | ⭐⭐⭐⭐⭐ |
| sgemm | 矩阵乘 | 7 (+cuBLAS) | ⭐⭐⭐⭐ |
| transpose | 矩阵转置 | 6 | ⭐⭐⭐ |
| gemv | 矩阵乘向量 | 1 | ⭐⭐⭐ |

### 1.2 RTX 3090 硬件特性

```
SM数量: 82 (vs GTX 1050 的 5)
CUDA Cores: 10496
FP32算力: ~35.6 TFLOPS
显存带宽: 936 GB/s
显存容量: 24 GB
共享内存/Block: 最大 100 KB (可配置)
最大线程数/SM: 1536
计算能力: 8.6 (Ampere)
Tensor Core: 第3代 (支持FP16/BF16/INT8/INT4)
```

### 1.3 与 GTX 1050 的对比

| 指标 | GTX 1050 | RTX 3090 | 提升倍数 |
|------|----------|----------|---------|
| SM数量 | 5 | 82 | 16.4x |
| FP32 TFLOPS | ~1.8 | ~35.6 | ~20x |
| 显存带宽 | 112 GB/s | 936 GB/s | 8.4x |
| 显存容量 | 2 GB | 24 GB | 12x |
| 共享内存/Block | 48 KB | 100 KB | 2.1x |

---

## 二、复现计划总览

### 2.1 阶段划分

```
阶段1: 环境配置 (Day 1)
阶段2: 基础算子复现 (Day 2-3)
阶段3: 核心算子复现 (Day 4-7)
阶段4: 性能对比分析 (Day 8-9)
阶段5: LLM推理关联实验 (Day 10-14)
```

### 2.2 时间线

```
Week 1: 环境配置 + 基础算子 + 核心算子
Week 2: 性能分析 + LLM关联实验 + 面试准备
```

---

## 三、详细执行计划

### 阶段1: 环境配置 (Day 1)

#### 1.1 检查CUDA环境

```bash
# 检查CUDA版本
nvcc --version

# 检查GPU信息
nvidia-smi

# 检查cuBLAS
ls /usr/local/cuda/lib64/libcublas*
```

#### 1.2 安装依赖

```bash
# CMake
sudo apt install cmake

# Python依赖 (用于性能绘图)
pip install matplotlib numpy

# 验证
cmake --version
python3 -c "import matplotlib; print(matplotlib.__version__)"
```

#### 1.3 编译测试

```bash
# 测试hello_cuda
cd example/hello_cuda
mkdir build && cd build
cmake .. && make
./hello_cuda

# 测试cuda_info
cd ../../cuda_info
nvcc main.cu -o cuda_info
./cuda_info  # 应该显示RTX 3090的信息
```

#### 1.4 选择GPU

```bash
# 使用指定GPU运行测试
CUDA_VISIBLE_DEVICES=0 ./main  # 使用第0张卡
# 或者在代码中
cudaSetDevice(0);
```

---

### 阶段2: 基础算子复现 (Day 2-3)

#### 2.1 Elementwise (Day 2 上午)

**目标**: 理解grid/block设计 + float4向量化

```bash
cd elementwise
nvcc add.cu -o add -arch=sm_86
./add
```

**学习要点**:
- [ ] 理解 `blockIdx.x * blockDim.x + threadIdx.x` 索引计算
- [ ] 理解 `float4` 向量化访存的原理
- [ ] 理解为什么grid维度除以4而不是block维度

**验证**:
- [ ] naive版本能正确运行
- [ ] float4版本能正确运行
- [ ] 输出结果与CPU参考一致

#### 2.2 Reduce/Sum (Day 2 下午)

**目标**: 掌握shared memory + warp shuffle

```bash
cd reduce/sum
mkdir build && cd build
cmake .. && make
./sum
```

**学习要点**:
- [ ] 理解v0: 原子操作的序列化问题
- [ ] 理解v1: shared memory的折半归约
- [ ] 理解v2: 动态shared memory
- [ ] 理解v3: atomicAdd的使用
- [ ] 理解v4: `__shfl_down_sync`的warp级归约
- [ ] 理解v5: float4 + warp shuffle的组合优化

**验证**:
- [ ] 6个版本都能正确运行
- [ ] 性能递增符合预期
- [ ] 与CPU参考结果一致

#### 2.3 Reduce/Max (Day 3 上午)

**目标**: 理解自定义atomicMax

```bash
cd reduce/max
mkdir build && cd build
cmake .. && make
./max
```

**学习要点**:
- [ ] 理解CUDA只提供int的atomicMax，float需要自己实现
- [ ] 理解`atomicCAS`的用法
- [ ] 理解`__float_as_int`和`__int_as_float`的转换

**验证**:
- [ ] 结果与CPU参考一致
- [ ] 理解为什么需要自定义atomicMax

#### 2.4 Reduce/Softmax (Day 3 下午)

**目标**: 理解多kernel协作 + 数值稳定性

```bash
cd reduce/softmax
mkdir build && cd build
cmake .. && make
./softmax
```

**学习要点**:
- [ ] 理解v1: 不稳定版本的问题
- [ ] 理解v2: 为什么需要减去最大值
- [ ] 理解为什么需要3个kernel而不是1个
- [ ] 理解`__threadfence()`为什么不能替代kernel边界

**验证**:
- [ ] v1和v2都能正确运行
- [ ] v2的结果更稳定（大数值不会溢出）

#### 2.5 Softmax_matrix (Day 3 晚上)

**目标**: 理解行归约模式 + `__shfl_xor_sync`

```bash
cd reduce/softmax_matrix
mkdir build && cd build
cmake .. && make
./softmax_matrix
```

**学习要点**:
- [ ] 理解一个warp处理一行的模式
- [ ] 理解`__shfl_xor_sync` vs `__shfl_down_sync`的区别
- [ ] 理解为什么`__shfl_xor_sync`可以省略shared memory

**验证**:
- [ ] 行softmax和列softmax都正确
- [ ] 性能对比符合预期

---

### 阶段3: 核心算子复现 (Day 4-7)

#### 3.1 SGEMM Kernel 1-3 (Day 4)

**目标**: 理解shared memory tiling + thread tile

```bash
cd sgemm
mkdir build && cd build
cmake .. && make

# 逐个测试
./main 1  # kernel1: naive
./main 2  # kernel2: shared memory tiling
./main 3  # kernel3: 1D thread tile
```

**学习要点**:
- [ ] Kernel1: 理解naive实现的性能瓶颈
- [ ] Kernel2: 理解shared memory如何减少全局内存访问
- [ ] Kernel3: 理解thread tile如何提高计算访存比

**验证**:
- [ ] 三个kernel都能正确运行
- [ ] 性能递增符合预期
- [ ] 理解每个优化带来的性能提升

#### 3.2 SGEMM Kernel 4-5 (Day 5)

**目标**: 理解2D thread tile + register caching

```bash
./main 4  # kernel4: 2D thread tile
./main 5  # kernel5: register caching
```

**学习要点**:
- [ ] Kernel4: 理解2D thread tile的设计
- [ ] Kernel5: 理解如何用寄存器缓存shared memory数据

**验证**:
- [ ] 两个kernel都能正确运行
- [ ] 理解register caching的优化原理

#### 3.3 SGEMM Kernel 6-7 (Day 6)

**目标**: 理解float4向量化 + 双缓冲

```bash
./main 6  # kernel6: float4 vectorized
./main 7  # kernel7: double buffering
```

**学习要点**:
- [ ] Kernel6: 理解float4如何减少访存指令
- [ ] Kernel6: 理解As转置存储的目的
- [ ] Kernel7: 理解双缓冲的原理
- [ ] Kernel7: 理解如何隐藏访存延迟

**验证**:
- [ ] 两个kernel都能正确运行
- [ ] kernel7达到cuBLAS的99%+性能

#### 3.4 Transpose (Day 7)

**目标**: 理解bank conflict + padding/swizzling

```bash
cd transpose
mkdir build && cd build
cmake .. && make
./transpose
```

**学习要点**:
- [ ] 理解什么是bank conflict
- [ ] 理解padding如何解决bank conflict
- [ ] 理解swizzling如何解决bank conflict
- [ ] 理解`__ldg()`的缓存作用

**验证**:
- [ ] 6个版本都能正确运行
- [ ] v3比v1慢（因为bank conflict）
- [ ] v4和v5解决bank conflict后性能恢复

---

### 阶段4: 性能对比分析 (Day 8-9)

#### 4.1 运行完整benchmark

```bash
cd sgemm
bash tools/test.sh  # 自动测试所有kernel并生成图表
```

#### 4.2 性能数据收集

| Kernel | RTX 3090 GFLOPS | GTX 1050 GFLOPS | 提升倍数 |
|--------|-----------------|-----------------|---------|
| cuBLAS | ? | 9359 | ? |
| Kernel1 | ? | 1012 | ? |
| Kernel2 | ? | 1249 | ? |
| Kernel3 | ? | 3671 | ? |
| Kernel4 | ? | 7243 | ? |
| Kernel5 | ? | 7158 | ? |
| Kernel6 | ? | 7806 | ? |
| Kernel7 | ? | 9325 | ? |

#### 4.3 性能分析

**分析维度**:
1. 计算访存比 vs 实际性能
2. 共享内存使用 vs 性能
3. 寄存器压力 vs Occupancy
4. 与cuBLAS的差距分析

#### 4.4 优化探索

**RTX 3090特有的优化机会**:
1. 更大的共享内存 (100 KB vs 48 KB) → 可以尝试更大的tile
2. Tensor Core → 可以尝试FP16/BF16计算
3. 更大的L2缓存 → 可以调整缓存策略

---

### 阶段5: LLM推理关联实验 (Day 10-14)

#### 5.1 Softmax在Attention中的应用 (Day 10)

**实验目标**: 验证softmax kernel在LLM推理中的应用

```python
# 模拟Attention计算
import torch

# 创建Q, K, V
batch_size, num_heads, seq_len, head_dim = 1, 32, 2048, 128
Q = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
K = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
V = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()

# 计算Attention
scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
# 这里的softmax可以用我们实现的kernel
attention = torch.softmax(scores, dim=-1)
output = torch.matmul(attention, V)
```

**学习要点**:
- [ ] 理解Attention中的softmax是行softmax
- [ ] 理解为什么需要online softmax (FlashAttention)
- [ ] 理解causal mask如何影响softmax

#### 5.2 SGEMM在Linear层中的应用 (Day 11)

**实验目标**: 验证SGEMM kernel在LLM推理中的应用

```python
# 模拟Linear层计算
import torch

# 7B模型的Linear层
in_features, out_features = 4096, 4096
weight = torch.randn(out_features, in_features).cuda()
input_tensor = torch.randn(1, in_features).cuda()  # batch_size=1

# 这就是GEMV (矩阵乘向量)
output = torch.matmul(input_tensor, weight.T)
```

**学习要点**:
- [ ] 理解batch_size=1时是GEMV不是GEMM
- [ ] 理解为什么LLM推理是memory-bound
- [ ] 理解为什么batch_size越大，GPU利用率越高

#### 5.3 量化对性能的影响 (Day 12)

**实验目标**: 验证量化如何影响访存和计算

```python
# 比较FP32 vs FP16 vs INT8
import torch

# FP32
x_fp32 = torch.randn(1024, 1024).cuda()
w_fp32 = torch.randn(1024, 1024).cuda()
# 计算量: 1024^3 FLOPS
# 访存量: 2 * 1024^2 * 4 bytes = 8 MB

# FP16
x_fp16 = x_fp32.half()
w_fp16 = w_fp32.half()
# 计算量: 同上
# 访存量: 2 * 1024^2 * 2 bytes = 4 MB (减半)
# Tensor Core: 吞吐量是FP32的2倍

# INT8
x_int8 = torch.quantize_per_tensor(x_fp32, 0.1, 10, torch.qint8)
w_int8 = torch.quantize_per_tensor(w_fp32, 0.1, 10, torch.qint8)
# 访存量: 2 * 1024^2 * 1 byte = 2 MB (再减半)
# Tensor Core: 吞吐量是FP32的4倍
```

**学习要点**:
- [ ] 理解量化如何减少访存量
- [ ] 理解Tensor Core的吞吐量优势
- [ ] 理解量化对精度的影响

#### 5.4 KV Cache管理 (Day 13)

**实验目标**: 理解KV Cache的显存占用和管理

```python
# 计算KV Cache显存占用
def calculate_kv_cache_size(
    num_layers, hidden_size, num_heads, 
    seq_len, batch_size, dtype_bytes=2
):
    # 每层的K和V
    per_layer = 2 * hidden_size * seq_len * batch_size * dtype_bytes
    total = per_layer * num_layers
    return total

# 7B模型
kv_cache = calculate_kv_cache_size(
    num_layers=32, hidden_size=4096, num_heads=32,
    seq_len=4096, batch_size=1, dtype_bytes=2  # FP16
)
print(f"KV Cache大小: {kv_cache / 1024**3:.2f} GB")  # 约2GB
```

**学习要点**:
- [ ] 理解KV Cache的显存占用计算
- [ ] 理解为什么需要PagedAttention
- [ ] 理解连续批处理的优势

#### 5.5 FlashAttention原理验证 (Day 14)

**实验目标**: 通过实验验证FlashAttention的优化原理

```python
# 对比标准Attention vs FlashAttention的显存占用
import torch

seq_len = 4096

# 标准Attention: 需要存储N×N的attention matrix
standard_memory = seq_len * seq_len * 2  # FP16, bytes
print(f"标准Attention显存: {standard_memory / 1024**2:.2f} MB")  # 约32MB

# FlashAttention: 只需要O(N)显存
flash_memory = seq_len * 2  # 只存储输出
print(f"FlashAttention显存: {flash_memory / 1024**2:.2f} MB")  # 约0.008MB
```

**学习要点**:
- [ ] 理解FlashAttention的tiling策略
- [ ] 理解online softmax的实现
- [ ] 理解为什么FlashAttention能减少显存

---

## 四、多GPU利用策略

### 4.1 并行测试

```bash
# 在不同GPU上并行运行不同kernel的测试
CUDA_VISIBLE_DEVICES=0 ./main 1 &  # GPU 0测试kernel1
CUDA_VISIBLE_DEVICES=1 ./main 2 &  # GPU 1测试kernel2
CUDA_VISIBLE_DEVICES=2 ./main 3 &  # GPU 2测试kernel3
# ...
wait
```

### 4.2 大规模矩阵测试

```bash
# 如果想测试更大的矩阵，可以修改代码
# 例如: 8192x8192矩阵，需要 8192^2 * 4 bytes = 256 MB
# 单卡24GB完全可以处理
```

### 4.3 批量性能测试

```python
# 编写脚本自动在多张GPU上运行测试
import subprocess
import os

gpus = [0, 1, 2, 3, 4, 5, 6, 7]
kernels = [0, 1, 2, 3, 4, 5, 6, 7]

for gpu in gpus:
    for kernel in kernels:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        subprocess.run(['./main', str(kernel)], env=env)
```

---

## 五、复现检查清单

### 5.1 环境配置

- [ ] CUDA版本确认 (>= 9.0)
- [ ] cuBLAS库可用
- [ ] CMake版本确认 (>= 3.16)
- [ ] Python + matplotlib可用
- [ ] GPU信息正确识别

### 5.2 基础算子

- [ ] elementwise/add: naive版本正确
- [ ] elementwise/add: float4版本正确
- [ ] reduce/sum: 6个版本都正确
- [ ] reduce/max: 结果正确
- [ ] reduce/softmax: v1和v2都正确
- [ ] reduce/softmax_matrix: 行和列softmax都正确

### 5.3 核心算子

- [ ] sgemm/kernel1: 运行正确
- [ ] sgemm/kernel2: 运行正确，性能提升
- [ ] sgemm/kernel3: 运行正确，性能提升
- [ ] sgemm/kernel4: 运行正确，性能提升
- [ ] sgemm/kernel5: 运行正确，性能提升
- [ ] sgemm/kernel6: 运行正确，性能提升
- [ ] sgemm/kernel7: 运行正确，接近cuBLAS性能
- [ ] transpose: 6个版本都正确

### 5.4 性能分析

- [ ] 收集RTX 3090性能数据
- [ ] 与GTX 1050数据对比
- [ ] 分析性能差异原因
- [ ] 生成性能对比图表

### 5.5 LLM关联

- [ ] 理解softmax在Attention中的应用
- [ ] 理解SGEMM在Linear层中的应用
- [ ] 理解量化对性能的影响
- [ ] 理解KV Cache的显存管理
- [ ] 理解FlashAttention的优化原理

---

## 六、预期成果

### 6.1 技术成果

1. **完整复现**: 所有CUDA kernel在RTX 3090上正确运行
2. **性能数据**: RTX 3090 vs GTX 1050的性能对比
3. **优化理解**: 深入理解每个优化的原理和效果
4. **LLM关联**: 理解CUDA优化与LLM推理的关系

### 6.2 面试准备

1. **项目介绍**: 能够清晰介绍项目内容和优化思路
2. **深挖回答**: 能够回答底层原理、实验验证、问题定位等问题
3. **LLM关联**: 能够将CUDA知识与LLM优化联系起来
4. **实战经验**: 能够分享复现过程中遇到的问题和解决方案

### 6.3 学习笔记

1. **新增知识点**: 记录复现过程中新学到的内容
2. **问题与解决**: 记录遇到的问题和解决方案
3. **性能分析**: 记录性能数据和分析结论
4. **LLM关联**: 记录CUDA知识与LLM优化的关联

---

## 七、时间安排

### Week 1: 核心复现

| Day | 任务 | 产出 |
|-----|------|------|
| 1 | 环境配置 | 环境就绪，基础测试通过 |
| 2 | elementwise + reduce/sum | 理解grid/block + shared memory |
| 3 | reduce/max + softmax + softmax_matrix | 理解warp shuffle + 数值稳定性 |
| 4 | sgemm kernel1-3 | 理解tiling + thread tile |
| 5 | sgemm kernel4-5 | 理解2D thread tile + register caching |
| 6 | sgemm kernel6-7 | 理解float4 + 双缓冲 |
| 7 | transpose | 理解bank conflict |

### Week 2: 分析与应用

| Day | 任务 | 产出 |
|-----|------|------|
| 8 | 性能benchmark | RTX 3090性能数据 |
| 9 | 性能分析 | 性能对比报告 |
| 10 | Softmax + Attention | 理解Attention中的softmax |
| 11 | SGEMM + Linear | 理解Linear层的计算 |
| 12 | 量化实验 | 理解量化对性能的影响 |
| 13 | KV Cache实验 | 理解显存管理 |
| 14 | FlashAttention验证 | 理解FlashAttention原理 |

---

## 附录：快速命令参考

### 编译命令

```bash
# elementwise
cd elementwise && nvcc add.cu -o add -arch=sm_86

# reduce/sum, reduce/max, reduce/softmax, reduce/softmax_matrix
cd reduce/sum && mkdir build && cd build && cmake .. && make

# sgemm
cd sgemm && mkdir build && cd build && cmake .. && make

# transpose
cd transpose && mkdir build && cd build && cmake .. && make
```

### 测试命令

```bash
# sgemm测试
./main 0  # cuBLAS
./main 1  # kernel1
...
./main 7  # kernel7

# 性能测试
bash tools/test.sh
```

### GPU选择

```bash
# 使用指定GPU
CUDA_VISIBLE_DEVICES=0 ./main 1

# 或在代码中
cudaSetDevice(0);
```
