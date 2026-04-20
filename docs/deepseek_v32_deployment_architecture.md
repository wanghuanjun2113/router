# DeepSeek V3.2 昇腾 910B 集群部署 — 系统架构与部署设计文档

> 版本: v1.1 | 日期: 2026-04-20 | 状态: 设计评审

---

## 1. 架构概述

### 1.1 方案背景

部署目标: 在昇腾 910B 集群上运行 DeepSeek V3.2 (685B MoE 模型)，支撑高并发长文本推理服务。

**核心矛盾矩阵:**

| 维度 | 约束 | 量级 |
|------|------|------|
| 模型规模 | 685B 参数 (256 路由专家 + 1 共享专家, 61 层 MLA Transformer) | FP16 原始体积 ~1.3 TB |
| 单卡显存 | 910B 单卡 64 GB HBM | 全量加载需 20+ 卡 |
| 输入特征 | 平均 32K, 长尾 128K | KV Cache 极大 |
| 并发目标 | 40 并发 | Decode 批次吞吐敏感 |
| MLA 特殊性 | `kv_lora_rank=512` 压缩 KV, 但 TP 会复制 KV Cache | Decode 需最小化 TP |
| MoE 特殊性 | 256 路由专家, Top-8 路由 | EP 是天然并行维度 |

**结论:** Prefill 和 Decode 对计算/显存的需求截然不同。必须采用 **PD 分离 + 差异化并行拓扑** 的架构: Prefill 偏计算密集用大 TP 榨取节点内带宽，Decode 偏显存密集用大 DP 扩展 KV 容量。

### 1.2 整体拓扑

```
  ┌──────────────────────────────────────────────────────────┐
  │                    Router Gateway                         │
  │               (Rust, Cache-Aware LB)                      │
  └────────────────────────┬─────────────────────────────────┘
                           │
         ┌─────────────────┼──────────────────┐
         │                 │                  │
   ┌─────▼─────────────────▼──────────────────▼─────┐
   │              Prefill Instance                    │
   │         DP=4  TP=8  EP=32                       │
   │         4 台机器 × 8 卡 = 32 卡                  │
   │                                                  │
   │  ┌DP0──────────────────────────────────────────┐ │
   │  │ TP=8 (node0): 8 cards, EP rank 0-7          │ │
   │  │ TP=8 (node1): 8 cards, EP rank 8-15         │ │
   │  │ TP=8 (node2): 8 cards, EP rank 16-23        │ │
   │  │ TP=8 (node3): 8 cards, EP rank 24-31        │ │
   │  └─────────────────────────────────────────────┘ │
   │  × 4 DP replicas (相同拓扑)                       │
   └──────────────────────┬───────────────────────────┘
                          │
              Mooncake Layerwise RDMA
              (KV Cache P2P Pipeline)
                          │
   ┌──────────────────────▼───────────────────────────┐
   │              Decode Instance                      │
   │         DP=8  TP=2  EP=16                        │
   │         2 台机器 × 8 卡 = 16 卡                   │
   │                                                  │
   │  ┌DP0───────┐ ┌DP1───────┐    ... × 8 DP        │
   │  │ TP=2     │ │ TP=2     │                       │
   │  │ card 0-1 │ │ card 2-3 │                       │
   │  │ EP 0-1   │ │ EP 2-3   │                       │
   │  └──────────┘ └──────────┘                       │
   └──────────────────────────────────────────────────┘

集群规模: 6 台机器 (Prefill 4 台 + Decode 2 台), 共 48 卡
余量:     0~4 台机器用于弹性扩容 (集群上限 10 台)
```

**核心设计原则:**

- **Prefill TP=8:** KV Cache 是临时性的 (计算完即通过 Mooncake 传给 Decode)，TP 带来的 KV 复制只是瞬时开销。大胆使用节点内全带宽 TP=8 最大化 Prefill 计算吞吐。
- **Decode TP=2:** KV Cache 是持久性的 (整个 Decode 生命周期常驻显存)，TP 会导致 KV 被复制到每个 TP rank。TP=2 是计算并行度与 KV 复制开销之间的最优平衡点。
- **EP = TP × DP:** MoE 层的专家在 DP × TP 的全部卡间分片。Prefill EP=32 (每卡仅 8 专家), Decode EP=16 (每卡 16 专家)。

**数据平面:** Prefill 节点计算 KV Cache → Mooncake Transfer Engine (RDMA) → Decode 节点消费。

**控制平面:** Router 通过 ZMQ 与各实例通信拓扑发现，HTTP 代理用户请求。

---

## 2. 资源与并行策略规划

### 2.1 DeepSeek V3.2 模型结构关键参数

从 HuggingFace config.json 及代码库 [deepseek_v2.py](../vllm/vllm/model_executor/models/deepseek_v2.py) 中提取 (V3 与 V3.2 共享相同的 MLA/MoE 维度):

| 参数 | 值 | 说明 |
|------|-----|------|
| `hidden_size` | 7168 | 隐藏层维度 |
| `num_hidden_layers` | 61 | Transformer 层数 |
| `first_k_dense_replace` | 3 | 前 3 层为 Dense MLP, 后 58 层为 MoE |
| `num_heads` (Q) | 128 | Query 头数 |
| `q_lora_rank` | 1536 | Query 压缩维度 |
| `kv_lora_rank` | 512 | KV 压缩维度 (MLA 核心) |
| `qk_nope_head_dim` | 128 | 非旋转注意力头维度 |
| `qk_rope_head_dim` | 64 | 旋转位置编码维度 |
| `v_head_dim` | 128 | Value 头维度 |
| `n_routed_experts` | 256 | 路由专家数 |
| `n_shared_experts` | 1 | 共享专家数 |
| `num_experts_per_tok` | 8 | 每 token 激活专家数 |
| `moe_intermediate_size` | **2048** | 路由专家 MLP 中间维度 |
| `intermediate_size` | 18432 | 共享专家 / Dense MLP 中间维度 |
| `vocab_size` | 129280 | 词表大小 |

**V3.2 新增参数 (稀疏注意力):**

| 参数 | 值 | 说明 |
|------|-----|------|
| `index_topk` | 2048 | 稀疏注意力 Top-K |
| `index_n_heads` | 64 | Indexer 头数 |
| `index_head_dim` | 128 | Indexer 头维度 |

### 2.2 量化方案: W4A16 权重量化

**选择理由:**

vllm-ascend 量化注册表 ([registry.py](../vllm-ascend/vllm_ascend/quantization/methods/registry.py)) 支持 W4A16 方案，实现文件为 [w4a16.py](../vllm-ascend/vllm_ascend/quantization/methods/w4a16.py)。4-bit 权重量化配合 16-bit 激活，在 910B 上有成熟的算子支持。

**量化范围:**

| 模块 | 量化策略 | 说明 |
|------|----------|------|
| Attention 层 (`fused_qkv_a_proj`, `q_b_proj`, `kv_b_proj`, `o_proj`) | W4A16 | 所有线性层 4-bit |
| 路由专家 MLP (`gate_proj`, `up_proj`, `down_proj`) × 256 | W4A16 | 模型主体，压缩收益最大 |
| 共享专家 MLP | W4A16 | 使用 `intermediate_size=18432`，体积大于路由专家 |
| Dense MLP (前 3 层) | W4A16 | 同上 |
| Embedding / LM Head | FP16 保留 | 精度敏感，不量化 |
| Gate (路由器) | FP16 保留 | 路由精度影响 expert 选择质量 |

### 2.3 显存占用建模

**权重计算 (W4A16, 每参数 0.5 bytes):**

```
Attention 层 (每层, W4A16):
  fused_qkv_a_proj:  7168 × 2112  × 0.5 = 7.6 MB
  q_b_proj:          1536 × 24576 × 0.5 = 18.9 MB
  kv_b_proj:          512 × 32768 × 0.5 = 8.4 MB
  o_proj:           16384 × 7168  × 0.5 = 58.7 MB
  LayerNorms:       ~0.1 MB
  小计:                                   ~93.5 MB/层

Dense MLP 层 (前 3 层, W4A16):
  gate + up + down:  3 × 7168 × 18432 × 0.5 = 198 MB/层

路由专家 (每 MoE 层, W4A16):
  每专家:  3 × 7168 × 2048 × 0.5 = 22.0 MB
  256 专家:  256 × 22.0 = 5,512 MB ≈ 5.38 GB/层

共享专家 (每 MoE 层, W4A16):
  3 × 7168 × 18432 × 0.5 = 198 MB/层

Gate (每 MoE 层, FP16):
  7168 × 256 × 2 = 3.67 MB/层

全模型汇总:
  3 Dense 层:   3 × (93.5 + 198) = 874 MB
  58 MoE 层:    58 × (93.5 + 5512 + 198 + 3.67) = 58 × 5807 = 330 GB
  Embedding:    129280 × 7168 × 2 = 1.85 GB (FP16)
  LMHead:       7168 × 129280 × 2 = 1.85 GB (FP16)
  ────────────────────────────────────────
  总权重量:     ~334 GB (W4A16)
```

### 2.4 并行策略设计

**核心原则 (来自 [parallel.py](../vllm/vllm/config/parallel.py) L116-118):**

> 当 `enable_expert_parallel=True` 时，MoE 层的专家按 `TP × DP` 的乘积分片 (EP 替代 TP)，Attention 层仍按 TP 切分。

**PD 分离下的差异化拓扑设计:**

Prefill 和 Decode 对资源的需求截然不同，因此采用完全不同的并行策略:

| 维度 | Prefill | Decode | 设计理由 |
|------|---------|--------|----------|
| **TP** | **8** | **2** | Prefill KV 临时性, TP=8 利用节点内全带宽; Decode KV 持久化, TP=2 控制 KV 复制 |
| **DP** | **4** | **8** | Prefill 需适度并发; Decode 需最大化 DP 扩展 KV Cache 容量 |
| **EP** | **32** (=4×8) | **16** (=8×2) | 自动等于 DP × TP, 专家在全实例范围分片 |
| **每卡专家数** | 256/32 = **8** | 256/16 = **16** | Prefill 卡数多, 每卡专家少, 通信少 |
| **卡数** | 32 (4 台) | 16 (2 台) | Prefill 计算密集分配更多卡 |
| **总卡数** | **48** | | 6 台机器, 余量 0~4 台弹性扩容 |

**为什么 Prefill 可以用 TP=8 而 Decode 不能?**

```
MLA 的 KV Cache 在 TP 下的行为:
  MLA 将 KV 压缩为 kv_lora_rank=512 的潜在向量。
  TP 切分 Q 头时, 每个 TP rank 仍需访问完整的压缩 KV。
  因此 KV Cache 在 TP ranks 之间被完整复制。

  TP=8 → KV 复制 8 份 (每份完整)
  TP=2 → KV 复制 2 份
  TP=1 → KV 零复制 (最优)

Prefill 场景:
  KV Cache 是临时性的 → 计算完即通过 Mooncake 发给 Decode
  TP=8 带来的 8× KV 复制 = 瞬时显存压力, 不构成持久负担
  收益: 节点内 8 卡 TP 带宽充分, Prefill 计算吞吐最大化
  → TP=8 是正确的选择

Decode 场景:
  KV Cache 是持久性的 → 整个 Decode 生命周期常驻显存
  TP=8 意味着每张卡存储完整的 KV 副本, 8 卡中有 7 份是冗余的
  TP=2 意味着每张卡存储完整的 KV 副本, 2 卡中仅 1 份冗余
  → TP=2 是 KV 容量与计算并行度的最优平衡
```

### 2.5 显存预算细项

**Prefill 节点 (每卡, DP=4, TP=8, EP=32):**

```
总显存:                               64 GB

权重 (W4A16):
  Attention (TP=8):   93.5 MB/层 × 61 层 / 8 =     713 MB
  Dense MLP (TP=8):   198 MB/层  × 3 层  / 8 =      74 MB
  路由专家 (EP=32):   5512 MB/层 × 58 层 / 32 =  10,013 MB  (~9.8 GB)
  共享专家 (TP=8):    198 MB/层  × 58 层 / 8 =    1,436 MB  (~1.4 GB)
  Gate (TP=8):        3.67 MB/层 × 58 层 / 8 =       27 MB
  Embed+LMHead (TP=8):                             463 MB
  ────────────────────────────────────────────────
  权重小计:                                      ~12.7 GB

Activation 缓冲:                          ~6 GB  (chunked prefill 大 batch)
运行时 (框架/通信缓冲):                    ~3 GB
────────────────────────────────────────────────
可用于 KV Cache:                          ~42.3 GB
```

Prefill 的 KV Cache 是**临时性**的，42 GB 绰绰有余。一个 128K token 的请求 KV Cache = 128K × 656 bytes × 61 layers ≈ 5.1 GB, 即使同时 Prefill 8 个请求也不到一半。

**Decode 节点 (每卡, DP=8, TP=2, EP=16):**

```
总显存:                               64 GB

权重 (W4A16):
  Attention (TP=2):   93.5 MB/层 × 61 层 / 2 =   2,852 MB  (~2.8 GB)
  Dense MLP (TP=2):   198 MB/层  × 3 层  / 2 =     297 MB
  路由专家 (EP=16):   5512 MB/层 × 58 层 / 16 = 19,981 MB  (~19.5 GB)
  共享专家 (TP=2):    198 MB/层  × 58 层 / 2 =   5,742 MB  (~5.6 GB)
  Gate (TP=2):        3.67 MB/层 × 58 层 / 2 =     107 MB
  Embed+LMHead (TP=2):                            1,850 MB  (~1.8 GB)
  ────────────────────────────────────────────────
  权重小计:                                      ~31.5 GB

Activation 缓冲:                          ~2 GB
运行时 (框架/通信缓冲):                    ~3 GB
────────────────────────────────────────────────
可用于 KV Cache:                          ~27.5 GB
```

**Decode KV Cache 容量分析:**

```
fp8_ds_mla 格式: 656 bytes/token/层 (来自 kv_cache_interface.py L284)
61 层合计: 656 × 61 = 40,016 bytes/token ≈ 39 KB/token

每卡 KV 容量: 27.5 GB / 40,016 bytes = 707K tokens

TP=2 影响: KV Cache 在 2 个 TP rank 间完整复制,
  每个 DP 组的有效容量 = 单卡容量 (两张卡存相同的 KV)

DP=8, 总 KV 容量: 8 × 707K = 5.66M tokens

业务需求:
  40 并发 × 32K 平均 = 1.28M tokens  → 余量 4.4× ✓
  40 并发 × 128K 最坏 = 5.12M tokens → 余量 10%  (需 prefix cache 辅助)

Prefix Cache 加持 (预估 30% 命中率):
  实际 KV 需求: 5.12M × 0.7 = 3.58M → 余量 58% ✓
```

**结论:** Decode TP=2 + DP=8 的拓扑在 fp8_ds_mla 下可满足 40 并发 128K 长尾需求。如果没有 prefix cache 命中，长尾场景余量紧张（10%），建议将 1-2 台弹性机器加入 Decode 集群（DP 扩至 10-12）。

### 2.6 EP 通信方法选择

根据 [ascend_forward_context.py](../vllm-ascend/vllm_ascend/ascend_forward_context.py) 中 `select_moe_comm_method()` 的选择逻辑:

| 场景 | 通信方法 | 依据 |
|------|----------|------|
| Prefill (EP=32, 大 token batch) | `ALLTOALL` | Token 数量大，序列级分发效率高 |
| Decode (EP=16, 小 token batch) | `MC2` | 每 step 仅 1 token/seq，MC2 融合通信+计算 |

MC2 层级通信 (`enable_mc2_hierarchy_comm`) 在跨节点 EP 场景下启用 ROCE 加速，对 Prefill 4 节点 EP=32 和 Decode 2 节点 EP=16 拓扑至关重要。

**Token Dispatcher 映射 (来自 [token_dispatcher.py](../vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py)):**

- Prefill → `TokenDispatcherWithAll2AllV`: `npu_moe_token_permute`/`npu_moe_token_unpermute` + all-to-all
- Decode → `TokenDispatcherWithMC2`: `npu_moe_distribute_dispatch_v2`/`npu_moe_distribute_combine_v2`

---

## 3. PD 分离架构设计

### 3.1 节点角色定义

**Prefill 节点 (KV Producer)**

- **职责:** 接收原始 prompt, 执行完整的 Prefill 计算, 生成 KV Cache
- **特征:** 计算密集型, KV Cache 临时性 (计算完即发送)
- **并行策略:** DP=4, TP=8, EP=32 (32 卡, 4 台机器)
- **量化方案:** W4A16 权重 + W8A8 动态激活量化
  - 可使用 `W8A8_MIX` 方案 ([w8a8_pdmix.py](../vllm-ascend/vllm_ascend/quantization/methods/w8a8_pdmix.py)): Prefill 走 Dynamic W8A8, Decode 走 Static W8A8
- **调度参数:**
  - `max_num_batched_tokens`: 114688 (chunked prefill, 每 chunk 最多 112K tokens)
  - `long_prefill_token_threshold`: 8192 (长文本分块阈值)
  - `max_num_seqs`: 64

**Decode 节点 (KV Consumer)**

- **职责:** 接收 KV Cache, 执行自回归生成
- **特征:** 显存密集型 (持久化 KV Cache), 延迟敏感
- **并行策略:** DP=8, TP=2, EP=16 (16 卡, 2 台机器)
- **量化方案:** W4A16 权重 + fp8_ds_mla KV Cache 量化
- **调度参数:**
  - `max_num_seqs`: 40
  - `gpu_memory_utilization`: 0.95

### 3.2 KV Cache 传输: Mooncake Layerwise Connector

**选择理由:**

| 方案 | 实现 | 特点 |
|------|------|------|
| `MooncakeConnector` | [mooncake_connector_v1.py](../Mooncake/mooncake-wheel/mooncake/mooncake_connector_v1.py) | 端到端整块传输 |
| `MooncakeLayerwiseConnector` | [mooncake_layerwise_connector.py](../vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py) | 逐层流水线传输 |

**推荐: MooncakeLayerwiseConnector** — 逐层流水线，Prefill 计算第 N+1 层 KV 的同时传输第 N 层 KV，掩蔽传输延迟。

**配置参数 (参考 [epd_disaggregated_guide.md](../vllm-ascend/examples/epd_disaggregated/epd_disaggregated_guide.md)):**

Prefill 端 `kv-transfer-config`:

```json
{
  "kv_connector": "MooncakeLayerwiseConnector",
  "kv_role": "kv_producer",
  "kv_port": "50001",
  "engine_id": "0",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_layerwise_connector",
  "kv_connector_extra_config": {
    "use_ascend_direct": true,
    "prefill": {
      "dp_size": 4,
      "tp_size": 8
    },
    "decode": {
      "dp_size": 8,
      "tp_size": 2
    }
  }
}
```

Decode 端 `kv-transfer-config` (仅 `kv_role` 和 `engine_id` 不同):

```json
{
  "kv_connector": "MooncakeLayerwiseConnector",
  "kv_role": "kv_consumer",
  "kv_port": "50002",
  "engine_id": "1",
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_layerwise_connector",
  "kv_connector_extra_config": {
    "use_ascend_direct": true,
    "prefill": {
      "dp_size": 4,
      "tp_size": 8
    },
    "decode": {
      "dp_size": 8,
      "tp_size": 2
    }
  }
}
```

> **关键:** `prefill.dp_size/tp_size` 和 `decode.dp_size/tp_size` 必须在两端配置中一致，Mooncake 据此计算跨实例的 KV Cache 地址映射。

**数据平面传输流:**

```
Prefill Worker (TP=8)             Decode Worker (TP=2)
    │                                    │
    │  ZMQ: metadata                     │
    │  (req_id, block_ids, base_addr)    │
    │───────────────────────────────────>│
    │                                    │
    │  RDMA Write: KV Cache              │
    │  (逐层流水线, layer N 计算          │
    │   与 layer N-1 传输重叠)            │
    │═══════════════════════════════════>│
    │                                    │
    │  ZMQ: transfer_complete             │
    │───────────────────────────────────>│
    │                                    │
```

**KV 传输量估算:**

```
单请求 KV Cache (MLA, fp8_ds_mla):
  32K tokens × 656 bytes × 61 layers = 1.29 GB
  128K tokens × 656 bytes × 61 layers = 5.14 GB

Mooncake RDMA 吞吐: ~87 GB/s (200Gbps RoCE, 4×200G)
  32K 请求传输延迟: ~15 ms (vs Prefill 计算 ~10s, 完全可忽略)
  128K 请求传输延迟: ~60 ms (vs Prefill 计算 ~40s, 完全可忽略)

Layerwise 流水线进一步掩蔽: 实际传输延迟趋近于 0。
```

### 3.3 PD 资源配比方案

**配比推导:**

```
Prefill 吞吐 (32K input, DP=4, TP=8):
  每 DP 组: TP=8 节点内全带宽 Prefill
  Chunked Prefill, max_batched_tokens=114688
  可同时处理 114688/32768 ≈ 3.5 个请求
  DP=4, 总并发 Prefill: 4 × 3.5 = 14 个请求
  预估单次 Prefill: ~10s (含 EP=32 跨 4 节点通信开销)
  Prefill 吞吐: 14 / 10 = 1.4 req/s

Decode 吞吐 (DP=8, TP=2):
  40 并发, avg output 1K tokens
  每 step: 40 tokens (batch=40)
  每 step 延迟目标: <80ms (TPOT)
  预估: ~50ms/step (EP=16 跨节点 MC2)
  Decode 吞吐: 40 / (1000 × 0.05) = 0.8 req/s

稳态平衡:
  Prefill 产能: 1.4 req/s
  Decode 消费:  0.8 req/s
  Prefill 余量: 43%

  → 4:2 的 P:D 机器配比在当前业务下 Prefill 有充足余量。
  → 如果输入变长 (平均 64K+), Prefill 压力增大, 可将弹性机器加入 Prefill。
```

**资源配比调整矩阵:**

| 业务特征变化 | 调整方向 | 操作 |
|-------------|----------|------|
| 平均输入增长至 64K+ | Prefill 不足 | 将弹性机器加入 Prefill (DP+2) |
| 长尾 128K 占比 > 20% | Decode KV 不足 | 将弹性机器加入 Decode (DP+4) |
| 并发目标提升至 60+ | 两端均不足 | 扩集群至 10 台, Prefill +2, Decode +2 |
| 平均输出增长至 2K+ | Decode 吞吐瓶颈 | 增加 Decode DP 或降低 max_num_seqs |

### 3.4 动态扩缩容逻辑

```
监控指标:
  - Prefill 队列深度 (Router 侧)
  - Decode KV Cache 使用率
  - 平均 TTFT / TPOT

扩容触发:
  - TTFT > SLA 且 Prefill 队列积压 → 增加Prefill DP (加入弹性机器)
  - TPOT > SLA 且 Decode KV > 80% → 增加Decode DP
  - KV Cache OOM 频率上升 → 紧急增加 Decode 节点

缩容触发:
  - Prefill 利用率 < 30% 持续 5 min → 减少Prefill DP rank
  - Decode 利用率 < 40% 持续 10 min → 减少Decode DP rank

实现方式:
  - 基于 Router 的 ZMQ Service Discovery (vllm_service_discovery.rs)
  - 新实例注册后自动纳入调度池
  - 实例下线前 drain 现有请求, 通过 Circuit Breaker 隔离
```

---

## 4. 路由网关架构

### 4.1 架构定位

Router 是整个系统的**请求调度中枢**, 基于 Rust 实现 (代码库 [router/](../router/))。其核心职责:

1. 接收用户 HTTP/gRPC 请求
2. 根据策略选择最优 Prefill 节点和 Decode 节点
3. 编排两阶段请求流 (Prefill → KV Transfer → Decode)
4. 汇聚响应返回用户

### 4.2 状态流转

```
Client Request
      │
      ▼
┌──────────────────────────────────────────────────┐
│                   Router                         │
│                                                  │
│  1. Tokenize input (extract text for prefix)     │
│  2. Prefix Match (Radix Tree per Decode worker)  │
│  3. Select Prefill Worker (cache_aware policy)   │
│  4. Select Decode Worker  (power_of_two policy)  │
│                                                  │
│  ┌─────────── Stage 1: Prefill ───────────┐      │
│  │  POST /v1/completions                  │      │
│  │  Headers:                              │      │
│  │    X-Request-Id: ___prefill_addr_{zmq} │      │
│  │                ___decode_addr_{zmq}_{id}│      │
│  │    X-data-parallel-rank: {rank}        │      │
│  │  Body: max_tokens=1, stream=false      │      │
│  │        kv_transfer_params: {           │      │
│  │          do_remote_decode: true        │      │
│  │        }                               │      │
│  └───────────────┬───────────────────────┘      │
│                  │                               │
│    Prefill Response: kv_transfer_params          │
│    (remote_engine_id, remote_block_ids, ...)     │
│                  │                               │
│  ┌─────────── Stage 2: Decode ───────────┐      │
│  │  POST /v1/completions                 │      │
│  │  Body: original request               │      │
│  │        kv_transfer_params: {           │      │
│  │          do_remote_prefill: true,      │      │
│  │          remote_engine_id,             │      │
│  │          remote_block_ids,             │      │
│  │          remote_host, remote_port      │      │
│  │        }                               │      │
│  └───────────────┬───────────────────────┘      │
│                  │                               │
│    Decode Response (streaming)                    │
│                  │                               │
└──────────────────┼──────────────────────────────┘
                   │
            Client Response (stream)
```

**关键代码路径:** [vllm_pd_router.rs](../router/src/routers/http/vllm_pd_router.rs) `process_vllm_two_stage_request()` (L856)

### 4.3 KV Cache 亲和性调度算法

**核心数据结构: 并发 Radix Tree** ([tree.rs](../router/src/tree.rs))

```
特性:
  - 多租户: 每棵子树按 worker_id 分区
  - 字符级匹配: 避免额外 tokenization 开销
  - LRU 驱逐: 基于全局原子 epoch 计数器, O(1) 更新
  - 锁并发: DashMap 分片锁, 读无阻塞

匹配流程 (cache_aware.rs L229-358):
  match_rate = matched_char_count / input_char_count

  if match_rate > cache_threshold (默认 0.5):
      → 路由到匹配度最高的 worker (KV Cache 命中)
  else:
      → 路由到树最小的 worker (最多可用缓存空间)

  负载失衡检测:
      is_imbalanced = (max_load - min_load) > balance_abs_threshold
                   && max_load > min_load × balance_rel_threshold
      若失衡 → 优先路由到最轻负载 worker, 忽略缓存亲和性
```

**对 DeepSeek V3.2 的特殊意义:**

MLA (kv_lora_rank=512) 的 KV Cache 体积仅为标准 MHA 的 ~1/20。Prefix Cache 命中不仅节省 Decode 侧的显存，更直接跳过对应 Prefill 计算 — 一个 10K token 系统 prompt 命中可节省 ~6.5 MB KV Cache 的 RDMA 传输和重计算。

### 4.4 负载均衡策略

**双策略栈设计:**

| 工作池 | 策略 | 理由 |
|--------|------|------|
| Prefill | `cache_aware` | 长文本 prefix 命中率高, 减少 Prefill 计算量 |
| Decode | `power_of_two` | Decode 延迟敏感, Po2 随机选 2 取最优, O(1) 复杂度 |

**负载追踪:**

```rust
// worker.rs - 每个 Worker 维护:
AtomicBool    is_healthy          // 健康状态
AtomicUsize   load                // 当前活跃请求数
AtomicUsize   processed_count     // 累计处理数
CircuitBreaker circuit_breaker    // 熔断器 (Closed/Open/HalfOpen)
```

**DP 感知路由** ([dp_utils.rs](../router/src/routers/http/dp_utils.rs)):

当 `intra_node_data_parallel_size > 1` 时, Router 将每个 worker URL 展开为多个 DP rank:

```
Prefill (DP=4): http://prefill-host:33003 → [@0, @1, @2, @3]
Decode  (DP=8): http://decode-host:33006  → [@0, @1, ..., @7]
请求头注入: X-data-parallel-rank: {rank}
```

### 4.5 网关 I/O 与状态同步

**连接模型:**

```
Router ←──── HTTP/1.1 Keep-Alive ────→ Client
Router ←──── HTTP/1.1 Connection Pool ────→ Prefill Workers
Router ←──── HTTP/1.1 Connection Pool ────→ Decode Workers
Router ←──── ZMQ REQ/ROUTER ────→ vLLM Worker Bootstrap (topology discovery)
```

**流式响应处理:**

Decode 阶段采用 SSE (Server-Sent Events) 流式传输。Router 作为透明代理:
- 非 logprobs 模式: Prefill 响应 drain 在后台 (fire-and-forget)
- logprobs 模式: 同时等待 Prefill 和 Decode 响应, 合并 logprobs

**拓扑发现 (Service Discovery):**

两种模式:

1. **静态配置:** 通过 `--prefill-urls` / `--decode-urls` 直接指定
2. **动态发现 (推荐):** ZMQ listener ([vllm_service_discovery.rs](../router/src/routers/http/vllm_service_discovery.rs))
   - Worker 启动后通过 ZMQ 发送 MessagePack 注册消息 (type="P"/"D", HTTP addr, ZMQ addr)
   - 心跳超时 5s, 自动摘除失活节点
   - 适用于 Kubernetes Pod 自动伸缩场景

**容错机制:**

```
Circuit Breaker:
  - 连续 N 次失败 (默认 3) → Open (拒绝新请求)
  - 超时后 → HalfOpen (放行探测请求)
  - 探测成功 M 次 (默认 2) → Closed (恢复正常)

Retry Executor:
  - 仅重试 5xx (服务端错误), 不重试 4xx
  - 指数退避 + 抖动

Rate Limiter:
  - Token Bucket 算法
  - 按 worker 维度限流
```

---

## 5. 实施与测试路径规划

### 5.1 阶段 1: 单节点功能验证 (Week 1-2)

**目标:** 验证 W4A16 量化 + TP=8 在单节点 8 卡上的基本功能。

```bash
# 单节点 TP=8 (无 DP, 无 EP)
vllm serve deepseek-v32-w4a16 \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --quantization w4a16 \
  --gpu-memory-utilization 0.95 \
  --enforce-eager
```

**验证项:**
- [ ] W4A16 权重加载正确性 (与 FP16 baseline 对比 perplexity)
- [ ] TP=8 下 MLA Attention 正确性
- [ ] 32K 序列 E2E 推理正确性
- [ ] MoE 路由正确性 (256 expert, top-8)
- [ ] fp8_ds_mla KV Cache 正确性

### 5.2 阶段 2: 多节点 EP + DP 验证 (Week 3-4)

**目标:** Prefill 拓扑 (DP=4, TP=8, EP=32) 跨节点通信验证。

```bash
# Prefill 拓扑验证: 4 节点 × 8 卡 = 32 卡
torchrun --nproc_per_node=8 --nnodes=4 \
  vllm serve deepseek-v32-w4a16 \
  --tensor-parallel-size 8 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --quantization w4a16 \
  --cache-dtype fp8_ds_mla \
  --max-model-len 131072

# Decode 拓扑验证: 2 节点 × 8 卡 = 16 卡
torchrun --nproc_per_node=8 --nnodes=2 \
  vllm serve deepseek-v32-w4a16 \
  --tensor-parallel-size 2 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --quantization w4a16 \
  --cache-dtype fp8_ds_mla \
  --max-model-len 131072
```

**验证项:**
- [ ] MC2 层级通信 (`enable_mc2_hierarchy_comm`) ROCE 正确性
- [ ] EP=32 跨 4 节点 All2All 延迟基准测试
- [ ] EP=16 跨 2 节点 MC2 延迟基准测试
- [ ] DP=4/DP=8 下 BalanceScheduler 负载均衡
- [ ] 显存占用验证 (是否符合第 2.5 节预估)
- [ ] 128K 长序列推理正确性

### 5.3 阶段 3: PD 分离 + Router 集成 (Week 5-6)

**目标:** Prefill-Decode 分离全链路打通。

```bash
# 1. 启动 Prefill 实例 (4 台机器, 32 卡, DP=4 TP=8 EP=32)
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=0 \
  --master_addr prefill-node0 --master_port 29500 \
  vllm serve deepseek-v32-w4a16 \
  --port 33003 \
  --tensor-parallel-size 8 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --quantization w4a16 \
  --cache-dtype fp8_ds_mla \
  --max-model-len 131072 \
  --max-num-batched-tokens 114688 \
  --max-num-seqs 64 \
  --kv-transfer-config '{
    "kv_connector": "MooncakeLayerwiseConnector",
    "kv_role": "kv_producer",
    "kv_port": "50001",
    "engine_id": "0",
    "kv_connector_module_path": "vllm_ascend.distributed.mooncake_layerwise_connector",
    "kv_connector_extra_config": {
      "use_ascend_direct": true,
      "prefill": {"dp_size": 4, "tp_size": 8},
      "decode": {"dp_size": 8, "tp_size": 2}
    }
  }'

# 2. 启动 Decode 实例 (2 台机器, 16 卡, DP=8 TP=2 EP=16)
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr decode-node0 --master-port 29500 \
  vllm serve deepseek-v32-w4a16 \
  --port 33006 \
  --tensor-parallel-size 2 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --quantization w4a16 \
  --cache-dtype fp8_ds_mla \
  --max-model-len 131072 \
  --max-num-seqs 40 \
  --gpu-memory-utilization 0.95 \
  --kv-transfer-config '{
    "kv_connector": "MooncakeLayerwiseConnector",
    "kv_role": "kv_consumer",
    "kv_port": "50002",
    "engine_id": "1",
    "kv_connector_module_path": "vllm_ascend.distributed.mooncake_layerwise_connector",
    "kv_connector_extra_config": {
      "use_ascend_direct": true,
      "prefill": {"dp_size": 4, "tp_size": 8},
      "decode": {"dp_size": 8, "tp_size": 2}
    }
  }'

# 3. 启动 Router (可部署在任一管理节点, 无需 GPU)
vllm-router \
  --vllm-pd-disaggregation \
  --prefill-urls http://prefill-node0:33003 \
  --decode-urls http://decode-node0:33006 \
  --prefill-policy cache_aware \
  --decode-policy power_of_two \
  --intra-node-data-parallel-size 4 \
  --port 8001
```

**验证项:**
- [ ] Mooncake Layerwise KV Transfer (Prefill TP=8 → Decode TP=2 跨拓扑传输)
- [ ] 两阶段请求流 (Router → Prefill → Decode)
- [ ] Prefix Cache 亲和性调度验证
- [ ] 40 并发压力测试 (TTFT / TPOT SLO 达标)
- [ ] 长尾 128K 请求的 KV Cache 容量
- [ ] 流式响应 + Logprobs 合并正确性
- [ ] 节点故障恢复 (Circuit Breaker + Retry)

### 5.4 阶段 4: 性能调优 (Week 7-8)

**调优维度:**

| 维度 | 调优项 | 目标 |
|------|--------|------|
| MoE 通信 | `multistream_overlap_shared_expert` | 共享专家计算与路由专家通信重叠 |
| MoE 通信 | `multistream_overlap_gate` | FlashCommon3 Gate 与共享专家重叠 |
| EP 通信 | `enable_mc2_hierarchy_comm` | 跨节点 ROCE 层级通信优化 |
| KV Cache | `enable_kv_nz` | MLA NZ 格式优化 (Decode 节点) |
| Prefill | `long_prefill_token_threshold` 调参 | 平衡 TTFT 与 chunked prefill 开销 |
| Decode | `max_num_seqs` 调参 | 平衡吞吐与 TPOT |
| 路由 | `cache_threshold` 调参 | 平衡命中率与负载均衡 |
| 显存 | `mix_placement` | 混合布局共享/路由专家减少显存碎片 |

**性能基准:**

```
目标 SLO:
  TTFT (32K input):  < 15s
  TTFT (128K input): < 45s
  TPOT:              < 80ms
  40 并发吞吐:        > 0.5 req/s (含 1K avg output)

监控:
  - Router Prometheus metrics (request latency, error rate, cache hit rate)
  - vLLM metrics (num_running_seqs, gpu_cache_usage, num_preemptions)
  - Mooncake Transfer Engine throughput & latency
```

---

## 附录 A: 关键配置参数速查

### vLLM 启动参数

| 参数 | Prefill | Decode | 说明 |
|------|---------|--------|------|
| `--tensor-parallel-size` | **8** | **2** | Prefill 大 TP 榨节点带宽, Decode 小 TP 控 KV 复制 |
| `--data-parallel-size` | **4** | **8** | Prefill 适度 DP, Decode 大 DP 扩 KV 容量 |
| `--enable-expert-parallel` | ✓ | ✓ | EP 替代 TP 用于 MoE 层 |
| 有效 EP | **32** (=4×8) | **16** (=8×2) | 自动等于 DP × TP |
| `--quantization` | w4a16 | w4a16 | 4-bit 权重量化 |
| `--cache-dtype` | fp8_ds_mla | fp8_ds_mla | MLA KV Cache FP8 量化 |
| `--max-model-len` | 131072 | 131072 | 最大序列长度 |
| `--max-num-batched-tokens` | 114688 | — | Prefill chunk 大小 |
| `--max-num-seqs` | 64 | 40 | 最大并发 |
| `--gpu-memory-utilization` | 0.90 | 0.95 | 显存利用率 |

### vllm-ascend AscendConfig 参数

| 参数 | Prefill | Decode | 说明 |
|------|---------|--------|------|
| `enable_shared_expert_dp` | true | true | 共享专家跨 DP 独立计算 |
| `multistream_overlap_shared_expert` | true | true | 共享专家计算与路由通信重叠 |
| `multistream_overlap_gate` | true | false | FlashCommon3 Gate 重叠 (仅 Prefill 受益) |
| `enable_mc2_hierarchy_comm` | true | true | 跨节点 ROCE 层级 MC2 |
| `enable_kv_nz` | false | true | Decode 节点 MLA NZ 格式 |
| `enable_sparse_c8` | true | true | Indexer INT8 量化 (V3.2) |

### Router 启动参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--vllm-pd-disaggregation` | — | 启用 vLLM PD 分离模式 |
| `--prefill-policy` | cache_aware | Prefill 池缓存感知调度 |
| `--decode-policy` | power_of_two | Decode 池 Po2 负载均衡 |
| `--intra-node-data-parallel-size` | 4 (P) / 8 (D) | DP 展开因子 |
| `--cache-threshold` | 0.5 | 前缀匹配阈值 |

---

## 附录 B: 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| W4A16 量化精度损失超出预期 | 中 | 高 | 先用 W8A8 FP8 上线, W4A16 作为优化项灰度切换 |
| EP=32 跨 4 节点通信延迟过高 | 中 | 高 | 启用 MC2 hierarchy + ROCE; 基准测试确定实际延迟 |
| fp8_ds_mla KV Cache 数值问题 | 低 | 高 | 回退至 bf16 MLA KV, 牺牲并发容量 |
| Mooncake RDMA 在昇腾环境不稳定 | 中 | 中 | 准备 NIXL Connector 作为备选; CPU Offload 兜底 |
| Decode 128K 长尾 KV 不足 | 中 | 高 | 将弹性机器加入 Decode (DP 8→12); 启用 prefix cache |
| Prefill TP=8 下 MLA KV 瞬时显存压力 | 低 | 中 | 减小 max_num_batched_tokens; 限制并发 Prefill 数 |
| 集群网络带宽不足 (RoCE) | 低 | 高 | 基准测试 RoCE 吞吐; 确认 200Gbps+ 链路 |

---

## 附录 C: 代码库关键文件索引

| 模块 | 文件 | 作用 |
|------|------|------|
| **vllm-ascend** | `vllm_ascend/ops/fused_moe/fused_moe.py` | MoE 层主类, EP/DP/TP 拓扑初始化 |
| | `vllm_ascend/ops/fused_moe/moe_comm_method.py` | 4 种通信方法 (ALLGATHER/MC2/ALLTOALL/FUSED_MC2) |
| | `vllm_ascend/ops/fused_moe/token_dispatcher.py` | Token 分发/汇聚 (MC2/AllGather/All2All) |
| | `vllm_ascend/ops/fused_moe/prepare_finalize.py` | MoE 前后处理, DP/EP AllGather/ReduceScatter |
| | `vllm_ascend/ascend_config.py` | 昇腾平台全部配置项 |
| | `vllm_ascend/ascend_forward_context.py` | MoE 通信方法自动选择逻辑 |
| | `vllm_ascend/quantization/methods/w4a16.py` | W4A16 量化实现 |
| | `vllm_ascend/quantization/methods/w8a8_pdmix.py` | PD 混合量化 (Prefill Dynamic / Decode Static) |
| | `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py` | Mooncake 逐层 KV 传输 |
| | `vllm_ascend/patch/platform/patch_kv_cache_interface.py` | MLA Sparse C8 KV Cache 扩展 |
| **vllm** | `vllm/model_executor/models/deepseek_v2.py` | DeepSeek V3.2 模型定义 (MLA + MoE) |
| | `vllm/v1/kv_cache_interface.py` | KV Cache 规格 (含 MLAAttentionSpec, fp8_ds_mla) |
| | `vllm/config/parallel.py` | 并行配置 (DP/TP/EP/PP/EPLB) |
| | `vllm/v1/core/sched/scheduler.py` | 统一调度器 (chunked prefill + KV connector) |
| | `vllm/distributed/kv_transfer/kv_connector/factory.py` | KV Connector 注册表 |
| **Router** | `router/src/routers/http/vllm_pd_router.rs` | vLLM PD 两阶段路由核心逻辑 |
| | `router/src/policies/cache_aware.rs` | Radix Tree 缓存感知负载均衡 |
| | `router/src/policies/power_of_two.rs` | Po2 快速负载均衡 |
| | `router/src/tree.rs` | 并发 Radix Tree 实现 |
| | `router/src/routers/http/dp_utils.rs` | DP Rank URL 展开与请求头注入 |
| | `router/src/routers/http/vllm_service_discovery.rs` | ZMQ 动态拓扑发现 |
| | `router/src/core/circuit_breaker.rs` | 熔断器实现 |
| **Mooncake** | `mooncake-wheel/mooncake/mooncake_connector_v1.py` | vLLM KV Connector (RDMA 传输) |
| | `mooncake-pg/include/mooncake_backend.h` | PyTorch Process Group 后端 |
| | `mooncake-ep/include/mooncake_ep_buffer.h` | EP GPU-to-GPU 通信缓冲区 |
