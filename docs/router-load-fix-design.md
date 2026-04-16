# Router Load 获取方式修复设计

## 背景

Router 的 `power_of_two` 策略通过后台任务轮询 worker 获取负载，当前代码调用 `{worker_url}/get_load` 端点并解析 JSON 中的 `load` 字段。但 vLLM（包括 vllm-ascend）未提供此端点，导致负载查询始终失败，`power_of_two` 退化为随机选择。

## 方案

将负载获取方式从 `/get_load`（私有 JSON API）改为 `/metrics`（vLLM 原生 Prometheus 端点），解析 `vllm:num_requests_running` 和 `vllm:num_requests_waiting` 指标。

### 负载计算

```
load = running + waiting × 10
```

- `running`：正在处理的请求，权重 1
- `waiting`：排队等待的请求，权重 10（排队意味着资源已饱和，需要更积极避让）

### 改动范围

#### 1. `src/routers/http/pd_router.rs`

**新增辅助函数：**
```rust
/// 从 Prometheus 文本中解析指定指标值
fn parse_prometheus_metric(text: &str, metric_name: &str) -> Option<f64>
```
- 按行扫描，跳过注释和空行
- 匹配 `metric_name{...}` 或 `metric_name ` 格式
- 取最后一个空格分隔的 token 作为值

**替换 `get_worker_load` 函数：**

| | 旧 | 新 |
|--|---|---|
| 端点 | `{url}/get_load` | `{url}/metrics` |
| 响应格式 | JSON `{"load": N}` | Prometheus 文本 |
| 解析方式 | `serde_json` | 按行文本匹配 |
| 负载值 | `data["load"]` | `running + waiting * 10` |

**无需修改的部分：**
- `monitor_worker_loads_with_client` — 调用签名不变，仍返回 `HashMap<String, isize>`
- `update_loads` — 接口不变
- `power_of_two` policy — 逻辑不变，仅获得正确的 load 值
- `WorkerMetrics` / `update_gpu_cache_usage` — 不引入，保持最小改动

#### 2. `src/routers/http/router.rs`

同步修改普通模式（非 PD）的 `get_worker_load` 函数，逻辑相同。

### 不改动的文件

- `src/policies/` — 所有 policy 接口不变
- `src/config/` — 无需新增配置项
- `src/server.rs` — 无变化
- `src/main.rs` — 无变化

## 验证

1. 启动 vllm-ascend worker，确认 `curl http://worker:8000/metrics` 返回包含 `vllm:num_requests_running` 和 `vllm:num_requests_waiting`
2. 启动 router（`--decode-policy power_of_two`），观察日志中 `Worker loads updated` 输出非零值
3. 发送请求后确认负载值动态变化
4. 确认 `power_of_two` 路由选择偏向低负载 worker
