# miniQED — 一期设计文档

**日期:** 2026-06-07
**状态:** 已确认
**基于:** QED (proofQED/QED, MIT License)

---

## 1. 项目目标

基于 QED 框架开发 miniQED，一期目标：

- **学习用途**：通过模块化重构深入理解 QED 的 multi-agent 数学证明系统
- **国产化 LLM 接入**：用 DeepSeek + OpenCode Go 替代御三家（Claude/Codex/Gemini），大幅降低 token 成本
- **为二期铺路**：清晰的模块边界，便于后续接入工作流引擎（LangGraph/Prefect）和 Lean 形式化验证（Aristotle CLI）

二期规划（不在本文档范围）：
- 架构完全重构：工作流引擎替代手写循环
- Lean 形式化审查：通过 Aristotle CLI 验证证明的正确性
- 更多国产模型：GLM、Mimo、MiniMax

---

## 2. 项目结构

```
miniQED/
├── config.yaml                          # 三层配置：adapters / pipeline / workflow
├── config.example.yaml                  # 不含 API key 的示例配置
├── run.sh                               # 入口脚本
├── problem/
│   └── problem.tex                      # LaTeX 问题输入
├── prompts/                             # 从 QED 继承，适配国产模型
│   ├── literature_survey.md
│   ├── proof_search.md
│   ├── proof_verify_structural.md
│   ├── proof_verify_detailed.md
│   ├── proof_verify_easy.md
│   ├── proof_select.md
│   ├── verdict_proof.md
│   ├── brainstorm.md
│   └── proof_effort_summary.md
├── skill/
│   └── super_math_skill.md              # 证明方法论 system prompt
├── human_help/
│   ├── additional_prove_human_help_global.md
│   └── additional_verify_rule_global.md
├── mini_qed/
│   ├── __init__.py
│   ├── orchestrator.py                  # 顶层编排：Stage 0 → 1 → 2
│   ├── config.py                        # YAML 加载 + ${ENV_VAR} 展开 + 校验
│   ├── logging.py                       # PipelineLogger + TokenTracker
│   ├── utils.py                         # prompt 加载、文件检查、resume 检测
│   ├── adapters/                        # LLM 适配层（全新）
│   │   ├── __init__.py
│   │   ├── base.py                      #   AbstractLLMAdapter + LLMResponse
│   │   ├── openai_compatible.py         #   DeepSeek / GLM / MiniMax / Mimo / OpenCode
│   │   ├── anthropic_adapter.py         #   Claude (anthropic SDK, 一期占位)
│   │   ├── openai_adapter.py            #   GPT (openai SDK, 一期占位)
│   │   └── registry.py                  #   按 name 解析 + 实例化 adapter
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── stage0_survey.py             # Stage 0: 文献调研
│   │   ├── stage1_simple.py             # Stage 1: Simple 模式证明循环编排
│   │   └── stage2_summary.py            # Stage 2: 证明总结
│   └── steps/
│       ├── __init__.py
│       ├── brainstorm.py                # 多模型并行头脑风暴
│       ├── proof_search.py              # 单模型 / 多模型并行证明搜索
│       ├── verification.py              # structural / detailed / easy 验证
│       ├── verdict.py                   # 裁决：读验证报告 → DONE/CONTINUE
│       └── selection.py                 # 多模型时选最佳 proof
├── frozen/                              # 一期冻结的 QED 代码（不加载）
│   ├── decomposition_prover.py
│   └── decomposition_prompts/
├── tests/
│   ├── test_adapters.py
│   ├── test_stage0_survey.py
│   ├── test_stage1_simple.py
│   ├── test_verification.py
│   └── test_verdict.py
└── proof_output/                        # 运行时输出目录（gitignore）
```

### 对比 QED 原结构

| QED | miniQED |
|-----|---------|
| `code/pipeline.py` (2481 行单文件) | `orchestrator.py` (~200行) + `stages/` + `steps/` (各 60-200行) |
| `code/model_runner.py` (635行, 仅 CLI) | `adapters/` 目录 (每个文件 100-200行, 纯 HTTP/SDK) |
| `config.yaml` (平铺式) | 三层分离：adapters / pipeline / workflow |
| `code/decomposition_prover.py` | 移到 `frozen/`, 不加载 |
| 无类型化接口 | dataclass 驱动的显式接口 |

---

## 3. LLM Adapter 层

### 3.1 抽象接口

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    elapsed_s: float

class AbstractLLMAdapter(ABC):
    """所有 LLM 后端的统一接口"""

    @abstractmethod
    async def chat(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 65536,
        thinking: bool = False,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """发送 prompt，返回统一格式的响应"""
        ...
```

### 3.2 实现策略

**两条路径：**

- **路径 A: OpenAI 兼容** — 一份 `OpenAICompatibleAdapter` 覆盖所有国产模型（DeepSeek / GLM / MiniMax / Mimo / OpenCode）。通过 `AsyncOpenAI` 客户端调 `/v1/chat/completions`。
  - `thinking: true` → 注入 `extra_body={"thinking": {"type": "enabled"}}`
  - `reasoning_effort: "max"` → 注入 `extra_body={"reasoning_effort": "max"}`

- **路径 B: 官方 SDK** — 每个海外模型独立的 adapter 类。
  - `AnthropicAdapter` — 使用 `anthropic` SDK
  - `OpenAIAdapter` — 使用 `openai` SDK

### 3.3 注册机制

```python
# registry.py
def build_adapter(name: str, cfg: dict) -> AbstractLLMAdapter:
    adapter_type = cfg["type"]
    if adapter_type == "openai_compatible":
        return OpenAICompatibleAdapter(
            base_url=cfg["base_url"],
            api_key=os.path.expandvars(cfg["api_key"]),
            model=cfg.get("model", ""),
        )
    elif adapter_type == "anthropic_sdk":
        return AnthropicAdapter(
            api_key=os.path.expandvars(cfg["api_key"]),
            model=cfg.get("model", ""),
        )
    # ...etc
```

### 3.4 DeepSeek V4 模型参数

| 参数 | Pro | Flash |
|------|-----|-------|
| API model name | `deepseek-v4-pro` | `deepseek-v4-flash` |
| 价格 (输入/输出 per M) | $0.435 / $0.87 | $0.14 / $0.28 |
| 上下文 | 1M | 1M |
| Thinking 模式 | ✅ (thinking + reasoning_effort) | ✅ (thinking) |
| Max reasoning | ✅ `reasoning_effort="max"` | ❌ |
| 系统提示词 | ✅ | ✅ |

> **注意:** 旧名称 `deepseek-chat` 和 `deepseek-reasoner` 将于 2026/07/24 废弃。一期使用新名称。

---

## 4. 配置设计

### 4.1 三层结构

```yaml
# 第一层：Adapters — 只声明有哪些 LLM 可用
adapters:
  deepseek:
    type: openai_compatible
    base_url: https://api.deepseek.com/v1
    api_key: ${DEEPSEEK_API_KEY}

  opencode_go:
    type: openai_compatible
    base_url: https://api.opencode.ai/v1
    api_key: ${OPENCODE_API_KEY}

# 第二层：Pipeline — 各 agent 角色使用哪个 adapter
pipeline:
  max_proof_iterations: 9

  literature_survey:
    adapter: deepseek
    model: deepseek-v4-pro
    thinking: true
    reasoning_effort: high

  simple_mode:
    proof_search:
      adapter: deepseek
      model: deepseek-v4-pro
      thinking: true
      reasoning_effort: max

    structural_verifier:
      adapter: deepseek
      model: deepseek-v4-flash
      thinking: false

    detailed_verifier:
      adapter: deepseek
      model: deepseek-v4-flash
      thinking: false

    verdict:
      adapter: deepseek
      model: deepseek-v4-flash
      thinking: false

    brainstorm:
      enabled: false
      providers:
        - adapter: deepseek
          model: deepseek-v4-flash

    multi_model:
      enabled: false
      proof_search_providers:
        - adapter: deepseek
          model: deepseek-v4-pro
          thinking: true
          reasoning_effort: max
        - adapter: opencode_go
          model: claude-sonnet-4-6
      verification_providers:
        - adapter: deepseek
          model: deepseek-v4-flash
        - adapter: opencode_go
          model: gpt-5.1

  proof_summary:
    adapter: deepseek
    model: deepseek-v4-pro
    thinking: false

# 第三层（二期）：工作流引擎 + Lean
# workflow:
#   engine: langgraph
# lean:
#   enabled: true
```

### 4.2 设计原则

- **Adapter 与 Pipeline 分离** — adapter 管"怎么连"，pipeline 管"用哪个连 + 干什么"
- **Model 在 agent 层指定** — 同一个 adapter 下不同角色可用不同 model（prover 用 Pro，verifier 用 Flash）
- **`${ENV_VAR}` 自动展开** — API key 不写明文
- **Thinking 参数化** — `thinking` + `reasoning_effort` 透传给 adapter

### 4.3 配置校验

`config.py` 启动时：
1. 加载 YAML + 展开 `${ENV_VAR}`
2. 校验 agent 角色引用的 adapter name 必须在 `adapters` 节存在
3. 校验必填字段（`type`、`base_url`、`api_key`）
4. 无效配置产生精确到字段的错误消息

---

## 5. 模块拆分与数据流

### 5.1 模块职责

| 模块 | 职责 | 预估行数 |
|------|------|----------|
| `orchestrator.py` | 顶层编排 Stage 0 → 1 → 2 | ~200 |
| `stages/stage0_survey.py` | 文献调研：单 agent 调用，输出 difficulty + related_work | ~120 |
| `stages/stage1_simple.py` | Simple 模式循环编排，委托各 step | ~150 |
| `stages/stage2_summary.py` | 证明总结：读所有文件，输出 proof_effort_summary | ~100 |
| `steps/brainstorm.py` | 多模型并行 brainstorm，汇总思路 | ~80 |
| `steps/proof_search.py` | 单模型 / 多模型并行证明搜索 | ~180 |
| `steps/verification.py` | structural / detailed / easy 三条验证路径 | ~250 |
| `steps/verdict.py` | 裁决 agent | ~60 |
| `steps/selection.py` | 多模型证明选择 | ~80 |
| `adapters/` | 约 400 行（含 base / openai_compatible / registry / 占位 adapter）| ~400 |

### 5.2 数据对象

模块间传递 dataclass，每个 step 完成后同时落盘（用于人工查看和二期 resume）：

```python
@dataclass
class ProofResult:
    proof_text: str
    proof_file: str
    status_file: str
    scratch_pad: str
    model: str
    tokens: TokenUsage

@dataclass
class VerificationReport:
    verdict: str             # "PASS" | "FAIL"
    report_text: str
    report_file: str
    phase: str               # "structural" | "detailed" | "easy"
    model: str
    tokens: TokenUsage
```

### 5.3 数据流（一次典型 Round）

```
ProofResult ──→ run_structural_verification() ──→ list[VerificationReport]
                                                          │
                                              ┌─ all PASS ─→ run_detailed_verification()
                                              │                    │
                                              │              list[VerificationReport]
                                              │                    │
                                              │              run_verdict() ──→ DONE | CONTINUE
                                              │
                                              └─ any FAIL ──→ CONTINUE（下一轮）
```

---

## 6. 一期交付物与验收

### 6.1 交付物清单

| # | 模块 | 验收标准 |
|---|------|----------|
| 1 | `adapters/base.py` + `openai_compatible.py` + `registry.py` | `deepseek-v4-pro` 和 `deepseek-v4-flash` 单次调用成功，返回 `LLMResponse` |
| 2 | `adapters/anthropic_adapter.py` + `openai_adapter.py` (占位) | 接口完整，import + type check 通过 |
| 3 | `config.py` | 无效配置产生可读错误，精确到哪个 adapter 未定义 |
| 4 | `stages/stage0_survey.py` | 输入 problem.tex → 输出 `difficulty_evaluation.md` + `related_work.md` |
| 5 | `steps/proof_search.py` | 输入 problem + context → 输出 proof.md + proof_status.md |
| 6 | `steps/verification.py` | structural / detailed / easy 三条路径均可运行 |
| 7 | `steps/verdict.py` | 输入验证报告 → 输出 DONE 或 CONTINUE |
| 8 | `steps/selection.py` | 多模型下选中最佳 proof |
| 9 | `steps/brainstorm.py` | 多模型并行 brainstorm，输出各模型思路文件 |
| 10 | `stages/stage1_simple.py` | 端到端：LaTeX → 多轮迭代 → proof.md 或 max_iterations |
| 11 | `stages/stage2_summary.py` | 输出 `proof_effort_summary.md` |
| 12 | `orchestrator.py` | Stage 0 → 1 → 2 完整跑通 |
| 13 | `logging.py` | 输出 `AUTO_RUN_STATUS.md` + `TOKEN_USAGE.md` |

### 6.2 一期不做

- ❌ Decomposition 模式（代码保留在 `frozen/`，不加载）
- ❌ 中断恢复 / Resume
- ❌ Web UI
- ❌ GLM / Mimo / MiniMax adapter（接口支持，不配置不测试）
- ❌ Lean 形式化验证
- ❌ 工作流引擎

### 6.3 验收流程

```bash
# 1. 配置
cp config.example.yaml config.yaml
# 填入 DEEPSEEK_API_KEY、OPENCODE_API_KEY

# 2. 运行
bash run.sh

# 3. 期望输出
proof_output/
├── proof.md
├── proof_effort_summary.md
├── TOKEN_USAGE.md
├── related_info/
│   ├── difficulty_evaluation.md
│   └── related_work.md
└── verification/
    └── round_1/
        ├── proof_before_round.md
        ├── proof_status.md
        ├── scratch_pad.md
        └── ...
```

---

## 7. 关键设计决策记录

| 决策 | 理由 |
|------|------|
| OpenAI 兼容路径统一国产模型 | DeepSeek/GLM/MiniMax/Mimo 全部支持 `/v1/chat/completions`，一套代码覆盖 |
| 对象驱动 + 落盘快照（非纯文件驱动）| 学习项目优先清晰接口；落盘格式与 QED 兼容 |
| 保留 Brainstorm | 代码量小 (~80行)，展示多模型并行模式，学习价值高 |
| 冻结 Decomposition | 完整子系统，嵌套状态管理复杂，与一期核心循环解耦 |
| 一期不做 Resume | Resume 逻辑依赖稳定的文件布局，在快速迭代阶段不值得维护 |
| DeepSeek V4 Pro + Max 做证明，Flash 做验证 | Pro + reasoning_effort=max 适合深度推理；Flash 性价比高适合检查类任务 |
