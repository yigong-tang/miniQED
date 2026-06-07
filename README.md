# miniQED

基于 [QED](https://github.com/proofQED/QED) 的模块化多智能体数学证明系统。一期用 DeepSeek + OpenCode 替代 Claude/Codex/Gemini 御三家，大幅降低 token 成本。

## 快速开始

### 1. 安装依赖

```bash
pip install openai pyyaml
```

### 2. 配置 API Key

```bash
cp config.example.yaml config.yaml
```

编辑 `config.yaml`，设置环境变量后启动（推荐），或者直接把 key 写进去。

**方式 A：环境变量（推荐，key 不落盘）**

```bash
export DEEPSEEK_API_KEY=sk-your-deepseek-key
# 如果要用 OpenCode Go 套餐做辅助模型
export OPENCODE_API_KEY=your-opencode-key
```

`config.example.yaml` 里已经用 `${DEEPSEEK_API_KEY}` 引用环境变量，启动时自动展开。

**方式 B：直接写在 config.yaml（不推荐，注意不要 commit）**

```yaml
adapters:
  deepseek:
    type: openai_compatible
    base_url: https://api.deepseek.com/v1
    api_key: sk-your-deepseek-key   # ← 直接写
```

### 3. 写一个数学问题

编辑 `problem/problem.tex`：

```latex
\begin{problem}
Let $f: [0,1] \to \mathbb{R}$ be continuous on $[0,1]$ and differentiable
on $(0,1)$, satisfying $f(0) = f(1) = 0$ and $f(x) > 0$ for all $x \in (0,1)$.
Prove that there exists $c \in (0,1)$ such that
\[
  \frac{f'(c)}{f(c)} = \frac{1}{1-c}.
\]
\end{problem}
```

### 4. 运行

```bash
bash run.sh
```

输出在 `proof_output/`：

```
proof_output/
├── proof.md                  ← 最终证明
├── proof_effort_summary.md   ← 证明历程总结
├── TOKEN_USAGE.md            ← Token 消耗明细
├── related_info/             ← 文献调研结果
└── verification/             ← 逐轮验证记录
    └── round_1/
        ├── proof_before_round.md
        ├── proof_status.md
        └── ...
```

也可以指定输入和输出：

```bash
bash run.sh my_problem.tex my_output/
```

---

## 模型选择

`config.yaml` 里每个 agent 角色可以独立指定模型：

```yaml
pipeline:
  simple_mode:
    proof_search:           # 证明生成 — 用最强模型
      adapter: deepseek
      model: deepseek-v4-pro
      thinking: true
      reasoning_effort: max

    structural_verifier:    # 结构验证 — 用性价比模型
      adapter: deepseek
      model: deepseek-v4-flash
      thinking: false
```

| 角色 | 推荐模型 | 理由 |
|------|----------|------|
| proof_search | `deepseek-v4-pro` + `reasoning_effort: max` | 证明需要深度推理 |
| structural_verifier | `deepseek-v4-flash` | 结构检查不需要太强 |
| detailed_verifier | `deepseek-v4-flash` | 逐步验证量大，性价比优先 |
| verdict | `deepseek-v4-flash` | 裁决只需读报告做判断 |
| literature_survey | `deepseek-v4-pro` + `reasoning_effort: high` | 调研需要一定推理但不用 max |

### 开启多模型并行（需要 OpenCode Go 或其他辅助模型）

```yaml
pipeline:
  simple_mode:
    multi_model:
      enabled: true
      proof_search_providers:
        - adapter: deepseek
          model: deepseek-v4-pro
          thinking: true
          reasoning_effort: max
        - adapter: opencode_go
          model: claude-sonnet-4-6     # ← OpenCode 套餐里的 Claude
      verification_providers:
        - adapter: deepseek
          model: deepseek-v4-flash
        - adapter: opencode_go
          model: gpt-5.1               # ← 第二个视角验证
```

### 开启 Brainstorm

```yaml
pipeline:
  simple_mode:
    brainstorm:
      enabled: true
      providers:
        - adapter: deepseek
          model: deepseek-v4-flash
```

---

## 添加新模型

所有支持 OpenAI 兼容接口的模型（`/v1/chat/completions`）都可以接入。只需要在 `config.yaml` 的 `adapters` 节加一项：

```yaml
adapters:
  glm:                          # 智谱 GLM
    type: openai_compatible
    base_url: https://open.bigmodel.cn/api/paas/v4
    api_key: ${GLM_API_KEY}

  minimax:                      # MiniMax
    type: openai_compatible
    base_url: https://api.minimax.chat/v1
    api_key: ${MINIMAX_API_KEY}
```

然后在 `pipeline` 里引用即可。

---

## 运行测试

```bash
# 单元测试（不需要 API key）
python -m pytest tests/ -v --ignore=tests/test_integration.py

# 端到端测试（需要 API key）
export DEEPSEEK_API_KEY=sk-your-key
bash test_e2e.sh
```

---

## 项目结构

```
miniQED/
├── run.sh                     # 入口
├── config.example.yaml        # 配置模板（复制为 config.yaml）
├── problem/problem.tex        # 你的问题写在这里
├── prompts/                   # 9 个 prompt 模板
├── skill/super_math_skill.md  # 证明方法论 system prompt
├── mini_qed/
│   ├── orchestrator.py        # 顶层编排 Stage 0→1→2
│   ├── config.py              # 配置加载与校验
│   ├── adapters/              # LLM 适配层
│   │   ├── base.py            #   抽象接口
│   │   ├── openai_compatible.py  # DeepSeek/GLM/MiniMax...
│   │   └── registry.py        #   工厂函数
│   ├── steps/                 # 各步骤模块
│   │   ├── proof_search.py    #   证明搜索
│   │   ├── verification.py    #   验证（结构/详细/easy）
│   │   ├── verdict.py         #   裁决 DONE/CONTINUE
│   │   ├── brainstorm.py      #   多模型头脑风暴
│   │   └── selection.py       #   多模型证明选择
│   └── stages/                # 阶段编排
│       ├── stage0_survey.py   #   文献调研
│       ├── stage1_simple.py   #   Simple 模式证明循环
│       └── stage2_summary.py  #   证明总结
├── frozen/                    # QED Decomposition 模式（二期激活）
└── tests/                     # 59 个测试
```

---

## 二期规划

- 工作流引擎（LangGraph / Prefect）替代手写循环
- Lean 形式化验证（Aristotle CLI）
- GLM / Mimo / MiniMax 正式适配
- Decomposition 模式激活
- 中断恢复

---

## 许可

基于 QED (MIT License)。Copyright (c) 2026 proofQED。
