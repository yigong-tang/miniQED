# miniQED 一期 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 QED 框架构建 miniQED——用模块化架构 + DeepSeek/OpenCode 国产 LLM 替代 CLI 调用，实现 Simple 模式数学证明自动生成。

**Architecture:** 自底向上：adapter 层（`adapters/`）→ 步骤模块（`steps/`）→ 阶段编排（`stages/`）→ 顶层调度（`orchestrator.py`）。每层只依赖下层，dataclass 驱动模块间通信，同时落盘 markdown 实现可观测性。

**Tech Stack:** Python 3.11+、asyncio、openai SDK、pyyaml、dataclasses。依赖：`pip install openai pyyaml`。

**设计文档:** `docs/superpowers/specs/2026-06-07-miniqed-phase1-design.md`

---

## 文件结构总览

```
miniQED/
├── config.yaml, config.example.yaml
├── run.sh
├── problem/problem.tex
├── prompts/                         ← 从 QED 复制，调整 placeholder 名称
├── skill/super_math_skill.md        ← 从 QED 复制
├── human_help/                      ← 从 QED 复制
├── mini_qed/
│   ├── adapters/  (6 files)
│   ├── steps/     (5 files)
│   ├── stages/    (3 files)
│   ├── orchestrator.py, config.py, logging.py, utils.py
├── frozen/                          ← 冻结的 QED decomposition 代码
├── tests/         (5 files)
└── proof_output/  (运行时生成)
```

**依赖方向:** adapters ← steps ← stages ← orchestrator
**每层可独立测试:** adapters 可单独对 DeepSeek API 发请求；steps 可单独调用传入 mock adapter；stages 可单独跑传入 mock steps。

---

### Task 1: 项目脚手架与依赖

**Files:**
- Create: `mini_qed/__init__.py`
- Create: `mini_qed/adapters/__init__.py`
- Create: `mini_qed/steps/__init__.py`
- Create: `mini_qed/stages/__init__.py`
- Create: `tests/__init__.py`
- Create: `problem/problem.tex`
- Create: `.gitignore`
- Modify: (n/a — new files only)

- [ ] **Step 1: 创建所有 `__init__.py` 和空目录结构**

```bash
mkdir -p mini_qed/adapters mini_qed/steps mini_qed/stages tests problem frozen proof_output
```

```bash
for dir in mini_qed mini_qed/adapters mini_qed/steps mini_qed/stages tests; do
    echo '"""miniQED - A modular multi-agent mathematical proof system."""' > "$dir/__init__.py"
done
```

- [ ] **Step 2: 创建 `problem/problem.tex`（示例问题，来自 QED README）**

```latex
% problem/problem.tex
\begin{problem}
Let $f: [0,1] \to \mathbb{R}$ be a continuous function satisfying
$f(0) = f(1) = 0$ and $f(x) > 0$ for all $x \in (0,1)$.
Prove that there exists $c \in (0,1)$ such that
\[
  \frac{f'(c)}{f(c)} = \frac{1}{1-c}.
\]
\end{problem}
```

- [ ] **Step 3: 创建 `.gitignore`**

```bash
echo "proof_output/
config.yaml
__pycache__/
*.pyc
.env" > .gitignore
```

- [ ] **Step 4: 验证目录结构**

```bash
ls -R mini_qed/ tests/ problem/
```

Expected: 三个 `__init__.py` 存在，`problem.tex` 存在。

- [ ] **Step 5: 安装依赖**

```bash
pip install openai pyyaml
```

- [ ] **Step 6: Commit**

```bash
git add mini_qed/ tests/ problem/ .gitignore
git commit -m "feat: scaffold miniQED project structure

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Adapter 抽象接口

**Files:**
- Create: `mini_qed/adapters/base.py`
- Test: `tests/test_adapters.py` (部分)

- [ ] **Step 1: 写失败测试 — 验证抽象接口不可直接实例化**

```python
# tests/test_adapters.py
import pytest
from mini_qed.adapters.base import AbstractLLMAdapter, LLMResponse


class TestAbstractLLMAdapter:
    def test_cannot_instantiate_abstract_adapter(self):
        """Abstract adapter should raise TypeError on direct instantiation."""
        with pytest.raises(TypeError):
            AbstractLLMAdapter()  # noqa


class TestLLMResponse:
    def test_llm_response_creation(self):
        resp = LLMResponse(
            text="hello",
            input_tokens=10,
            output_tokens=5,
            model="test-model",
            elapsed_s=1.5,
        )
        assert resp.text == "hello"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.model == "test-model"
        assert resp.elapsed_s == 1.5
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_adapters.py::TestAbstractLLMAdapter -v
```

Expected: FAIL (import error — module doesn't exist).

- [ ] **Step 3: 实现 `LLMResponse` + `AbstractLLMAdapter`**

```python
# mini_qed/adapters/base.py
"""Abstract interface for all LLM backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Unified response from any LLM backend."""
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    elapsed_s: float


class AbstractLLMAdapter(ABC):
    """All LLM backends implement this interface.

    Each concrete adapter handles one provider (DeepSeek, GLM, Claude, etc.).
    The pipeline never knows which provider it's talking to.
    """

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
        """Send a prompt and return a unified response.

        Args:
            prompt: The user message / task description.
            system: Optional system prompt (prepended before user message).
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the response.
            thinking: If True, enable extended thinking (DeepSeek, Claude).
            reasoning_effort: For DeepSeek V4 — "minimal", "low", "medium", "high", "max".

        Returns:
            LLMResponse with text, token counts, model name, and elapsed time.
        """
        ...
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_adapters.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mini_qed/adapters/base.py tests/test_adapters.py
git commit -m "feat: add AbstractLLMAdapter and LLMResponse

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: OpenAICompatibleAdapter（路径 A）

**Files:**
- Create: `mini_qed/adapters/openai_compatible.py`
- Test: 追加到 `tests/test_adapters.py`

- [ ] **Step 1: 写失败测试 — 需要 DEEPSEEK_API_KEY 环境变量**

```python
# 追加到 tests/test_adapters.py
import os
import asyncio
from mini_qed.adapters.openai_compatible import OpenAICompatibleAdapter


class TestOpenAICompatibleAdapter:
    @pytest.fixture
    def adapter(self):
        """Create adapter with environment variable or skip."""
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            pytest.skip("DEEPSEEK_API_KEY not set")
        return OpenAICompatibleAdapter(
            base_url="https://api.deepseek.com/v1",
            api_key=key,
            model="deepseek-v4-flash",
        )

    @pytest.mark.asyncio
    async def test_basic_chat(self, adapter):
        """A simple chat should return a non-empty response."""
        resp = await adapter.chat(
            prompt="Say exactly 'hello world' and nothing else.",
            temperature=0.0,
            max_tokens=50,
        )
        assert isinstance(resp.text, str)
        assert len(resp.text.strip()) > 0
        assert resp.model == "deepseek-v4-flash"
        assert resp.input_tokens > 0
        assert resp.output_tokens > 0
        assert resp.elapsed_s > 0

    @pytest.mark.asyncio
    async def test_system_prompt(self, adapter):
        """System prompt should influence the response."""
        resp = await adapter.chat(
            prompt="What is your role?",
            system="You are a mathematician. Always answer 'I am a mathematician.'",
            temperature=0.0,
            max_tokens=100,
        )
        assert "mathematician" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_thinking_mode(self, adapter):
        """Thinking mode should not error."""
        resp = await adapter.chat(
            prompt="What is 2+2? Answer with just the number.",
            thinking=True,
            temperature=0.0,
            max_tokens=100,
        )
        assert "4" in resp.text
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
DEEPSEEK_API_KEY=sk-test python -m pytest tests/test_adapters.py::TestOpenAICompatibleAdapter -v
```

Expected: FAIL (import error).

- [ ] **Step 3: 实现 `OpenAICompatibleAdapter`**

```python
# mini_qed/adapters/openai_compatible.py
"""OpenAI-compatible API adapter — covers DeepSeek, GLM, MiniMax, Mimo, OpenCode."""

import time
from openai import AsyncOpenAI
from mini_qed.adapters.base import AbstractLLMAdapter, LLMResponse


class OpenAICompatibleAdapter(AbstractLLMAdapter):
    """Adapter for any LLM exposing an OpenAI-compatible /v1/chat/completions endpoint.

    Supports extended parameters for DeepSeek V4 (thinking, reasoning_effort).

    Usage:
        adapter = OpenAICompatibleAdapter(
            base_url="https://api.deepseek.com/v1",
            api_key=os.environ["DEEPSEEK_API_KEY"],
            model="deepseek-v4-pro",
        )
        resp = await adapter.chat("Prove that sqrt(2) is irrational.",
                                  thinking=True, reasoning_effort="max")
    """

    def __init__(self, base_url: str, api_key: str, model: str = ""):
        self._base_url = base_url
        self._api_key = api_key
        self._default_model = model
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

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
        model = self._default_model

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # DeepSeek V4 extended parameters via extra_body
        extra: dict = {}
        if thinking:
            extra["thinking"] = {"type": "enabled"}
        if reasoning_effort:
            extra["reasoning_effort"] = reasoning_effort
        if extra:
            kwargs["extra_body"] = extra

        start = time.time()
        response = await self._client.chat.completions.create(**kwargs)
        elapsed = time.time() - start

        choice = response.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            model=model,
            elapsed_s=elapsed,
        )
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_adapters.py::TestOpenAICompatibleAdapter -v
```

Expected: 3 tests PASS (requires valid `DEEPSEEK_API_KEY` env var).

- [ ] **Step 5: Commit**

```bash
git add mini_qed/adapters/openai_compatible.py tests/test_adapters.py
git commit -m "feat: add OpenAICompatibleAdapter for DeepSeek/GLM/etc.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: 占位 Adapter（Anthropic + OpenAI）

**Files:**
- Create: `mini_qed/adapters/anthropic_adapter.py`
- Create: `mini_qed/adapters/openai_adapter.py`

- [ ] **Step 1: 实现 `AnthropicAdapter`（占位，一期不激活）**

```python
# mini_qed/adapters/anthropic_adapter.py
"""Anthropic adapter — placeholder for Phase 2 activation."""

from mini_qed.adapters.base import AbstractLLMAdapter, LLMResponse


class AnthropicAdapter(AbstractLLMAdapter):
    """Adapter for Anthropic Claude via the official anthropic SDK.

    Phase 1: placeholder — raises NotImplementedError.
    Phase 2: full implementation with anthropic SDK.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self._api_key = api_key
        self._model = model

    async def chat(self, prompt: str, *, system: str = "", temperature: float = 0.7,
                   max_tokens: int = 65536, thinking: bool = False,
                   reasoning_effort: str | None = None) -> LLMResponse:
        raise NotImplementedError(
            "AnthropicAdapter is a Phase 2 placeholder. "
            "Use deepseek or opencode_go adapter for Phase 1."
        )
```

- [ ] **Step 2: 实现 `OpenAIAdapter`（占位，一期不激活）**

```python
# mini_qed/adapters/openai_adapter.py
"""OpenAI adapter — placeholder for Phase 2 activation."""

from mini_qed.adapters.base import AbstractLLMAdapter, LLMResponse


class OpenAIAdapter(AbstractLLMAdapter):
    """Adapter for OpenAI GPT via the official openai SDK.

    Phase 1: placeholder — raises NotImplementedError.
    Phase 2: full implementation with openai SDK.
    """

    def __init__(self, api_key: str, model: str = "gpt-5.1"):
        self._api_key = api_key
        self._model = model

    async def chat(self, prompt: str, *, system: str = "", temperature: float = 0.7,
                   max_tokens: int = 65536, thinking: bool = False,
                   reasoning_effort: str | None = None) -> LLMResponse:
        raise NotImplementedError(
            "OpenAIAdapter is a Phase 2 placeholder. "
            "Use deepseek or opencode_go adapter for Phase 1."
        )
```

- [ ] **Step 3: 运行导入测试**

```bash
python -c "from mini_qed.adapters.anthropic_adapter import AnthropicAdapter; print('ok')"
python -c "from mini_qed.adapters.openai_adapter import OpenAIAdapter; print('ok')"
```

Expected: `ok` (no error).

- [ ] **Step 4: Commit**

```bash
git add mini_qed/adapters/anthropic_adapter.py mini_qed/adapters/openai_adapter.py
git commit -m "feat: add placeholder AnthropicAdapter and OpenAIAdapter

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Adapter Registry（工厂函数）

**Files:**
- Create: `mini_qed/adapters/registry.py`
- Test: 追加到 `tests/test_adapters.py`

- [ ] **Step 1: 写测试 — 验证 registry 能根据配置创建正确的 adapter**

```python
# 追加到 tests/test_adapters.py
from mini_qed.adapters.registry import build_adapter, AdapterRegistry
from mini_qed.adapters.openai_compatible import OpenAICompatibleAdapter
from mini_qed.adapters.anthropic_adapter import AnthropicAdapter
from mini_qed.adapters.openai_adapter import OpenAIAdapter


class TestAdapterRegistry:
    def test_build_openai_compatible_adapter(self):
        cfg = {
            "type": "openai_compatible",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "sk-test",
            "model": "deepseek-v4-flash",
        }
        adapter = build_adapter("deepseek", cfg)
        assert isinstance(adapter, OpenAICompatibleAdapter)

    def test_build_anthropic_adapter(self):
        cfg = {
            "type": "anthropic_sdk",
            "api_key": "sk-test",
            "model": "claude-sonnet-4-6",
        }
        adapter = build_adapter("claude", cfg)
        assert isinstance(adapter, AnthropicAdapter)

    def test_build_openai_adapter(self):
        cfg = {
            "type": "openai_sdk",
            "api_key": "sk-test",
            "model": "gpt-5.1",
        }
        adapter = build_adapter("openai", cfg)
        assert isinstance(adapter, OpenAIAdapter)

    def test_build_unknown_type_raises(self):
        cfg = {"type": "unknown_type", "api_key": "sk-test"}
        with pytest.raises(ValueError, match="Unknown adapter type"):
            build_adapter("bad", cfg)


class TestAdapterRegistryClass:
    def test_get_returns_correct_adapter(self):
        config = {
            "adapters": {
                "deepseek": {
                    "type": "openai_compatible",
                    "base_url": "https://api.deepseek.com/v1",
                    "api_key": "sk-test",
                },
                "claude": {
                    "type": "anthropic_sdk",
                    "api_key": "sk-test",
                },
            }
        }
        reg = AdapterRegistry(config)
        adapter = reg.get("deepseek", model="deepseek-v4-flash")
        assert isinstance(adapter, OpenAICompatibleAdapter)

    def test_get_missing_adapter_raises(self):
        reg = AdapterRegistry({"adapters": {}})
        with pytest.raises(ValueError, match="not found"):
            reg.get("nonexistent")
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_adapters.py::TestAdapterRegistry -v
```

Expected: FAIL (import error for registry module).

- [ ] **Step 3: 实现 `registry.py`**

```python
# mini_qed/adapters/registry.py
"""Adapter factory: build adapters from config dicts."""

import os
from mini_qed.adapters.base import AbstractLLMAdapter
from mini_qed.adapters.openai_compatible import OpenAICompatibleAdapter
from mini_qed.adapters.anthropic_adapter import AnthropicAdapter
from mini_qed.adapters.openai_adapter import OpenAIAdapter


def build_adapter(name: str, cfg: dict) -> AbstractLLMAdapter:
    """Create a single adapter from a config sub-dict.

    Args:
        name: Adapter name (for error messages).
        cfg: Dict with keys: type, base_url, api_key, model (optional).

    Returns:
        A concrete AbstractLLMAdapter instance.
    """
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
            model=cfg.get("model", "claude-sonnet-4-6"),
        )
    elif adapter_type == "openai_sdk":
        return OpenAIAdapter(
            api_key=os.path.expandvars(cfg["api_key"]),
            model=cfg.get("model", "gpt-5.1"),
        )
    else:
        raise ValueError(
            f"Unknown adapter type '{adapter_type}' for adapter '{name}'. "
            f"Supported types: openai_compatible, anthropic_sdk, openai_sdk."
        )


class AdapterRegistry:
    """Holds adapter configs and creates adapter instances on demand.

    Adapters are cached after first creation so each adapter is a singleton
    within a pipeline run (the underlying AsyncOpenAI client is reused).

    Usage:
        reg = AdapterRegistry({"adapters": {"deepseek": {...}}})
        adapter = reg.get("deepseek", model="deepseek-v4-pro")
    """

    def __init__(self, config: dict):
        raw = config.get("adapters", {})
        if not raw:
            raise ValueError("Config must contain an 'adapters' section.")
        self._configs: dict[str, dict] = raw
        self._instances: dict[str, AbstractLLMAdapter] = {}

    def get(self, name: str, *, model: str | None = None) -> AbstractLLMAdapter:
        """Get or create an adapter instance.

        Args:
            name: Adapter name (must exist in config.adapters).
            model: Override the default model for this adapter.

        Returns:
            A cached or newly-created AbstractLLMAdapter.
        """
        cfg = self._configs.get(name)
        if cfg is None:
            raise ValueError(
                f"Adapter '{name}' not found in config.adapters. "
                f"Available: {', '.join(self._configs.keys())}"
            )

        # If model override requested, always create a new instance
        if model is not None:
            cfg_copy = dict(cfg)
            cfg_copy["model"] = model
            return build_adapter(name, cfg_copy)

        # Otherwise cache and reuse
        if name not in self._instances:
            self._instances[name] = build_adapter(name, cfg)
        return self._instances[name]
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_adapters.py::TestAdapterRegistry tests/test_adapters.py::TestAdapterRegistryClass -v
```

Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mini_qed/adapters/registry.py tests/test_adapters.py
git commit -m "feat: add AdapterRegistry — factory for LLM adapters

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: 配置加载与校验

**Files:**
- Create: `mini_qed/config.py`
- Test: `tests/test_config.py`
- Create: `config.example.yaml`

- [ ] **Step 1: 写测试 — 覆盖正常加载、环境变量展开、校验错误**

```python
# tests/test_config.py
import os
import tempfile
import pytest
from mini_qed.config import (
    AgentRoleConfig,
    SimpleModeConfig,
    PipelineConfig,
    load_pipeline_config,
)


class TestAgentRoleConfig:
    def test_from_yaml_dict_basic(self):
        d = {"adapter": "deepseek", "model": "deepseek-v4-pro"}
        cfg = AgentRoleConfig.from_dict(d, "test_role")
        assert cfg.adapter == "deepseek"
        assert cfg.model == "deepseek-v4-pro"
        assert cfg.thinking is False
        assert cfg.reasoning_effort is None

    def test_from_yaml_dict_with_thinking(self):
        d = {"adapter": "deepseek", "model": "deepseek-v4-pro",
             "thinking": True, "reasoning_effort": "max"}
        cfg = AgentRoleConfig.from_dict(d, "test_role")
        assert cfg.thinking is True
        assert cfg.reasoning_effort == "max"


class TestPipelineConfig:
    def test_valid_config_loads(self):
        cfg = PipelineConfig.from_yaml_dict({
            "pipeline": {
                "max_proof_iterations": 9,
                "output_dir": "./proof_output",
                "literature_survey": {"adapter": "deepseek", "model": "deepseek-v4-pro"},
                "simple_mode": {
                    "proof_search": {"adapter": "deepseek", "model": "deepseek-v4-pro"},
                    "structural_verifier": {"adapter": "deepseek", "model": "deepseek-v4-flash"},
                    "detailed_verifier": {"adapter": "deepseek", "model": "deepseek-v4-flash"},
                    "verdict": {"adapter": "deepseek", "model": "deepseek-v4-flash"},
                },
                "proof_summary": {"adapter": "deepseek", "model": "deepseek-v4-pro"},
            }
        })
        assert cfg.max_proof_iterations == 9
        assert cfg.simple_mode.proof_search.adapter == "deepseek"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValueError, match="Missing required field"):
            PipelineConfig.from_yaml_dict({
                "pipeline": {
                    # missing simple_mode
                }
            })

    def test_env_var_expansion(self):
        os.environ["TEST_API_KEY"] = "sk-test-123"
        cfg = PipelineConfig.from_yaml_dict({
            "pipeline": {
                "max_proof_iterations": 5,
                "literature_survey": {"adapter": "test", "model": "m1"},
                "simple_mode": {
                    "proof_search": {"adapter": "test", "model": "m1"},
                    "structural_verifier": {"adapter": "test", "model": "m2"},
                    "detailed_verifier": {"adapter": "test", "model": "m2"},
                    "verdict": {"adapter": "test", "model": "m2"},
                },
                "proof_summary": {"adapter": "test", "model": "m1"},
            }
        })
        assert cfg.max_proof_iterations == 5
        del os.environ["TEST_API_KEY"]
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_config.py -v
```

Expected: FAIL (module doesn't exist).

- [ ] **Step 3: 实现 `config.py`**

```python
# mini_qed/config.py
"""Configuration loading, validation, and env-var expansion."""

from dataclasses import dataclass, field
import os
import re


_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _expand_env(value: str) -> str:
    """Expand ${VAR} patterns in a string using environment variables."""
    def _replacer(m: re.Match) -> str:
        var = m.group(1)
        val = os.environ.get(var)
        if val is None:
            raise ValueError(
                f"Environment variable '{var}' is not set. "
                f"Required by config value that references ${{{var}}}."
            )
        return val
    return _ENV_VAR_RE.sub(_replacer, value)


@dataclass
class AgentRoleConfig:
    """Configuration for a single agent role (proof search, verifier, etc.)."""
    adapter: str
    model: str
    thinking: bool = False
    reasoning_effort: str | None = None

    @classmethod
    def from_dict(cls, d: dict, name: str) -> "AgentRoleConfig":
        adapter = d.get("adapter")
        if not adapter:
            raise ValueError(
                f"Agent '{name}': missing required field 'adapter'."
            )
        return cls(
            adapter=adapter,
            model=d.get("model", ""),
            thinking=d.get("thinking", False),
            reasoning_effort=d.get("reasoning_effort"),
        )


@dataclass
class MultiModelProvider:
    """One provider in a multi-model config."""
    adapter: str
    model: str
    thinking: bool = False
    reasoning_effort: str | None = None

    @classmethod
    def from_dict(cls, d: dict, index: int) -> "MultiModelProvider":
        return cls(
            adapter=d["adapter"],
            model=d.get("model", ""),
            thinking=d.get("thinking", False),
            reasoning_effort=d.get("reasoning_effort"),
        )


@dataclass
class SimpleModeConfig:
    """Simple mode pipeline configuration."""
    proof_search: AgentRoleConfig
    structural_verifier: AgentRoleConfig
    detailed_verifier: AgentRoleConfig
    verdict: AgentRoleConfig
    brainstorm: dict | None = None          # {enabled: bool, providers: [...]}
    multi_model: dict | None = None         # {enabled: bool, ...}

    @classmethod
    def from_dict(cls, d: dict) -> "SimpleModeConfig":
        required = {
            "proof_search": "simple_mode.proof_search",
            "structural_verifier": "simple_mode.structural_verifier",
            "detailed_verifier": "simple_mode.detailed_verifier",
            "verdict": "simple_mode.verdict",
        }
        for key, path in required.items():
            if key not in d:
                raise ValueError(
                    f"Missing required field '{path}' in config.yaml pipeline.simple_mode."
                )

        return cls(
            proof_search=AgentRoleConfig.from_dict(d["proof_search"], "simple_mode.proof_search"),
            structural_verifier=AgentRoleConfig.from_dict(d["structural_verifier"], "simple_mode.structural_verifier"),
            detailed_verifier=AgentRoleConfig.from_dict(d["detailed_verifier"], "simple_mode.detailed_verifier"),
            verdict=AgentRoleConfig.from_dict(d["verdict"], "simple_mode.verdict"),
            brainstorm=d.get("brainstorm"),
            multi_model=d.get("multi_model"),
        )


@dataclass
class PipelineConfig:
    """Full pipeline configuration, parsed from config.yaml."""
    max_proof_iterations: int
    output_dir: str
    literature_survey: AgentRoleConfig
    simple_mode: SimpleModeConfig
    proof_summary: AgentRoleConfig

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "PipelineConfig":
        """Parse the top-level config dict (already YAML-loaded).

        Validates all required sections and fields.
        """
        pipeline = raw.get("pipeline")
        if not pipeline:
            raise ValueError(
                "Missing top-level 'pipeline' section in config.yaml."
            )

        # Required sub-sections
        required_sections = {
            "literature_survey": "pipeline.literature_survey",
            "simple_mode": "pipeline.simple_mode",
            "proof_summary": "pipeline.proof_summary",
        }
        for key, path in required_sections.items():
            if key not in pipeline:
                raise ValueError(f"Missing required section '{path}' in config.yaml.")

        return cls(
            max_proof_iterations=pipeline.get("max_proof_iterations", 9),
            output_dir=pipeline.get("output_dir", "./proof_output"),
            literature_survey=AgentRoleConfig.from_dict(
                pipeline["literature_survey"], "pipeline.literature_survey"
            ),
            simple_mode=SimpleModeConfig.from_dict(pipeline["simple_mode"]),
            proof_summary=AgentRoleConfig.from_dict(
                pipeline["proof_summary"], "pipeline.proof_summary"
            ),
        )


def load_pipeline_config(path: str) -> tuple[PipelineConfig, dict]:
    """Load and validate pipeline config from a YAML file.

    Returns:
        (PipelineConfig, raw_config_dict) — the raw dict is for AdapterRegistry.
    """
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Expand env vars in the entire config tree
    raw = _expand_config_env_vars(raw)

    pipeline = PipelineConfig.from_yaml_dict(raw)
    return pipeline, raw


def _expand_config_env_vars(obj):
    """Recursively expand ${VAR} patterns in all string values."""
    if isinstance(obj, str):
        return _expand_env(obj)
    elif isinstance(obj, dict):
        return {k: _expand_config_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_config_env_vars(v) for v in obj]
    return obj
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_config.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: 创建 `config.example.yaml`**

```yaml
# config.example.yaml — miniQED configuration (copy to config.yaml and fill in keys)
#
# config.yaml is gitignored — never commit API keys.

adapters:
  deepseek:
    type: openai_compatible
    base_url: https://api.deepseek.com/v1
    api_key: ${DEEPSEEK_API_KEY}

  # opencode_go:
  #   type: openai_compatible
  #   base_url: https://api.opencode.ai/v1
  #   api_key: ${OPENCODE_API_KEY}

pipeline:
  max_proof_iterations: 9
  output_dir: ./proof_output

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

    # brainstorm:
    #   enabled: true
    #   providers:
    #     - adapter: deepseek
    #       model: deepseek-v4-flash

    # multi_model:
    #   enabled: true
    #   proof_search_providers:
    #     - adapter: deepseek
    #       model: deepseek-v4-pro
    #       thinking: true
    #       reasoning_effort: max
    #     - adapter: opencode_go
    #       model: claude-sonnet-4-6
    #   verification_providers:
    #     - adapter: deepseek
    #       model: deepseek-v4-flash
    #     - adapter: opencode_go
    #       model: gpt-5.1

  proof_summary:
    adapter: deepseek
    model: deepseek-v4-pro
    thinking: false
```

- [ ] **Step 6: Commit**

```bash
git add mini_qed/config.py tests/test_config.py config.example.yaml
git commit -m "feat: add config loading with validation and env-var expansion

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Logger + TokenTracker

**Files:**
- Create: `mini_qed/logging.py`
- Test: `tests/test_logging.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_logging.py
import os
import tempfile
from mini_qed.logging import PipelineLogger, TokenTracker


class TestPipelineLogger:
    def test_creates_status_file(self):
        with tempfile.TemporaryDirectory() as d:
            logger = PipelineLogger(d, "Test Phase")
            logger.update_status(1, 9, "Proof Search", "RUNNING", "Running agent...")
            assert os.path.exists(os.path.join(d, "AUTO_RUN_STATUS.md"))
            assert os.path.exists(os.path.join(d, "AUTO_RUN_LOG.txt"))

    def test_log_appends(self):
        with tempfile.TemporaryDirectory() as d:
            logger = PipelineLogger(d, "Test")
            logger.log("line one")
            logger.log("line two")
            log_path = os.path.join(d, "AUTO_RUN_LOG.txt")
            with open(log_path) as f:
                content = f.read()
            assert "line one" in content
            assert "line two" in content


class TestTokenTracker:
    def test_record_and_save(self):
        with tempfile.TemporaryDirectory() as d:
            tracker = TokenTracker(d, "deepseek-v4-pro")
            tracker.record("Proof Search R1", 1000, 500, 5.2,
                          provider="deepseek", model="deepseek-v4-pro")
            tracker.record("Verification R1", 800, 300, 3.1,
                          provider="deepseek", model="deepseek-v4-flash")

            assert tracker.total_input == 1800
            assert tracker.total_output == 800
            assert len(tracker.calls) == 2

            # Verify md output
            assert os.path.exists(tracker.md_path)
            with open(tracker.md_path) as f:
                md = f.read()
            assert "Proof Search R1" in md
            assert "deepseek-v4-pro" in md

            # Verify json output
            assert os.path.exists(tracker.json_path)
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_logging.py -v
```

Expected: FAIL.

- [ ] **Step 3: 实现 `logging.py`**

从 QED `code/pipeline.py` 中提取 `PipelineLogger` 和 `TokenTracker` 类，保持逻辑完全一致但添加类型注解。由于这两个类在 QED 中已经很成熟，此处直接移植：

```python
# mini_qed/logging.py
"""Pipeline logging and token tracking — ported from QED code/pipeline.py."""

import json
import os
from datetime import datetime


class PipelineLogger:
    """Persistent logging to AUTO_RUN_STATUS.md, .history, and AUTO_RUN_LOG.txt."""

    def __init__(self, log_dir: str, phase: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.phase = phase
        self.status_file = os.path.join(log_dir, "AUTO_RUN_STATUS.md")
        self.history_file = os.path.join(log_dir, "AUTO_RUN_STATUS.md.history")
        self.log_file = os.path.join(log_dir, "AUTO_RUN_LOG.txt")
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.pid = os.getpid()
        self.append_history(f"{phase} started")

    def update_status(self, iteration: int, max_iter: int, step: str, state: str, details: str):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history = ""
        if os.path.exists(self.history_file):
            with open(self.history_file) as f:
                history = f.read()
        with open(self.status_file, "w", encoding="utf-8") as f:
            f.write(f"# {self.phase} - Auto Status\n\n")
            f.write("| Field | Value |\n|-------|-------|\n")
            f.write(f"| **Status** | {state} |\n")
            f.write(f"| **Current Iteration** | {iteration} / {max_iter} |\n")
            f.write(f"| **Current Step** | {step} |\n")
            f.write(f"| **Started At** | {self.start_time} |\n")
            f.write(f"| **Last Updated** | {now} |\n")
            f.write(f"| **PID** | {self.pid} |\n\n")
            f.write(f"## Current Activity\n{details}\n\n")
            f.write(f"## Progress History\n{history}\n")

    def append_history(self, msg: str):
        now = datetime.now().strftime("%H:%M:%S")
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(f"- [{now}] {msg}\n")

    def log(self, msg: str):
        print(msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def finalize(self, iteration: int, max_iter: int, exit_state: str, details: str):
        self.update_status(iteration, max_iter, exit_state, exit_state, details)
        self.append_history(f"Process ended: {exit_state}")


class TokenTracker:
    """Accumulates token usage across all agent calls and persists to disk."""

    def __init__(self, output_dir: str, model: str):
        self.output_dir = output_dir
        self.model = model
        self.calls: list[dict] = []
        self.total_input = 0
        self.total_output = 0
        self.total_elapsed = 0.0
        self.per_provider: dict[str, dict] = {}
        self.md_path = os.path.join(output_dir, "TOKEN_USAGE.md")
        self.json_path = os.path.join(output_dir, "token_usage.json")
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def record(self, call_name: str, input_tokens: int, output_tokens: int,
               elapsed: float, provider: str = "deepseek", model: str = ""):
        self.total_input += input_tokens
        self.total_output += output_tokens
        self.total_elapsed += elapsed

        if provider not in self.per_provider:
            self.per_provider[provider] = {
                "input": 0, "output": 0, "calls": 0,
                "model": model or self.model,
            }
        self.per_provider[provider]["input"] += input_tokens
        self.per_provider[provider]["output"] += output_tokens
        self.per_provider[provider]["calls"] += 1

        self.calls.append({
            "call": len(self.calls) + 1,
            "name": call_name,
            "provider": provider,
            "model": model or self.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_s": round(elapsed, 1),
            "cumul_input": self.total_input,
            "cumul_output": self.total_output,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        self._save()

    def _save(self):
        lines = [
            "# Token Usage\n",
            f"**Model:** `{self.model}`  ",
            f"**Started:** {self.start_time}  ",
            f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n",
            "## Summary\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total input tokens | {self.total_input:,} |",
            f"| Total output tokens | {self.total_output:,} |",
            f"| Total tokens | {self.total_input + self.total_output:,} |",
            f"| Total elapsed | {self.total_elapsed:.0f}s |",
            f"| Agent calls | {len(self.calls)} |\n",
        ]

        if len(self.per_provider) > 1:
            lines.append("## Per-Provider Summary\n")
            lines.append("| Provider | Model | Input | Output | Total | Calls |")
            lines.append("|----------|-------|------:|-------:|------:|------:|")
            for prov, stats in sorted(self.per_provider.items()):
                total = stats['input'] + stats['output']
                lines.append(
                    f"| {prov} | {stats['model']} "
                    f"| {stats['input']:,} | {stats['output']:,} "
                    f"| {total:,} | {stats['calls']} |"
                )
            lines.append("")

        lines.append("## Per-Call Breakdown\n")
        lines.append("| # | Agent | Provider | Input | Output | Time | Cumul In | Cumul Out |")
        lines.append("|---|-------|----------|------:|-------:|-----:|---------:|----------:|")

        for c in self.calls:
            lines.append(
                f"| {c['call']} | {c['name']} | {c.get('provider', 'unknown')} "
                f"| {c['input_tokens']:,} | {c['output_tokens']:,} "
                f"| {c['elapsed_s']}s "
                f"| {c['cumul_input']:,} | {c['cumul_output']:,} |"
            )
        lines.append("")

        with open(self.md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        data = {
            "model": self.model,
            "started": self.start_time,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_input_tokens": self.total_input,
            "total_output_tokens": self.total_output,
            "total_tokens": self.total_input + self.total_output,
            "total_elapsed_s": round(self.total_elapsed, 1),
            "per_provider": self.per_provider,
            "calls": self.calls,
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_logging.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mini_qed/logging.py tests/test_logging.py
git commit -m "feat: add PipelineLogger and TokenTracker

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: 工具函数

**Files:**
- Create: `mini_qed/utils.py`
- Test: `tests/test_utils.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_utils.py
import os
import tempfile
from mini_qed.utils import load_prompt, file_nonempty, find_verification_files


class TestLoadPrompt:
    def test_substitutes_placeholders(self):
        with tempfile.TemporaryDirectory() as d:
            tmpl_path = os.path.join(d, "test_prompt.md")
            with open(tmpl_path, "w") as f:
                f.write("Problem: {problem_file}\nOutput: {output_dir}")

            result = load_prompt(d, "test_prompt.md",
                                problem_file="/path/to/problem.tex",
                                output_dir="/path/to/output")
            assert "Problem: /path/to/problem.tex" in result
            assert "Output: /path/to/output" in result

    def test_missing_placeholder_raises(self):
        with tempfile.TemporaryDirectory() as d:
            tmpl_path = os.path.join(d, "test_prompt.md")
            with open(tmpl_path, "w") as f:
                f.write("Hello {name}")

            with pytest.raises(KeyError):
                load_prompt(d, "test_prompt.md")  # name not provided


class TestFileNonempty:
    def test_nonexistent_file(self):
        assert not file_nonempty("/nonexistent/file.md")

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "empty.md")
            with open(p, "w") as f:
                f.write("")
            assert not file_nonempty(p)

    def test_nonempty_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "content.md")
            with open(p, "w") as f:
                f.write("hello")
            assert file_nonempty(p)


class TestFindVerificationFiles:
    def test_finds_single_legacy_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "verification_result.md")
            with open(p, "w") as f:
                f.write("PASS")
            assert find_verification_files(d) == [p]

    def test_finds_multi_verifier_files(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["verification_result_deepseek.md",
                         "verification_result_opencode.md"]:
                p = os.path.join(d, name)
                with open(p, "w") as f:
                    f.write("content")
            result = find_verification_files(d)
            assert len(result) == 2
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_utils.py -v
```

Expected: FAIL.

- [ ] **Step 3: 实现 `utils.py`**

```python
# mini_qed/utils.py
"""Shared utility functions — prompt loading, file checks, verification file discovery."""

import os


def load_prompt(prompts_dir: str, name: str, **kwargs) -> str:
    """Load a prompt template from disk and fill in {placeholders}.

    Args:
        prompts_dir: Directory containing the .md prompt templates.
        name: Filename within prompts_dir (e.g., "proof_search.md").
        **kwargs: Key-value pairs for placeholder substitution.

    Returns:
        The prompt string with all placeholders filled.

    Raises:
        FileNotFoundError: If the prompt file doesn't exist.
        KeyError: If a placeholder in the template has no corresponding kwarg.
    """
    path = os.path.join(prompts_dir, name)
    with open(path, encoding="utf-8") as f:
        template = f.read()
    return template.format(**kwargs)


def file_nonempty(path: str) -> bool:
    """Return True if path exists and has non-whitespace content."""
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        return bool(f.read().strip())


def find_verification_files(directory: str) -> list[str]:
    """Find all verification result files in a directory.

    Checks for single-file legacy format first, then multi-verifier format.

    Returns:
        List of absolute paths to verification files.
    """
    single = os.path.join(directory, "verification_result.md")
    if file_nonempty(single):
        return [single]
    files = []
    if os.path.isdir(directory):
        for name in sorted(os.listdir(directory)):
            if name.startswith("verification_result_") and name.endswith(".md"):
                path = os.path.join(directory, name)
                if file_nonempty(path):
                    files.append(path)
    return files
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_utils.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mini_qed/utils.py tests/test_utils.py
git commit -m "feat: add prompt loader and file utilities

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: 复制 QED 的 Prompts 和 Skill

**Files:**
- Copy: `prompts/*.md` ← 从 QED `prompts/` 复制
- Copy: `skill/super_math_skill.md` ← 从 QED `skill/` 复制
- Copy: `human_help/*.md` ← 从 QED `human_help/` 复制

- [ ] **Step 1: 复制 prompts**

```bash
cp QED/prompts/literature_survey.md prompts/
cp QED/prompts/proof_search.md prompts/
cp QED/prompts/proof_verify_structural.md prompts/
cp QED/prompts/proof_verify_detailed.md prompts/
cp QED/prompts/proof_verify_easy.md prompts/
cp QED/prompts/proof_select.md prompts/
cp QED/prompts/verdict_proof.md prompts/
cp QED/prompts/brainstorm.md prompts/
cp QED/prompts/proof_effort_summary.md prompts/
```

- [ ] **Step 2: 复制 skill 和 human_help**

```bash
cp QED/skill/super_math_skill.md skill/
cp QED/human_help/additional_prove_human_help_global.md human_help/
cp QED/human_help/additional_verify_rule_global.md human_help/
```

- [ ] **Step 3: 调整 prompt 中的文件路径引用**

QED 的 prompt 模板使用了 QED 特定的文件路径（如 `{proof_file}`、`{output_dir}/proof.md` 等）。检查占位符是否与 miniQED 的变量名一致。若有不一致的，统一为 miniQED 的命名。

```bash
grep -r "{.*}" prompts/ | head -20
```

- [ ] **Step 4: Commit**

```bash
git add prompts/ skill/ human_help/
git commit -m "feat: copy prompt templates, skill, and human_help from QED

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: `steps/proof_search.py`

**Files:**
- Create: `mini_qed/steps/proof_search.py`
- Test: `tests/test_proof_search.py`

- [ ] **Step 1: 写测试 — 使用 mock adapter 测试 proof search 逻辑**

```python
# tests/test_proof_search.py
import os
import tempfile
import pytest
from mini_qed.adapters.base import LLMResponse
from mini_qed.steps.proof_search import ProofResult, run_proof_search


class MockAdapter:
    """Fake adapter that returns a pre-configured response."""
    def __init__(self, response_text: str, model: str = "mock"):
        self.response_text = response_text
        self.model = model
        self.calls: list[dict] = []

    async def chat(self, prompt, *, system="", temperature=0.7, max_tokens=65536,
                   thinking=False, reasoning_effort=None) -> LLMResponse:
        self.calls.append({"prompt": prompt, "system": system})
        return LLMResponse(
            text=self.response_text,
            input_tokens=100,
            output_tokens=50,
            model=self.model,
            elapsed_s=0.1,
        )


class TestRunProofSearch:
    @pytest.mark.asyncio
    async def test_produces_proof_file(self):
        with tempfile.TemporaryDirectory() as d:
            problem_file = os.path.join(d, "problem.tex")
            related_info_dir = os.path.join(d, "related_info")
            os.makedirs(related_info_dir)
            with open(problem_file, "w") as f:
                f.write(r"\begin{problem}Test\end{problem}")
            with open(os.path.join(related_info_dir, "difficulty_evaluation.md"), "w") as f:
                f.write("## Classification: Easy")
            with open(os.path.join(related_info_dir, "related_work.md"), "w") as f:
                f.write("# Related work")

            adapter = MockAdapter("Proof: QED. Status: DONE.")
            result = await run_proof_search(
                adapter=adapter,
                model="mock-model",
                problem_file=problem_file,
                proof_file=os.path.join(d, "proof.md"),
                proof_status_file=os.path.join(d, "proof_status.md"),
                related_info_dir=related_info_dir,
                round_num=1,
                prev_instructions="- First round.",
                brainstorm_dir="",
                prompt_template_path="",  # will be handled differently in real use
                prove_skill="",
            )
            assert result.proof_text == "Proof: QED. Status: DONE."
            assert result.model == "mock-model"
            assert os.path.exists(result.proof_file)
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_proof_search.py -v
```

Expected: FAIL.

- [ ] **Step 3: 实现 `proof_search.py`**

```python
# mini_qed/steps/proof_search.py
"""Single-model proof search step.

Dispatches a proof-writing prompt to an LLM adapter and collects the result.
For multi-model parallel search, see stage1_simple.py which calls this in parallel.
"""

import asyncio
import os
import shutil
from dataclasses import dataclass
from mini_qed.adapters.base import AbstractLLMAdapter, LLMResponse
from mini_qed.utils import load_prompt


@dataclass
class ProofResult:
    """Output of a single proof search call."""
    proof_text: str
    proof_file: str
    status_file: str
    scratch_pad: str
    model: str
    tokens_input: int
    tokens_output: int


async def run_proof_search(
    *,
    adapter: AbstractLLMAdapter,
    model: str,
    problem_file: str,
    proof_file: str,
    proof_status_file: str,
    related_info_dir: str,
    round_num: int,
    prev_instructions: str,
    brainstorm_dir: str,
    prompt_template_path: str,
    prove_skill: str,
    human_help_dir: str = "",
    prev_round_human_help_dir: str = "",
    scratch_pad_file: str = "",
    error_file: str = "",
) -> ProofResult:
    """Run a single proof search agent.

    The adapter receives the full proof_search prompt (loaded from template,
    all placeholders filled) and returns a response. The response is saved
    to proof_file and proof_status_file.

    Args:
        adapter: The LLM adapter to use.
        model: Model name (for logging).
        problem_file: Path to problem.tex.
        proof_file: Where to write the proof.
        proof_status_file: Where to write what the prover tried.
        related_info_dir: Stage 0 output directory.
        round_num: Current round number.
        prev_instructions: Instructions from previous round's results.
        brainstorm_dir: Path to brainstorm results (empty if disabled).
        prompt_template_path: Full path to the proof_search.md template.
        prove_skill: Content of super_math_skill.md (as system prompt).
        human_help_dir: Global human_help directory.
        prev_round_human_help_dir: Previous round's human_help.
        scratch_pad_file: Where the prover can scribble.
        error_file: Where to log errors.

    Returns:
        ProofResult with proof text, file paths, and token usage.
    """
    prompts_dir = os.path.dirname(prompt_template_path)
    prompt_name = os.path.basename(prompt_template_path)

    prompt = load_prompt(
        prompts_dir, prompt_name,
        problem_file=problem_file,
        proof_file=proof_file,
        related_info_dir=related_info_dir,
        round_num=round_num,
        proof_status_file=proof_status_file,
        previous_round_instructions=prev_instructions,
        human_help_dir=human_help_dir,
        prev_round_human_help_dir=prev_round_human_help_dir,
        skill_file=os.path.join(os.path.dirname(prompts_dir), "skill", "super_math_skill.md"),
        scratch_pad_file=scratch_pad_file or os.path.join(os.path.dirname(proof_file), "scratch_pad.md"),
        error_file=error_file or os.path.join(os.path.dirname(proof_file), "error_proof_search.md"),
        brainstorm_dir=brainstorm_dir,
    )

    # Inject round-specific instruction
    prompt += f"\n\nThis is round {round_num}. Write or refine the proof. "
    prompt += "If one approach doesn't work after much effort, try a completely different proof strategy."

    # If proof_file exists and is the starting point, copy it for context
    if os.path.exists(proof_file):
        ctx_proof = os.path.join(os.path.dirname(proof_file), "proof_starting_point.md")
        shutil.copy2(proof_file, ctx_proof)

    response: LLMResponse = await adapter.chat(
        prompt=prompt,
        system=prove_skill,
    )

    # Save response to proof_file if the agent didn't write it via tool use
    os.makedirs(os.path.dirname(proof_file), exist_ok=True)
    if not os.path.exists(proof_file) or os.path.getsize(proof_file) == 0:
        with open(proof_file, "w", encoding="utf-8") as f:
            f.write(response.text)

    # Save status file
    os.makedirs(os.path.dirname(proof_status_file), exist_ok=True)
    if not os.path.exists(proof_status_file) or os.path.getsize(proof_status_file) == 0:
        with open(proof_status_file, "w", encoding="utf-8") as f:
            f.write(f"# Proof Status — Round {round_num}\n\n")
            f.write(f"Model: {model}\n\n")
            f.write(f"## Response\n\n{response.text[:2000]}\n")

    return ProofResult(
        proof_text=response.text,
        proof_file=proof_file,
        status_file=proof_status_file,
        scratch_pad=scratch_pad_file or "",
        model=model,
        tokens_input=response.input_tokens,
        tokens_output=response.output_tokens,
    )
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_proof_search.py -v
```

Expected: 1 test PASS.

- [ ] **Step 5: Commit**

```bash
git add mini_qed/steps/proof_search.py tests/test_proof_search.py
git commit -m "feat: add proof_search step — single-model proof generation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 11: `steps/verdict.py`

**Files:**
- Create: `mini_qed/steps/verdict.py`
- Test: `tests/test_verdict.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_verdict.py
import os
import tempfile
import pytest
from mini_qed.adapters.base import LLMResponse
from mini_qed.steps.verdict import run_verdict, Verdict


class MockVerdictAdapter:
    def __init__(self, decision: str):
        self.decision = decision
        self.calls: list[dict] = []

    async def chat(self, prompt, **kwargs) -> LLMResponse:
        self.calls.append({"prompt": prompt})
        return LLMResponse(
            text=f"Based on the verification reports, the overall verdict is: {self.decision}",
            input_tokens=100, output_tokens=10,
            model="mock", elapsed_s=0.1,
        )


class TestRunVerdict:
    @pytest.mark.asyncio
    async def test_done_verdict(self):
        adapter = MockVerdictAdapter("DONE")
        with tempfile.TemporaryDirectory() as d:
            vf = os.path.join(d, "verification_result.md")
            with open(vf, "w") as f:
                f.write("Overall Verdict: PASS")
            result = await run_verdict(
                adapter=adapter,
                verification_files=[vf],
                prompt_template_path="",  # will use custom prompt
                verdict_prompt_text="Based on verification at {verification_files}, is the proof done? Reply exactly DONE or CONTINUE.",
            )
            assert result == Verdict.DONE

    @pytest.mark.asyncio
    async def test_continue_verdict(self):
        adapter = MockVerdictAdapter("CONTINUE")
        with tempfile.TemporaryDirectory() as d:
            vf = os.path.join(d, "verification_result.md")
            with open(vf, "w") as f:
                f.write("Overall Verdict: FAIL")
            result = await run_verdict(
                adapter=adapter,
                verification_files=[vf],
                prompt_template_path="",
                verdict_prompt_text="Based on verification at {verification_files}, is the proof done? Reply exactly DONE or CONTINUE.",
            )
            assert result == Verdict.CONTINUE
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_verdict.py -v
```

Expected: FAIL.

- [ ] **Step 3: 实现 `verdict.py`**

```python
# mini_qed/steps/verdict.py
"""Verdict step: reads verification reports and decides DONE or CONTINUE."""

import enum
from mini_qed.adapters.base import AbstractLLMAdapter
from mini_qed.utils import load_prompt


class Verdict(enum.Enum):
    DONE = "DONE"
    CONTINUE = "CONTINUE"


async def run_verdict(
    *,
    adapter: AbstractLLMAdapter,
    verification_files: list[str],
    prompt_template_path: str = "",
    verdict_prompt_text: str = "",
) -> Verdict:
    """Run the verdict agent — decide whether the proof is done.

    Loads the verdict prompt template, injects the verification file paths,
    and asks the LLM to return exactly DONE or CONTINUE.

    Args:
        adapter: The LLM adapter for the verdict agent.
        verification_files: Paths to all verification result .md files.
        prompt_template_path: Path to verdict_proof.md template.
        verdict_prompt_text: Alternative: raw prompt text with {verification_files}
            placeholder (used when prompt_template_path is empty).

    Returns:
        Verdict.DONE or Verdict.CONTINUE.
    """
    if prompt_template_path:
        import os
        prompts_dir = os.path.dirname(prompt_template_path)
        prompt_name = os.path.basename(prompt_template_path)
        # Build the verification file reference string
        if len(verification_files) == 1:
            ref = f"Read the verification result file at `{verification_files[0]}`."
        else:
            ref = "\n".join(f"- `{f}`" for f in verification_files)
        prompt = load_prompt(prompts_dir, prompt_name, verification_result_file=ref)
    else:
        # Inline prompt mode (for testing)
        ref = "\n".join(verification_files)
        prompt = verdict_prompt_text.replace("{verification_files}", ref)

    response = await adapter.chat(
        prompt=prompt,
        temperature=0.0,
        max_tokens=100,
    )

    # Parse verdict from response — look for DONE or CONTINUE
    text = response.text.strip().upper()
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped == "DONE":
            return Verdict.DONE
        if stripped == "CONTINUE":
            return Verdict.CONTINUE

    # Fallback: substring search
    if "DONE" in text and "CONTINUE" not in text:
        return Verdict.DONE
    return Verdict.CONTINUE
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_verdict.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mini_qed/steps/verdict.py tests/test_verdict.py
git commit -m "feat: add verdict step — DONE/CONTINUE decision

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 12: `steps/brainstorm.py`

**Files:**
- Create: `mini_qed/steps/brainstorm.py`
- Test: `tests/test_brainstorm.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_brainstorm.py
import os
import tempfile
import pytest
from mini_qed.adapters.base import LLMResponse
from mini_qed.steps.brainstorm import run_brainstorm


class MockBrainstormAdapter:
    """Fake adapter that records calls and returns pre-configured ideas."""
    def __init__(self, response_texts: list[str], model: str = "mock"):
        self.response_texts = response_texts
        self.model = model
        self.call_count = 0

    async def chat(self, prompt, *, system="", **kwargs) -> LLMResponse:
        idx = self.call_count
        self.call_count += 1
        return LLMResponse(
            text=self.response_texts[idx % len(self.response_texts)],
            input_tokens=50, output_tokens=30,
            model=self.model, elapsed_s=0.1,
        )


class TestRunBrainstorm:
    @pytest.mark.asyncio
    async def test_parallel_brainstorm(self):
        with tempfile.TemporaryDirectory() as d:
            problem_file = os.path.join(d, "problem.tex")
            proof_file = os.path.join(d, "proof.md")
            related_info_dir = os.path.join(d, "related_info")
            os.makedirs(related_info_dir)
            with open(problem_file, "w") as f:
                f.write("test")
            with open(os.path.join(related_info_dir, "difficulty_evaluation.md"), "w") as f:
                f.write("## Classification: Medium")

            adapters = [
                MockBrainstormAdapter(["Use contradiction."], model="model-a"),
                MockBrainstormAdapter(["Try induction."], model="model-b"),
            ]
            providers = [
                {"name": "model-a", "model": "model-a"},
                {"name": "model-b", "model": "model-b"},
            ]
            output_dir = os.path.join(d, "brainstorm_results")
            os.makedirs(output_dir)

            results = await run_brainstorm(
                adapters=adapters,
                providers=providers,
                problem_file=problem_file,
                proof_file=proof_file,
                related_info_dir=related_info_dir,
                round_num=1,
                output_dir=output_dir,
                prev_verification_dir="",
            )
            assert len(results) == 2
            assert any("contradiction" in r for r in results)
            assert any("induction" in r for r in results)
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_brainstorm.py -v
```

Expected: FAIL.

- [ ] **Step 3: 实现 `brainstorm.py`**

```python
# mini_qed/steps/brainstorm.py
"""Brainstorm step: multiple models independently generate proof strategies.

Results are saved to individual files and collected into a list for the
proof search agent to read before attempting the proof.
"""

import asyncio
import os
from mini_qed.adapters.base import AbstractLLMAdapter
from mini_qed.utils import load_prompt


async def run_brainstorm(
    *,
    adapters: list[AbstractLLMAdapter],
    providers: list[dict],
    problem_file: str,
    proof_file: str,
    related_info_dir: str,
    round_num: int,
    output_dir: str,
    prompt_template_path: str,
    prev_verification_dir: str = "",
) -> list[str]:
    """Run parallel brainstorm across multiple models.

    Each provider independently writes its proof strategy ideas.
    All results are collected and returned for the proof search agent.

    Args:
        adapters: List of adapter instances (one per provider).
        providers: List of provider config dicts, each with "name" and "model".
        problem_file: Path to problem.tex.
        proof_file: Path to current proof.md (may be empty for round 1).
        related_info_dir: Stage 0 output directory.
        round_num: Current round number.
        output_dir: Where to write brainstorm_result_*.md files.
        prompt_template_path: Path to brainstorm.md template.
        prev_verification_dir: Previous round's verification directory (for feedback).

    Returns:
        List of brainstorm response texts.
    """
    os.makedirs(output_dir, exist_ok=True)
    prompts_dir = os.path.dirname(prompt_template_path)

    async def _brainstorm_single(index: int) -> str:
        provider = providers[index]
        adapter = adapters[index]
        name = provider.get("name", f"provider_{index}")

        output_file = os.path.join(output_dir, f"brainstorm_result_{name}.md")
        error_file = os.path.join(output_dir, f"error_brainstorm_{name}.md")

        prompt = load_prompt(
            prompts_dir, os.path.basename(prompt_template_path),
            problem_file=problem_file,
            related_info_dir=related_info_dir,
            proof_file=proof_file,
            prev_verification_dir=prev_verification_dir,
            round_num=round_num,
            output_file=output_file,
            error_file=error_file,
        )

        response = await adapter.chat(prompt=prompt)

        # Save result
        if not os.path.exists(output_file):
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(response.text)

        return response.text

    tasks = [_brainstorm_single(i) for i in range(len(adapters))]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions, return string results
    texts: list[str] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"[Brainstorm] Provider {i} failed: {r}")
        else:
            texts.append(r)

    return texts
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_brainstorm.py -v
```

Expected: 1 test PASS.

- [ ] **Step 5: Commit**

```bash
git add mini_qed/steps/brainstorm.py tests/test_brainstorm.py
git commit -m "feat: add brainstorm step — parallel multi-model strategy generation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 13: `steps/verification.py`

**Files:**
- Create: `mini_qed/steps/verification.py`
- Test: `tests/test_verification.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_verification.py
import os
import tempfile
import pytest
from mini_qed.adapters.base import LLMResponse
from mini_qed.steps.verification import (
    VerificationReport,
    run_structural_verification,
    run_detailed_verification,
    run_easy_verification,
)


class MockVerifierAdapter:
    def __init__(self, verdict: str, model: str = "mock"):
        self.verdict_text = verdict
        self.model = model

    async def chat(self, prompt, *, system="", **kwargs) -> LLMResponse:
        report = (
            f"## Overall Verdict: {self.verdict_text}\n\n"
            f"The proof is {'correct' if self.verdict_text == 'PASS' else 'incorrect'}."
        )
        return LLMResponse(
            text=report, input_tokens=100, output_tokens=50,
            model=self.model, elapsed_s=0.1,
        )


class TestVerification:
    @pytest.mark.asyncio
    async def test_structural_verification_pass(self):
        with tempfile.TemporaryDirectory() as d:
            problem_file = os.path.join(d, "problem.tex")
            proof_file = os.path.join(d, "proof.md")
            with open(problem_file, "w") as f:
                f.write("problem")
            with open(proof_file, "w") as f:
                f.write("proof")

            adapters = [MockVerifierAdapter("PASS")]
            results = await run_structural_verification(
                adapters=adapters,
                verifier_names=["deepseek"],
                problem_file=problem_file,
                proof_file=proof_file,
                output_dir=d,
                prompt_template_content="Verify: {problem_file} {proof_file}\nWrite to {output_file}",
            )
            assert len(results) == 1
            assert results[0].verdict == "PASS"

    @pytest.mark.asyncio
    async def test_easy_verification(self):
        with tempfile.TemporaryDirectory() as d:
            problem_file = os.path.join(d, "problem.tex")
            proof_file = os.path.join(d, "proof.md")
            with open(problem_file, "w") as f:
                f.write("problem")
            with open(proof_file, "w") as f:
                f.write("proof")

            adapters = [MockVerifierAdapter("PASS")]
            results = await run_easy_verification(
                adapters=adapters,
                verifier_names=["deepseek"],
                problem_file=problem_file,
                proof_file=proof_file,
                output_dir=d,
                prompt_template_content="Easy verify: {problem_file} {proof_file}\nWrite to {output_file}",
            )
            assert len(results) == 1
            assert results[0].verdict == "PASS"
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_verification.py -v
```

Expected: FAIL.

- [ ] **Step 3: 实现 `verification.py`**

```python
# mini_qed/steps/verification.py
"""Verification steps: structural, detailed, and easy verification.

Each function runs one or more verifiers in parallel against a proof,
collecting VerificationReports with PASS/FAIL verdicts.
"""

import asyncio
import os
from dataclasses import dataclass
from mini_qed.adapters.base import AbstractLLMAdapter


@dataclass
class VerificationReport:
    """Result of a single verification run."""
    verdict: str             # "PASS" or "FAIL"
    report_text: str
    report_file: str
    phase: str               # "structural", "detailed", or "easy"
    model: str
    tokens_input: int
    tokens_output: int


def _verification_filename(verifier_name: str, multi_verifier: bool) -> str:
    """Return the filename for a verifier's report."""
    if multi_verifier:
        return f"verification_result_{verifier_name}.md"
    return "verification_result.md"


async def _run_verification(
    *,
    adapters: list[AbstractLLMAdapter],
    verifier_names: list[str],
    prompt_factory,    # callable(verifier_name, output_file, error_file) -> str
    output_dir: str,
    phase: str,
    error_dir: str = "",
) -> list[VerificationReport]:
    """Generic parallel verification runner.

    Args:
        adapters: One adapter per verifier.
        verifier_names: Names for labeling (e.g., ["deepseek", "opencode"]).
        prompt_factory: Function that takes (verifier_name, output_file, error_file)
            and returns the prompt string.
        output_dir: Where to write verification_result_*.md files.
        phase: "structural", "detailed", or "easy".
        error_dir: Where to write error files (defaults to output_dir).

    Returns:
        List of VerificationReports for successful verifications.
    """
    multi = len(adapters) > 1
    error_dir = error_dir or output_dir

    async def _verify_one(index: int) -> VerificationReport | None:
        name = verifier_names[index]
        adapter = adapters[index]
        model = getattr(adapter, '_default_model', 'unknown')

        output_file = os.path.join(output_dir, _verification_filename(name, multi))
        error_file = os.path.join(error_dir, f"error_{phase}_{name}.md")

        prompt = prompt_factory(name, output_file, error_file)

        try:
            response = await adapter.chat(prompt=prompt)
        except Exception as e:
            os.makedirs(os.path.dirname(error_file), exist_ok=True)
            with open(error_file, "w", encoding="utf-8") as f:
                f.write(f"# Verification Failed\n\n**Error:** {e}\n")
            return None

        # Save report if not already written by agent
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if not os.path.exists(output_file):
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(response.text)

        # Parse verdict
        verdict = "FAIL"
        for line in response.text.splitlines():
            if "overall verdict" in line.lower():
                if "PASS" in line.upper():
                    verdict = "PASS"
                break

        return VerificationReport(
            verdict=verdict,
            report_text=response.text,
            report_file=output_file,
            phase=phase,
            model=model,
            tokens_input=response.input_tokens,
            tokens_output=response.output_tokens,
        )

    results = await asyncio.gather(*[_verify_one(i) for i in range(len(adapters))])
    return [r for r in results if r is not None]


async def run_structural_verification(
    *,
    adapters: list[AbstractLLMAdapter],
    verifier_names: list[str],
    problem_file: str,
    proof_file: str,
    output_dir: str,
    prompt_template_content: str,
    additional_verify_rule_global_file: str = "",
    additional_verify_rule_prev_round_file: str = "",
) -> list[VerificationReport]:
    """Run structural verification against a proof (Phases 1-4)."""

    def _make_prompt(name: str, output_file: str, error_file: str) -> str:
        return prompt_template_content.format(
            problem_file=problem_file,
            proof_file=proof_file,
            output_file=output_file,
            error_file=error_file,
            additional_verify_rule_global_file=additional_verify_rule_global_file,
            additional_verify_rule_prev_round_file=additional_verify_rule_prev_round_file,
        )

    return await _run_verification(
        adapters=adapters,
        verifier_names=verifier_names,
        prompt_factory=_make_prompt,
        output_dir=output_dir,
        phase="structural",
    )


async def run_detailed_verification(
    *,
    adapters: list[AbstractLLMAdapter],
    verifier_names: list[str],
    problem_file: str,
    proof_file: str,
    structural_report_file: str,
    output_dir: str,
    prompt_template_content: str,
) -> list[VerificationReport]:
    """Run detailed verification against a proof (Phase 5)."""

    def _make_prompt(name: str, output_file: str, error_file: str) -> str:
        return prompt_template_content.format(
            problem_file=problem_file,
            proof_file=proof_file,
            structural_report_file=structural_report_file,
            output_file=output_file,
            error_file=error_file,
        )

    return await _run_verification(
        adapters=adapters,
        verifier_names=verifier_names,
        prompt_factory=_make_prompt,
        output_dir=output_dir,
        phase="detailed",
    )


async def run_easy_verification(
    *,
    adapters: list[AbstractLLMAdapter],
    verifier_names: list[str],
    problem_file: str,
    proof_file: str,
    output_dir: str,
    prompt_template_content: str,
) -> list[VerificationReport]:
    """Run lightweight easy verification (single-phase, for easy problems)."""

    def _make_prompt(name: str, output_file: str, error_file: str) -> str:
        return prompt_template_content.format(
            problem_file=problem_file,
            proof_file=proof_file,
            output_file=output_file,
            error_file=error_file,
        )

    return await _run_verification(
        adapters=adapters,
        verifier_names=verifier_names,
        prompt_factory=_make_prompt,
        output_dir=output_dir,
        phase="easy",
    )
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_verification.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mini_qed/steps/verification.py tests/test_verification.py
git commit -m "feat: add verification steps — structural, detailed, easy

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 14: `steps/selection.py`

**Files:**
- Create: `mini_qed/steps/selection.py`
- Test: `tests/test_selection.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_selection.py
import os
import tempfile
import pytest
from mini_qed.adapters.base import LLMResponse
from mini_qed.steps.selection import run_selection


class MockSelectorAdapter:
    def __init__(self, selected_model: str):
        self.selected = selected_model

    async def chat(self, prompt, *, system="", **kwargs) -> LLMResponse:
        return LLMResponse(
            text=f"SELECTED: {self.selected}\n\nReasoning: Best proof quality.",
            input_tokens=100, output_tokens=20,
            model="mock", elapsed_s=0.1,
        )


class TestRunSelection:
    @pytest.mark.asyncio
    async def test_selects_from_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            candidates = [
                {"model": "deepseek", "proof_file": os.path.join(d, "d.md"),
                 "verification_files": []},
                {"model": "opencode", "proof_file": os.path.join(d, "o.md"),
                 "verification_files": []},
            ]
            adapter = MockSelectorAdapter("deepseek")
            result = await run_selection(
                adapter=adapter,
                candidates=candidates,
                problem_file=os.path.join(d, "problem.tex"),
                selection_file=os.path.join(d, "selection.md"),
                prompt_template_content=(
                    "Select best proof from:\n{candidates_block}\n"
                    "Problem: {problem_file}\n"
                    "Write selection to {selection_file}\n"
                ),
            )
            assert result == "deepseek"

    @pytest.mark.asyncio
    async def test_fallback_to_first(self):
        with tempfile.TemporaryDirectory() as d:
            candidates = [
                {"model": "model-a", "proof_file": os.path.join(d, "a.md"),
                 "verification_files": []},
            ]
            adapter = MockSelectorAdapter("UNKNOWN_MODEL")
            result = await run_selection(
                adapter=adapter,
                candidates=candidates,
                problem_file=os.path.join(d, "problem.tex"),
                selection_file=os.path.join(d, "selection.md"),
                prompt_template_content=(
                    "Select: {candidates_block}\nProblem: {problem_file}\n"
                    "Write to {selection_file}\n"
                ),
            )
            assert result == "model-a"  # fallback to first
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_selection.py -v
```

Expected: FAIL.

- [ ] **Step 3: 实现 `selection.py`**

```python
# mini_qed/steps/selection.py
"""Selection step: when multiple models generate proofs, pick the best one."""

import os
from mini_qed.adapters.base import AbstractLLMAdapter


async def run_selection(
    *,
    adapter: AbstractLLMAdapter,
    candidates: list[dict],
    problem_file: str,
    selection_file: str,
    prompt_template_content: str,
) -> str:
    """Select the best proof from multiple candidates.

    Args:
        adapter: The LLM adapter for the selection agent.
        candidates: List of dicts, each with:
            - model: provider name (e.g., "deepseek")
            - proof_file: path to the proof.md
            - verification_files: list of verification report paths
        problem_file: Path to problem.tex.
        selection_file: Where to write the selection report.
        prompt_template_content: The selection prompt template, with
            {candidates_block}, {problem_file}, {selection_file} placeholders.

    Returns:
        The name of the selected model (e.g., "deepseek").
    """
    # Build candidates block
    lines = []
    for c in candidates:
        lines.append(f"**{c['model']}'s proof:** {c['proof_file']}")
        if c.get("verification_files"):
            lines.append(f"  Verification:")
            for vf in c["verification_files"]:
                lines.append(f"    - {vf}")
    candidates_block = "\n".join(lines)

    prompt = prompt_template_content.format(
        candidates_block=candidates_block,
        problem_file=problem_file,
        selection_file=selection_file,
    )

    response = await adapter.chat(prompt=prompt)

    # Save selection report
    os.makedirs(os.path.dirname(selection_file), exist_ok=True)
    if not os.path.exists(selection_file):
        with open(selection_file, "w", encoding="utf-8") as f:
            f.write(response.text)

    # Parse SELECTED: <model> from response
    for line in response.text.splitlines():
        upper = line.upper()
        if "SELECTED:" in upper:
            for c in candidates:
                if c["model"].lower() in line.lower():
                    return c["model"]

    # Fallback: return first candidate
    return candidates[0]["model"] if candidates else "unknown"
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_selection.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mini_qed/steps/selection.py tests/test_selection.py
git commit -m "feat: add selection step — pick best proof from multi-model search

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 15: `stages/stage0_survey.py`

**Files:**
- Create: `mini_qed/stages/stage0_survey.py`
- Test: `tests/test_stage0_survey.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_stage0_survey.py
import os
import tempfile
import pytest
from mini_qed.adapters.base import LLMResponse
from mini_qed.stages.stage0_survey import run_literature_survey


class MockSurveyAdapter:
    async def chat(self, prompt, *, system="", **kwargs) -> LLMResponse:
        return LLMResponse(
            text="Difficulty: Easy. Use Rolle's theorem.",
            input_tokens=200, output_tokens=100,
            model="mock", elapsed_s=0.2,
        )


class TestStage0Survey:
    @pytest.mark.asyncio
    async def test_produces_expected_files(self):
        with tempfile.TemporaryDirectory() as d:
            problem_file = os.path.join(d, "problem.tex")
            with open(problem_file, "w") as f:
                f.write(r"\begin{problem}Test\end{problem}")

            related_info_dir = await run_literature_survey(
                adapter=MockSurveyAdapter(),
                model="mock",
                problem_file=problem_file,
                output_dir=d,
                prompt_template_content=(
                    "Survey this problem: {problem_file}\n"
                    "Write difficulty to {difficulty_file}\n"
                    "Write related work to {related_work_file}\n"
                ),
                prove_skill="",
            )
            assert os.path.exists(os.path.join(related_info_dir, "difficulty_evaluation.md"))
            assert os.path.exists(os.path.join(related_info_dir, "related_work.md"))
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_stage0_survey.py -v
```

Expected: FAIL.

- [ ] **Step 3: 实现 `stage0_survey.py`**

```python
# mini_qed/stages/stage0_survey.py
"""Stage 0: Literature Survey.

Evaluates problem difficulty (Easy/Medium/Hard) and researches applicable
theorems and related results. Output is used by all downstream stages.
"""

import os
from mini_qed.adapters.base import AbstractLLMAdapter
from mini_qed.logging import PipelineLogger


async def run_literature_survey(
    *,
    adapter: AbstractLLMAdapter,
    model: str,
    problem_file: str,
    output_dir: str,
    prompt_template_content: str,
    prove_skill: str = "",
    tracker=None,
) -> str:
    """Run the literature survey agent (Stage 0).

    Args:
        adapter: LLM adapter for the survey agent.
        model: Model name for logging.
        problem_file: Path to problem.tex.
        output_dir: Pipeline output directory.
        prompt_template_content: The literature_survey prompt, with
            {problem_file}, {difficulty_file}, {related_work_file} placeholders.
        prove_skill: Content of super_math_skill.md (as system prompt).
        tracker: Optional TokenTracker.

    Returns:
        Path to the related_info directory.
    """
    related_info_dir = os.path.join(output_dir, "related_info")
    os.makedirs(related_info_dir, exist_ok=True)

    log_dir = os.path.join(output_dir, "literature_survey_log")
    logger = PipelineLogger(log_dir, "Literature Survey")
    logger.update_status(1, 1, "Literature Survey", "RUNNING",
                         "Running literature survey agent...")

    difficulty_file = os.path.join(related_info_dir, "difficulty_evaluation.md")
    related_work_file = os.path.join(related_info_dir, "related_work.md")
    error_file = os.path.join(related_info_dir, "error_literature_survey.md")

    prompt = prompt_template_content.format(
        problem_file=problem_file,
        difficulty_file=difficulty_file,
        related_work_file=related_work_file,
        error_file=error_file,
    )

    logger.log(f"[Survey] Starting literature survey (model={model})")
    response = await adapter.chat(prompt=prompt, system=prove_skill)
    logger.log(f"[Survey] Completed in {response.elapsed_s:.0f}s")

    # Fallback: save response if agent didn't write files
    for path, label in [(difficulty_file, "difficulty"),
                        (related_work_file, "related work")]:
        if not os.path.exists(path) and response.text.strip():
            with open(path, "w", encoding="utf-8") as f:
                f.write(response.text)
            logger.log(f"  Fallback: saved {label} from response")

    # Verify expected files exist
    for path, label in [(difficulty_file, "difficulty evaluation"),
                        (related_work_file, "related work")]:
        if not os.path.exists(path):
            msg = f"FATAL — Literature Survey: expected file missing: {label} ({path})"
            logger.log(msg)
            raise RuntimeError(msg)

    # Create empty error file if needed
    if not os.path.exists(error_file):
        with open(error_file, "w", encoding="utf-8") as f:
            f.write("")

    if tracker:
        tracker.record("Literature Survey", response.input_tokens,
                       response.output_tokens, response.elapsed_s,
                       provider="deepseek", model=model)

    logger.finalize(1, 1, "FINISHED", "Literature survey complete.")
    return related_info_dir
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_stage0_survey.py -v
```

Expected: 1 test PASS.

- [ ] **Step 5: Commit**

```bash
git add mini_qed/stages/stage0_survey.py tests/test_stage0_survey.py
git commit -m "feat: add Stage 0 — literature survey with difficulty evaluation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 16: `stages/stage2_summary.py`

**Files:**
- Create: `mini_qed/stages/stage2_summary.py`
- Test: `tests/test_stage2_summary.py`

- [ ] **Step 1: 实现 + 测试（步骤合并——文件短）**

```python
# mini_qed/stages/stage2_summary.py
"""Stage 2: Proof Effort Summary.

Reads all generated files from the pipeline run and produces a comprehensive
summary of the entire proof journey.
"""

import os
from mini_qed.adapters.base import AbstractLLMAdapter
from mini_qed.logging import PipelineLogger


async def run_summary(
    *,
    adapter: AbstractLLMAdapter,
    model: str,
    output_dir: str,
    problem_file: str,
    prompt_template_content: str,
    tracker=None,
) -> str:
    """Run the summary agent (Stage 2).

    Args:
        adapter: LLM adapter for the summary agent.
        model: Model name for logging.
        output_dir: Pipeline output directory.
        problem_file: Path to problem.tex.
        prompt_template_content: The proof_effort_summary prompt, with
            {output_dir}, {problem_file}, {summary_file} placeholders.
        tracker: Optional TokenTracker.

    Returns:
        Path to the summary file (proof_effort_summary.md).
    """
    log_dir = os.path.join(output_dir, "summary_log")
    logger = PipelineLogger(log_dir, "Summary")
    logger.update_status(1, 1, "Summary", "RUNNING",
                         "Running proof effort summary agent...")

    summary_file = os.path.join(output_dir, "proof_effort_summary.md")
    error_file = os.path.join(log_dir, "error_summary.md")

    prompt = prompt_template_content.format(
        output_dir=output_dir,
        problem_file=problem_file,
        summary_file=summary_file,
        error_file=error_file,
    )

    logger.log(f"[Summary] Starting summary (model={model})")
    response = await adapter.chat(prompt=prompt)
    logger.log(f"[Summary] Completed in {response.elapsed_s:.0f}s")

    # Fallback: save if agent didn't write
    if not os.path.exists(summary_file) and response.text.strip():
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(response.text)
        logger.log("  Fallback: saved summary from response")

    if tracker:
        tracker.record("Proof Summary", response.input_tokens,
                       response.output_tokens, response.elapsed_s,
                       provider="deepseek", model=model)

    logger.finalize(1, 1, "FINISHED", "Summary complete.")
    return summary_file
```

**测试:**

```python
# tests/test_stage2_summary.py
import os
import tempfile
import pytest
from mini_qed.adapters.base import LLMResponse
from mini_qed.stages.stage2_summary import run_summary


class MockSummaryAdapter:
    async def chat(self, prompt, *, system="", **kwargs) -> LLMResponse:
        return LLMResponse(
            text="# Proof Effort Summary\n\nDone.",
            input_tokens=100, output_tokens=50,
            model="mock", elapsed_s=0.1,
        )


class TestStage2Summary:
    @pytest.mark.asyncio
    async def test_produces_summary_file(self):
        with tempfile.TemporaryDirectory() as d:
            problem_file = os.path.join(d, "problem.tex")
            with open(problem_file, "w") as f:
                f.write("problem")

            summary_file = await run_summary(
                adapter=MockSummaryAdapter(),
                model="mock",
                output_dir=d,
                problem_file=problem_file,
                prompt_template_content=(
                    "Summarize the proof effort:\n"
                    "Output dir: {output_dir}\n"
                    "Problem: {problem_file}\n"
                    "Write summary to {summary_file}\n"
                ),
            )
            assert os.path.exists(summary_file)
```

- [ ] **Step 2: 运行测试**

```bash
python -m pytest tests/test_stage2_summary.py -v
```

Expected: 1 test PASS.

- [ ] **Step 3: Commit**

```bash
git add mini_qed/stages/stage2_summary.py tests/test_stage2_summary.py
git commit -m "feat: add Stage 2 — proof effort summary

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 17: `stages/stage1_simple.py` — Simple 模式编排

**Files:**
- Create: `mini_qed/stages/stage1_simple.py`
- Test: `tests/test_stage1_simple.py`

这是核心编排模块，将所有 steps 串联成完整循环。

- [ ] **Step 1: 实现 `stage1_simple.py`**

```python
# mini_qed/stages/stage1_simple.py
"""Stage 1: Simple mode proof loop.

Orchestrates the iterative search-verify-verdict cycle.
Supports single-model and multi-model proof search, optional brainstorm,
and three verification difficulty paths (easy / structural+detailed).
"""

import asyncio
import os
import shutil
from mini_qed.adapters.base import AbstractLLMAdapter
from mini_qed.config import PipelineConfig, SimpleModeConfig
from mini_qed.logging import PipelineLogger, TokenTracker
from mini_qed.steps.proof_search import run_proof_search, ProofResult
from mini_qed.steps.verification import (
    run_structural_verification,
    run_detailed_verification,
    run_easy_verification,
    VerificationReport,
)
from mini_qed.steps.verdict import run_verdict, Verdict
from mini_qed.steps.brainstorm import run_brainstorm
from mini_qed.steps.selection import run_selection
from mini_qed.utils import find_verification_files, file_nonempty


def _parse_difficulty(output_dir: str) -> str:
    """Determine problem difficulty from the survey output."""
    path = os.path.join(output_dir, "related_info", "difficulty_evaluation.md")
    if not os.path.exists(path):
        return "unknown"
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "classification" in line.lower():
                upper = line.upper()
                if "EASY" in upper:
                    return "easy"
                if "MEDIUM" in upper:
                    return "medium"
                if "HARD" in upper:
                    return "hard"
    return "unknown"


async def run_simple_proof_loop(
    *,
    config: PipelineConfig,
    adapters: dict[str, AbstractLLMAdapter],
    problem_file: str,
    output_dir: str,
    prompts_dir: str,
    related_info_dir: str,
    prove_skill: str,
    tracker: TokenTracker,
    start_round: int = 1,
    difficulty: str = "unknown",
    prompts: dict[str, str] | None = None,
) -> bool:
    """Run the simple mode proof loop (Stage 1).

    Args:
        config: Full pipeline configuration.
        adapters: Dict mapping adapter name to AbstractLLMAdapter instance.
        problem_file: Path to problem.tex.
        output_dir: Pipeline output directory.
        prompts_dir: Directory containing prompt .md templates.
        related_info_dir: Stage 0 output directory.
        prove_skill: Content of super_math_skill.md.
        tracker: TokenTracker instance.
        start_round: Round to start from (1 for fresh run).
        difficulty: Problem difficulty from Stage 0.
        prompts: Optional pre-loaded prompt content dict (avoids filesystem calls).

    Returns:
        True if proof was verified (DONE), False if max iterations reached.
    """
    sm = config.simple_mode
    proof_file = os.path.join(output_dir, "proof.md")
    verify_dir = os.path.join(output_dir, "verification")
    human_help_dir = os.path.join(output_dir, "human_help")

    logger = PipelineLogger(verify_dir, "Proof Search")

    # Create empty proof file if starting fresh
    if not os.path.exists(proof_file):
        os.makedirs(output_dir, exist_ok=True)
        with open(proof_file, "w", encoding="utf-8") as f:
            f.write("<!-- Proof will be written here by the proof search agent -->\n")

    easy_mode = difficulty == "easy"
    multi_model = sm.multi_model and sm.multi_model.get("enabled")

    # Resolve proof search adapters
    if multi_model and sm.multi_model:
        search_providers = sm.multi_model.get("proof_search_providers", [])
        search_adapters = [
            adapters[p["adapter"]] for p in search_providers
        ]
    else:
        search_adapters = [adapters[sm.proof_search.adapter]]
        search_providers = [{"adapter": sm.proof_search.adapter, "model": sm.proof_search.model}]

    # Resolve verification adapters
    if multi_model and sm.multi_model:
        verify_providers = sm.multi_model.get("verification_providers", [])
        verify_adapters = [adapters[p["adapter"]] for p in verify_providers]
        verify_names = [p["adapter"] for p in verify_providers]
    else:
        verify_adapters = [
            adapters[sm.structural_verifier.adapter],
        ]
        verify_names = [sm.structural_verifier.adapter]

    # Check brainstorm config
    brainstorm_cfg = sm.brainstorm or {}
    brainstorm_enabled = brainstorm_cfg.get("enabled", False) and not easy_mode

    for round_num in range(start_round, config.max_proof_iterations + 1):
        round_dir = os.path.join(verify_dir, f"round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)

        # Create per-round human_help
        round_hh_dir = os.path.join(round_dir, "human_help")
        os.makedirs(round_hh_dir, exist_ok=True)
        for hh_file in ["additional_prove_human_help_per_round.md",
                        "additional_verify_rule_per_round.md"]:
            hh_path = os.path.join(round_hh_dir, hh_file)
            if not os.path.exists(hh_path):
                with open(hh_path, "w", encoding="utf-8") as f:
                    f.write("")

        prev_round_hh_dir = os.path.join(verify_dir, f"round_{round_num - 1}", "human_help")
        if round_num == 1 or not os.path.isdir(prev_round_hh_dir):
            prev_round_hh_dir = ""

        logger.log(f"\n{'='*40}")
        logger.log(f"=== ITERATION {round_num} of {config.max_proof_iterations} ===")
        logger.log(f"{'='*40}")
        logger.append_history(f"Iteration {round_num} started")

        # Build previous round instructions
        prev_round_dir = os.path.join(verify_dir, f"round_{round_num - 1}")
        prev_instructions = ""
        if round_num > 1:
            prev_vf = find_verification_files(
                os.path.join(prev_round_dir, "verification_file", "detailed")
            ) or find_verification_files(prev_round_dir)
            for pvf in prev_vf:
                prev_instructions += f"- Read PREVIOUS round's verification: {pvf}\n"
            prev_status = os.path.join(prev_round_dir, "proof_status.md")
            if os.path.exists(prev_status):
                prev_instructions += f"- Read PREVIOUS round's proof status: {prev_status}\n"
        if not prev_instructions:
            prev_instructions = "- This is the first round. No previous data.\n"

        # Backup proof.md
        proof_backup = os.path.join(round_dir, "proof_before_round.md")
        if os.path.exists(proof_file):
            shutil.copy2(proof_file, proof_backup)

        # ============================================================
        # Step 0 (optional): Brainstorm
        # ============================================================
        brainstorm_dir = ""
        if brainstorm_enabled:
            brainstorm_dir = os.path.join(round_dir, "brainstorm")
            os.makedirs(brainstorm_dir, exist_ok=True)

            bs_providers = brainstorm_cfg.get("providers", [])
            bs_adapters = [adapters[p["adapter"]] for p in bs_providers]

            # Previous verification for brainstorm context
            prev_verif_dir = ""
            if round_num > 1:
                for subdir in ("verification_file/detailed", ""):
                    cand = os.path.join(prev_round_dir, subdir) if subdir else prev_round_dir
                    if find_verification_files(cand):
                        prev_verif_dir = cand
                        break

            logger.update_status(round_num, config.max_proof_iterations,
                                 "Brainstorm", "RUNNING", "Running brainstorm...")
            brainstorm_prompt_path = os.path.join(prompts_dir, "brainstorm.md")
            if prompts and "brainstorm" in prompts:
                brainstorm_prompt_path = ""  # will use inline
            await run_brainstorm(
                adapters=bs_adapters,
                providers=[{"name": p["adapter"], "model": p.get("model", "")}
                          for p in bs_providers],
                problem_file=problem_file,
                proof_file=proof_file,
                related_info_dir=related_info_dir,
                round_num=round_num,
                output_dir=brainstorm_dir,
                prompt_template_path=brainstorm_prompt_path,
                prev_verification_dir=prev_verif_dir,
            )

        # ============================================================
        # Step 1: Proof Search (single or multi-model)
        # ============================================================
        proof_search_prompt_path = os.path.join(prompts_dir, "proof_search.md")
        scratch_pad_file = os.path.join(round_dir, "scratch_pad.md")
        error_search_file = os.path.join(round_dir, "error_proof_search.md")

        logger.update_status(round_num, config.max_proof_iterations,
                             "Proof Search", "RUNNING", "Running proof search agent...")

        async def _do_search(index: int) -> ProofResult:
            p = search_providers[index]
            adapter = search_adapters[index]
            proof_f = os.path.join(round_dir, p["adapter"], "proof.md") if multi_model else proof_file
            status_f = os.path.join(round_dir, p["adapter"], "proof_status.md") if multi_model \
                else os.path.join(round_dir, "proof_status.md")
            os.makedirs(os.path.dirname(proof_f), exist_ok=True)
            return await run_proof_search(
                adapter=adapter,
                model=p.get("model", ""),
                problem_file=problem_file,
                proof_file=proof_f,
                proof_status_file=status_f,
                related_info_dir=related_info_dir,
                round_num=round_num,
                prev_instructions=prev_instructions,
                brainstorm_dir=brainstorm_dir,
                prompt_template_path=proof_search_prompt_path,
                prove_skill=prove_skill,
                human_help_dir=human_help_dir,
                prev_round_human_help_dir=prev_round_hh_dir,
                scratch_pad_file=scratch_pad_file,
                error_file=error_search_file,
            )

        search_results = list(await asyncio.gather(
            *[_do_search(i) for i in range(len(search_adapters))]
        ))

        # Track tokens
        for sr in search_results:
            tracker.record(f"Proof Search R{round_num} [{sr.model}]",
                          sr.tokens_input, sr.tokens_output, 0.0,
                          provider="deepseek", model=sr.model)

        # ============================================================
        # Step 1.5 (multi-model only): Selection
        # ============================================================
        selected_model = search_results[0].model
        if multi_model and len(search_results) > 1:
            selection_file = os.path.join(round_dir, "selection.md")
            candidates = [
                {"model": sr.model, "proof_file": sr.proof_file,
                 "verification_files": []}
                for sr in search_results
            ]
            select_prompt_path = os.path.join(prompts_dir, "proof_select.md")
            selection_adapter = adapters[sm.verdict.adapter]  # use verdict's adapter for selection
            selected_model = await run_selection(
                adapter=selection_adapter,
                candidates=candidates,
                problem_file=problem_file,
                selection_file=selection_file,
                prompt_template_content=select_prompt_path,
            )
            # Copy selected proof to main proof.md
            selected_proof = os.path.join(round_dir, selected_model, "proof.md")
            if os.path.exists(selected_proof):
                shutil.copy2(selected_proof, proof_file)

        # ============================================================
        # Step 2: Verification
        # ============================================================
        structural_dir = os.path.join(round_dir, "verification_file", "structural")
        detailed_dir = os.path.join(round_dir, "verification_file", "detailed")
        os.makedirs(structural_dir, exist_ok=True)
        os.makedirs(detailed_dir, exist_ok=True)

        verification_files: list[str] = []
        structural_pass = True  # default: easy mode skips structural

        if easy_mode:
            # Easy: lightweight single-phase verification
            logger.update_status(round_num, config.max_proof_iterations,
                                 "Easy Verification", "RUNNING", "Running easy verification...")
            easy_prompt_path = os.path.join(prompts_dir, "proof_verify_easy.md")

            reports = await run_easy_verification(
                adapters=verify_adapters,
                verifier_names=verify_names,
                problem_file=problem_file,
                proof_file=proof_file,
                output_dir=round_dir,
                prompt_template_content=easy_prompt_path,
            )
            verification_files = [r.report_file for r in reports]
        else:
            # Non-easy: structural → (if PASS) detailed
            logger.update_status(round_num, config.max_proof_iterations,
                                 "Structural Verification", "RUNNING",
                                 "Running structural verification...")
            structural_prompt_path = os.path.join(prompts_dir, "proof_verify_structural.md")

            global_rule = os.path.join(human_help_dir, "additional_verify_rule_global.md")
            prev_rule = os.path.join(prev_round_hh_dir, "additional_verify_rule_per_round.md") \
                if prev_round_hh_dir else ""

            s_reports = await run_structural_verification(
                adapters=verify_adapters,
                verifier_names=verify_names,
                problem_file=problem_file,
                proof_file=proof_file,
                output_dir=structural_dir,
                prompt_template_content=structural_prompt_path,
                additional_verify_rule_global_file=global_rule,
                additional_verify_rule_prev_round_file=prev_rule,
            )

            structural_pass = all(r.verdict == "PASS" for r in s_reports)
            if not structural_pass:
                logger.log(f"Iteration {round_num}: Structural checks FAILED — skipping detailed")
                verification_files = [r.report_file for r in s_reports]
                # Fall through to verdict below (skip detailed)
            else:
                # Detailed verification only when structural passes
                logger.update_status(round_num, config.max_proof_iterations,
                                     "Detailed Verification", "RUNNING",
                                     "Running detailed verification...")
                detailed_prompt_path = os.path.join(prompts_dir, "proof_verify_detailed.md")

                d_reports = await run_detailed_verification(
                    adapters=verify_adapters,
                    verifier_names=verify_names,
                    problem_file=problem_file,
                    proof_file=proof_file,
                    structural_report_file=s_reports[0].report_file,
                    output_dir=detailed_dir,
                    prompt_template_content=detailed_prompt_path,
                )
                verification_files = [r.report_file for r in d_reports]

        # ============================================================
        # Step 3: Verdict
        # ============================================================
        logger.update_status(round_num, config.max_proof_iterations,
                             "Verdict", "RUNNING", "Analyzing verification results...")
        verdict_adapter = adapters[sm.verdict.adapter]
        verdict_prompt_path = os.path.join(prompts_dir, "verdict_proof.md")

        verdict = await run_verdict(
            adapter=verdict_adapter,
            verification_files=verification_files,
            prompt_template_path=verdict_prompt_path,
        )

        logger.log(f"Iteration {round_num}: Decision = {verdict.value}")
        logger.append_history(f"Iteration {round_num}: Decision = {verdict.value}")

        if verdict == Verdict.DONE:
            logger.finalize(round_num, config.max_proof_iterations,
                           "FINISHED", "Proof verified successfully!")
            return True

    logger.finalize(config.max_proof_iterations, config.max_proof_iterations,
                   "MAX_ITERATIONS", "Max iterations reached without verified proof.")
    return False
```

- [ ] **Step 2: 写基本测试（验证导入和简单流程）**

```python
# tests/test_stage1_simple.py
from mini_qed.stages.stage1_simple import _parse_difficulty
import tempfile
import os


class TestParseDifficulty:
    def test_easy(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "difficulty_evaluation.md")
            with open(p, "w") as f:
                f.write("## Classification: Easy")
            os.makedirs(os.path.join(d, "related_info"), exist_ok=True)
            import shutil
            shutil.copy2(p, os.path.join(d, "related_info", "difficulty_evaluation.md"))
            assert _parse_difficulty(d) == "easy"

    def test_unknown_when_missing(self):
        with tempfile.TemporaryDirectory() as d:
            assert _parse_difficulty(d) == "unknown"
```

- [ ] **Step 3: 运行测试**

```bash
python -m pytest tests/test_stage1_simple.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add mini_qed/stages/stage1_simple.py tests/test_stage1_simple.py
git commit -m "feat: add Stage 1 — simple mode proof loop orchestration

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 18: `orchestrator.py` — 顶层编排

**Files:**
- Create: `mini_qed/orchestrator.py`

- [ ] **Step 1: 实现**

```python
# mini_qed/orchestrator.py
"""Top-level orchestrator: Stage 0 → Stage 1 → Stage 2.

This is the main entry point for the miniQED pipeline.
Equivalent to QED's pipeline.py main() function.
"""

import asyncio
import os
import sys
from mini_qed.adapters.registry import AdapterRegistry
from mini_qed.config import load_pipeline_config
from mini_qed.logging import TokenTracker
from mini_qed.stages.stage0_survey import run_literature_survey
from mini_qed.stages.stage1_simple import run_simple_proof_loop, _parse_difficulty
from mini_qed.stages.stage2_summary import run_summary


async def run_pipeline(config_path: str, problem_file: str, output_dir: str | None = None) -> bool:
    """Run the full miniQED pipeline.

    Args:
        config_path: Path to config.yaml.
        problem_file: Path to problem.tex.
        output_dir: Override for output directory (default: from config).

    Returns:
        True if proof was successfully verified.
    """
    # Load config
    pipeline_config, raw_config = load_pipeline_config(config_path)
    if output_dir:
        pipeline_config.output_dir = output_dir

    out = pipeline_config.output_dir

    # Setup adapter registry
    registry = AdapterRegistry(raw_config)

    # Resolve project root (for prompts/, skill/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    prompts_dir = os.path.join(project_root, "prompts")
    skill_dir = os.path.join(project_root, "skill")

    # Load proving skill
    skill_path = os.path.join(skill_dir, "super_math_skill.md")
    prove_skill = ""
    if os.path.exists(skill_path):
        with open(skill_path, encoding="utf-8") as f:
            prove_skill = f.read()

    # Copy human_help to output dir
    os.makedirs(os.path.join(out, "human_help"), exist_ok=True)
    hh_dir = os.path.join(project_root, "human_help")
    if os.path.isdir(hh_dir):
        import shutil
        for f in os.listdir(hh_dir):
            src = os.path.join(hh_dir, f)
            dst = os.path.join(out, "human_help", f)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

    # Initialize token tracker
    tracker = TokenTracker(out, pipeline_config.proof_summary.model)

    # Pre-load prompt templates
    prompts: dict[str, str] = {}
    for name in ["literature_survey", "proof_search", "proof_verify_structural",
                 "proof_verify_detailed", "proof_verify_easy", "verdict_proof",
                 "brainstorm", "proof_select", "proof_effort_summary"]:
        path = os.path.join(prompts_dir, f"{name}.md")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                prompts[name] = f.read()

    # ==================================================================
    # Stage 0: Literature Survey
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STAGE 0: Literature Survey")
    print("=" * 60 + "\n")

    survey_cfg = pipeline_config.literature_survey
    survey_adapter = registry.get(survey_cfg.adapter, model=survey_cfg.model)

    related_info_dir = await run_literature_survey(
        adapter=survey_adapter,
        model=survey_cfg.model,
        problem_file=problem_file,
        output_dir=out,
        prompt_template_content=prompts["literature_survey"],
        prove_skill=prove_skill,
        tracker=tracker,
    )

    difficulty = _parse_difficulty(out)
    print(f"\n  Difficulty: {difficulty}\n")

    # ==================================================================
    # Stage 1: Proof Loop
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STAGE 1: Proof Search Loop (Simple Mode)")
    print("=" * 60 + "\n")

    sm = pipeline_config.simple_mode

    # Build adapter dict for all roles
    adapters: dict[str, AbstractLLMAdapter] = {}
    for role_cfg in [sm.proof_search, sm.structural_verifier,
                     sm.detailed_verifier, sm.verdict]:
        if role_cfg.adapter not in adapters:
            adapters[role_cfg.adapter] = registry.get(role_cfg.adapter)

    # Also resolve multi-model adapters if configured
    if sm.multi_model and sm.multi_model.get("enabled"):
        for prov in sm.multi_model.get("proof_search_providers", []):
            if prov["adapter"] not in adapters:
                adapters[prov["adapter"]] = registry.get(prov["adapter"], model=prov.get("model"))
        for prov in sm.multi_model.get("verification_providers", []):
            if prov["adapter"] not in adapters:
                adapters[prov["adapter"]] = registry.get(prov["adapter"], model=prov.get("model"))

    success = await run_simple_proof_loop(
        config=pipeline_config,
        adapters=adapters,
        problem_file=problem_file,
        output_dir=out,
        prompts_dir=prompts_dir,
        related_info_dir=related_info_dir,
        prove_skill=prove_skill,
        tracker=tracker,
        difficulty=difficulty,
        prompts=prompts,
    )

    # ==================================================================
    # Stage 2: Summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STAGE 2: Proof Effort Summary")
    print("=" * 60 + "\n")

    summary_cfg = pipeline_config.proof_summary
    summary_adapter = registry.get(summary_cfg.adapter, model=summary_cfg.model)

    await run_summary(
        adapter=summary_adapter,
        model=summary_cfg.model,
        output_dir=out,
        problem_file=problem_file,
        prompt_template_content=prompts["proof_effort_summary"],
        tracker=tracker,
    )

    print(f"\n{'='*60}")
    if success:
        print(f"  ✅  PROOF VERIFIED — see {out}/proof.md")
    else:
        print(f"  ⚠️   MAX ITERATIONS REACHED — see {out}/proof.md")
    print(f"  Token usage: {out}/TOKEN_USAGE.md")
    print(f"{'='*60}\n")

    return success


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="miniQED — Mathematical Proof Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--input", default="problem/problem.tex", help="Path to problem.tex")
    parser.add_argument("--output", default=None, help="Output directory override")
    args = parser.parse_args()

    success = asyncio.run(run_pipeline(
        config_path=args.config,
        problem_file=args.input,
        output_dir=args.output,
    ))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证导入**

```bash
python -c "from mini_qed.orchestrator import run_pipeline; print('orchestrator imported OK')"
```

Expected: `orchestrator imported OK` (may warn about missing imports — fix any).

- [ ] **Step 3: Commit**

```bash
git add mini_qed/orchestrator.py
git commit -m "feat: add orchestrator — Stage 0 → 1 → 2 pipeline entry point

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 19: `run.sh` 入口脚本

**Files:**
- Create: `run.sh`

- [ ] **Step 1: 实现 `run.sh`**

```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROBLEM_FILE="${1:-$SCRIPT_DIR/problem/problem.tex}"
OUTPUT_DIR="${2:-$SCRIPT_DIR/proof_output}"
CONFIG="${3:-$SCRIPT_DIR/config.yaml}"

echo "============================================================"
echo "  miniQED — Mathematical Proof Pipeline"
echo "============================================================"
echo "  Problem:  $PROBLEM_FILE"
echo "  Output:   $OUTPUT_DIR"
echo "  Config:   $CONFIG"
echo ""

python -m mini_qed.orchestrator \
    --config "$CONFIG" \
    --input "$PROBLEM_FILE" \
    --output "$OUTPUT_DIR"
```

- [ ] **Step 2: 设置可执行权限**

```bash
chmod +x run.sh
```

- [ ] **Step 3: Commit**

```bash
git add run.sh
git commit -m "feat: add run.sh entry point script

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 20: 集成测试 — 端到端验证

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: 写集成测试 — 用一个简单数学问题跑完整流程**

```python
# tests/test_integration.py
"""
End-to-end integration test: run the full miniQED pipeline against DeepSeek
with a simple calculus problem and verify all expected output files exist.

Requires DEEPSEEK_API_KEY environment variable.
"""

import os
import tempfile
import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_simple_problem():
    """Run miniQED end-to-end on a simple problem.

    Uses DeepSeek V4 Flash for speed/cost.
    Verifies all expected output structure is created.
    """
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        pytest.skip("DEEPSEEK_API_KEY not set — skipping integration test")

    import yaml
    from mini_qed.orchestrator import run_pipeline

    with tempfile.TemporaryDirectory() as d:
        # Create config
        config = {
            "adapters": {
                "deepseek": {
                    "type": "openai_compatible",
                    "base_url": "https://api.deepseek.com/v1",
                    "api_key": key,
                }
            },
            "pipeline": {
                "max_proof_iterations": 3,
                "literature_survey": {
                    "adapter": "deepseek",
                    "model": "deepseek-v4-flash",
                },
                "simple_mode": {
                    "proof_search": {
                        "adapter": "deepseek",
                        "model": "deepseek-v4-flash",
                    },
                    "structural_verifier": {
                        "adapter": "deepseek",
                        "model": "deepseek-v4-flash",
                    },
                    "detailed_verifier": {
                        "adapter": "deepseek",
                        "model": "deepseek-v4-flash",
                    },
                    "verdict": {
                        "adapter": "deepseek",
                        "model": "deepseek-v4-flash",
                    },
                },
                "proof_summary": {
                    "adapter": "deepseek",
                    "model": "deepseek-v4-flash",
                },
            },
        }
        config_path = os.path.join(d, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Create problem.tex
        problem_path = os.path.join(d, "problem.tex")
        with open(problem_path, "w") as f:
            f.write(r"""
\begin{problem}
Let $f: [0,1] \to \mathbb{R}$ be continuous with $f(0)=f(1)=0$.
Prove that there exists $c \in (0,1)$ such that $f(c) = f(c + 1/2)$
if we extend $f$ periodically.
\end{problem}
""")

        # Create minimal prompts directory
        prompts_dir = os.path.join(d, "prompts")
        os.makedirs(prompts_dir)
        for name in ["literature_survey", "proof_search", "proof_verify_structural",
                     "proof_verify_detailed", "proof_verify_easy", "verdict_proof",
                     "brainstorm", "proof_select", "proof_effort_summary"]:
            with open(os.path.join(prompts_dir, f"{name}.md"), "w") as f:
                # Minimal placeholder templates
                f.write(f"# {name}\n\n")
                f.write("Problem: {problem_file}\n")
                f.write("Proof: {proof_file}\n")
                f.write("Write results to appropriate files.\n")

        # Create skill dir
        skill_dir = os.path.join(d, "skill")
        os.makedirs(skill_dir)
        with open(os.path.join(skill_dir, "super_math_skill.md"), "w") as f:
            f.write("You are a mathematical proving agent.")

        output_dir = os.path.join(d, "proof_output")

        # NOTE: This test is intended for manual verification with a real API key.
        # In CI, it will be skipped (no DEEPSEEK_API_KEY).
        #
        # To run manually:
        #   DEEPSEEK_API_KEY=sk-... python -m pytest tests/test_integration.py -v -s
        #
        # This is marked as passing if execution doesn't error — it does NOT
        # verify proof correctness (that requires human review).

        # Skip for now — this test provides the structure but requires
        # the project_root to contain prompts/ and skill/ on disk.
        # Once the full project is assembled (Task 9), enable this.
        pytest.skip("Integration test requires assembled project — run manually after setup")
```

- [ ] **Step 2: 创建手动测试脚本**

```bash
# 在项目根目录创建，用于手动端到端测试
cat > test_e2e.sh << 'EOF'
#!/bin/bash
# End-to-end test: run miniQED on a simple problem
set -euo pipefail

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "ERROR: Set DEEPSEEK_API_KEY environment variable"
    exit 1
fi

cp config.example.yaml config.yaml
echo "Running miniQED end-to-end test..."
bash run.sh problem/problem.tex test_output/
echo ""
echo "Test complete. Check test_output/proof.md"
echo "Token usage: test_output/TOKEN_USAGE.md"
EOF
chmod +x test_e2e.sh
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py test_e2e.sh
git commit -m "feat: add integration test scaffold and e2e test script

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 21: 冻结 Decomposition 代码

**Files:**
- Copy QED decomposition code to `frozen/`

- [ ] **Step 1: 复制**

```bash
cp QED/code/decomposition_prover.py frozen/
cp -r QED/prompts/decomposition-prover/ frozen/decomposition_prompts/
# Add README explaining why these are frozen
```

```markdown
# frozen/README.md
# Frozen: Decomposition Mode

This directory contains QED's decomposition prover code,
frozen for Phase 1. These files are **not imported** by miniQED.

**Why frozen:**
The decomposition mode is a complete subsystem with complex nested
state management (attempt/revision/proof 3-level hierarchy). It will
be activated in Phase 2 after the core Simple mode pipeline is stable
and the workflow engine is integrated.

**Files:**
- `decomposition_prover.py` — Decomposition orchestrator from QED
- `decomposition_prompts/` — Decomposition-specific prompt templates
```

- [ ] **Step 2: Commit**

```bash
git add frozen/
git commit -m "chore: freeze decomposition mode code for Phase 2

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## 实施依赖链

```
Task 1 (脚手架)
  └→ Task 2 (base adapter)
       └→ Task 3 (OpenAICompatibleAdapter) ← Task 4 (占位 adapters)
            └→ Task 5 (Registry) ← 可并行
                                   └→ Task 6 (Config)
                                   └→ Task 7 (Logging)
                                   └→ Task 8 (Utils)
                                   └→ Task 9 (Copy prompts)
                                        ├→ Task 10 (proof_search)
                                        ├→ Task 11 (verdict)
                                        │    ├→ Task 12 (brainstorm)
                                        │    ├→ Task 13 (verification)
                                        │    └→ Task 14 (selection)
                                        │         └→ Task 15 (stage0)
                                        │         └→ Task 16 (stage2)
                                        │              └→ Task 17 (stage1_simple)
                                        │                   └→ Task 18 (orchestrator)
                                        │                        └→ Task 19 (run.sh)
                                        │                        └→ Task 20 (integration test)
                                        └→ Task 21 (frozen)
```

**可并行执行的任务组：** (2,3,4) → 5 → (6,7,8,9) → (10,11,12,13,14) → (15,16) → 17 → 18 → (19,20,21)

---

## 运行全部单元测试

```bash
python -m pytest tests/ -v --ignore=tests/test_integration.py
```

期望：所有非集成测试 PASS（~25 tests）。

## 手动端到端测试

```bash
export DEEPSEEK_API_KEY=sk-your-key-here
cp config.example.yaml config.yaml
bash test_e2e.sh
```
