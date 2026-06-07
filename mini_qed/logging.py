"""Pipeline logging and token tracking -- ported from QED code/pipeline.py."""

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
        # Touch the log file so it exists from initialization
        open(self.log_file, "a", encoding="utf-8").close()
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
    """Accumulates token usage across all agent calls and persists to disk.

    Supports multi-provider tracking: each call can specify a provider
    and model name. Per-provider subtotals are shown in TOKEN_USAGE.md
    when more than one provider is used.
    """

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
            f"**Primary Model:** `{self.model}`  ",
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
