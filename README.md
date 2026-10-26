# TaskBoard recovery environment

RL environment for one capability: recovering from failures in a multistep API workflow.
Apple Silicon (MLX) only for the LLM parts; the environment itself is pure Python.

```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q tests          # environment tests
.venv/bin/python run_eval.py                 # scripted baselines -> results/baselines.json
.venv/bin/python -m train.make_sft           # SFT demos for the three arms -> data/
bash train/run_sft.sh                        # LoRA SFT (Qwen2.5-0.5B-Instruct-4bit), 500 iters/arm
.venv/bin/python eval_llm.py <name> <adapter|none> 50   # -> results/llm_<name>.json
```

Layout: `taskboard/` service + env, `agents/` scripted and LLM policies, `train/` SFT and an untested GRPO script.
See `REPORT.md` for results and `SKILL.md` for the playbook.
