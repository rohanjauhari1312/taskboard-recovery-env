import json, sys, time
from agents.llm import LLMPolicy
from run_eval import evaluate

import os
M = os.environ.get("EVAL_MODEL", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
name, adapter = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "none" else None)
n = int(sys.argv[3]) if len(sys.argv) > 3 else 60
p = LLMPolicy(M, adapter=adapter)
out = {}
t = time.time()
for split in ["clean", "eval_id", "eval_ood", "eval_ood_task"]:
    out[split] = evaluate(p, split, range(100000, 100000 + n))
    print(name, split, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[split].items()}, round(time.time() - t), "s", flush=True)
json.dump(out, open(f"results/llm_{name}.json", "w"), indent=1)
