import json, re
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

SYSTEM = "You are an automation agent operating a ticket API. Output exactly one JSON action and nothing else."


class LLMPolicy:
    def __init__(self, model_path, adapter=None, temp=0.0, max_tokens=120):
        self.model, self.tok = load(model_path, adapter_path=adapter)
        self.sampler = make_sampler(temp=temp)
        self.max_tokens = max_tokens

    def prompt(self, obs):
        return self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": obs}],
            add_generation_prompt=True, tokenize=False)

    def act(self, obs):
        out = generate(self.model, self.tok, prompt=self.prompt(obs), max_tokens=self.max_tokens, sampler=self.sampler)
        m = re.search(r"\{.*\}", out, re.S)
        return m.group(0) if m else out
