# Recovering from failures: an RL environment built from a stateful API

**Date:** 2026-10-26 · **Hardware:** Apple M4, 16 GB, MLX · **Code:** this directory

## Summary

I built a runnable RL environment (TaskBoard) that targets one capability — recovering from
failures during a multistep API workflow — and tested whether training on it improves that
capability.

Two results, and they point in different directions:

1. **The environment discriminates recovery behaviour cleanly.** Three scripted policies that
   differ *only* in recovery strategy separate monotonically on unseen fault types:
   0.41 → 0.52 → 1.00. A random policy scores 0.00 and a fully-recovering policy scores 1.00,
   so the reward is neither trivially reachable nor unreachable.

2. **Supervised fine-tuning on it produced no measurable learning**, at the model scale I could
   run. All three arms scored at or near zero **including on the no-fault split**, which places
   the bottleneck in base task competence rather than in recovery. The capability claim is
   therefore **unvalidated — not refuted**, and I did not run RLVR, because verifiable-reward RL
   needs a policy that already succeeds sometimes.

The most reusable output is the failure itself: the diagnostic that distinguishes "cannot
recover" from "cannot do the task at all" is a no-fault control split, and running it *before*
training would have saved every training hour spent here. That is now Gate 4 of `SKILL.md`.

---

## 1. Environment

**Target.** A ticket-tracker API (create / update / bulk-update / delete / list), implemented as
an in-process simulator rather than a wrapped real repository. The subset is small enough to
reimplement in ~90 lines and runs ~10k steps/s with no network or credentials. Fidelity to a
specific product was not the goal; transactional semantics with inspectable state was.

**Interface.** `reset(seed) -> obs`, `step(action_json) -> (obs, reward, done, info)`.
One JSON action per step, 24-step limit. Fully deterministic given a seed.

**Tasks.** Five generated templates. Three in training (`create`, `reassign`, `close`), two held
out (`mixed`, `handoff` — the latter has a sequential dependency, where the second half operates
on what the first half produced).

**Reward.** Binary, terminal-state equality:

```python
success = (snapshot(PROJ) == expected) and (snapshot(OTHER) == other_expected)
```

`expected` is computed at reset by the same generator that writes the goal text. The snapshot
compares `(title, assignee, status)` and deliberately **omits ticket IDs**, so delete-and-recreate
is scored identically to update. `OTHER` is a second project the task never mentions, required
to be unchanged — this is what closes the "delete everything and rebuild" exploit.

**Faults.** Injected 2–3 per episode at random positions among mutating calls.

| Fault | Split | Observation | Correct response |
|---|---|---|---|
| `timeout_not_applied` | train | `timeout` | retry |
| `timeout_applied` | train | `timeout` *(identical)* | re-read state; do **not** retry |
| `rate_limit` | train | `429, retry after 3s` | `wait`, then retry |
| `bulk_partial` | held out | `500`, half applied | diff, apply remainder |
| `silent_drop` | held out | `{"ok": true}` | only a terminal verify catches it |
| `external_mutation` | held out | *(none)* | verify catches the reverted change |

The first two are the core design decision: **identical observation, opposite correct action**.
No policy can succeed by pattern-matching the error string.

**Splits.** Disjoint on three independent axes, enforced in code: seeds (`[0,50k)` train vs
`[100000,…)` eval), fault types (3 train / 3 held out), task templates (3 train / 2 held out).
Yields `clean`, `eval_id`, `eval_ood`, `eval_ood_task`.

**Tests.** 11 pytest tests, all passing: determinism; the scripted ladder; split disjointness;
timeout-applied vs not-applied semantics; rate-limit requiring `wait`; and five adversarial
reward probes — `finish` immediately, delete-everything, spam-creates, collateral damage to the
control region, and malformed / wrongly-typed actions.

---

## 2. Scripted baselines

Three policies sharing one `reconcile(current_rows, expected) -> ops` planner, differing only in
recovery:

- **naive** — blind retry up to 3×, never re-reads state
- **reactive** — re-lists state after any error, recomputes remaining work
- **full** — reactive, plus a mandatory verification pass before `finish`

200 eval seeds per split. Scripted policies may read privileged state; learned policies never do.

| Policy | clean | eval_id | eval_ood | eval_ood_task |
|---|---|---|---|---|
| random | 0.00 | 0.00 | 0.00 | 0.00 |
| naive | 1.00 | 0.94 | **0.41** | 0.83 |
| reactive | 1.00 | 1.00 | **0.52** | 0.99 |
| full | 1.00 | 1.00 | **1.00** | 1.00 |

Reading: the task is solvable (`full` = 1.00) and not accidentally easy (`random` = 0.00). The
`eval_ood` column is the capability measurement — a 0.59 spread between blind retry and
verify-before-finish, on fault types never seen in training.

`silent_drop` is what separates `reactive` from `full`: it returns `{"ok": true}`, so error-driven
recovery never triggers and only a terminal verification catches it.

**One design change came out of this phase.** At the original fault density (1–2 per episode)
`naive` scored 0.94 on `eval_id` — almost no headroom for training to demonstrate anything.
Raising density to 2–3 dropped it to 0.94/0.41 and opened the gap. *Measure baseline headroom
before training, not after.*

---

## 3. Model experiments

### 3.1 Zero-shot

Qwen2.5-1.5B-Instruct-4bit, greedy, no adapter: **0.00 on `clean`** (10 seeds), 24.0 mean steps
(i.e. every episode hit the step limit).

Failure mode on inspection: well-formed JSON, correct tool names, but it never calls
`list_tickets`. It acts on imagined state, repeatedly emitting `bulk_update` with
`"ids": []`. Not a formatting problem — a grounding one.

### 3.2 Supervised fine-tuning, three arms

Demonstrations from the scripted policies on training seeds, successful episodes only, with the
demonstrator forced to open with a state read (otherwise demos start from privileged knowledge
the learned policy will not have). 1500 examples per arm.

| Arm | Demos from | Question it answers |
|---|---|---|
| `fault_full` | faulted episodes, `full` policy | the main claim |
| `clean_verify` | fault-free episodes, `full` policy | is verification alone enough? |
| `clean_noverify` | fault-free episodes, `reactive` policy | does fault exposure matter at all? |

LoRA (0.594% of params trainable), Qwen2.5-0.5B-Instruct-4bit, 500 iters, batch 4, lr 2e-4,
prompt-masked, max_seq_len 1536. Validation loss **1.876 → 0.135**.

Success rate, 50 eval seeds per split, greedy decoding:

| Arm | clean | eval_id | eval_ood | eval_ood_task |
|---|---|---|---|---|
| zero-shot (0.5B) | 0.00 | — | — | — |
| `fault_full` | 0.04 | 0.00 | 0.00 | 0.00 |
| `clean_verify` | 0.00 | 0.00 | 0.00 | 0.00 |
| `clean_noverify` | 0.16 | 0.14 | 0.06 | 0.00 |
| *scripted `full`* | *1.00* | *1.00* | *1.00* | *1.00* |

**No arm learned the capability, and the ordering is the reverse of the hypothesis.**

### 3.3 What the numbers mean

Validation loss of 0.135 alongside ~0 task success is the signature of a model that learned
action *syntax* without state *grounding*. A representative `clean_verify` episode:

```
> {"tool": "list_tickets", "args": {"project": "web"}}
< {"ok": true, "tickets": [{"id": 1, "title": "Draft onboarding", "assignee": "ben", ...
> {"tool": "bulk_update", "args": {"project": "web", "titles": ["Draft onboarding",
    "Ship backup", "Ship search", "Ship search", ...          <- hallucinated, wrong schema
< {"ok": false, "error": "invalid_action"}
> {"tool": "list_tickets", "args": {"project": "web"}}
> {"tool": "finish", "args": {}}                               <- gives up, scores 0
```

It reads state, then ignores what it read. Copying IDs out of a long observation into a
structured call is the hard part, and token-level loss barely penalises getting them wrong —
one wrong integer is a small loss penalty and a total task failure.

**The `clean` column is what makes this interpretable.** All arms are near zero *with no faults
present at all*. The bottleneck is base task competence, not recovery — so this measurement
cannot speak to the recovery hypothesis in either direction. `clean_noverify` ranking highest is
consistent with that reading: its demonstrations are the shortest, giving the fewest
opportunities to hallucinate an ID, and its `eval_ood` number (0.06, 3/50) is within noise of
zero.

A scale control (same `fault_full` recipe on 1.5B) was started to test whether the floor is model
capacity; see §6.

### 3.4 RLVR: not run, and why

GRPO forms its advantage from reward variance within a group of rollouts on the same task. With
base success ≈ 0, every group is all-zero, every advantage is 0, and the gradient is 0. RLVR
needs a policy that already succeeds sometimes — roughly 10–40% is the usual working range.

A GRPO implementation is in `train/grpo.py`. It is **untested and should be treated as a
sketch**, not a result.

---

## 4. What I would claim, and what I would not

**Supported by evidence here:**
- The environment is runnable, deterministic, and tested, with a reward that survives five
  adversarial probes.
- It discriminates recovery strategy: 0.41 / 0.52 / 1.00 on held-out fault types, across 200
  seeds, for policies differing only in recovery.
- Low training loss is not evidence of capability. These arms reached 0.135 validation loss with
  ~0 task success.
- A no-fault control split is the cheap diagnostic that separates "cannot recover" from "cannot
  do the task", and it belongs *before* training.
- Baseline headroom must be checked before training: at the original fault density the naive
  baseline scored 0.94 and nothing could have been demonstrated.

**Not supported:**
- That training on this environment improves recovery. Untested at a usable model scale.
- That fault exposure beats clean-demo verification training. The arms are indistinguishable
  at the floor.
- Anything about RLVR on this environment.

---

## 5. Limitations

- **Model scale.** 0.5B for training, 1.5B for zero-shot — well below where agentic tool use
  becomes reliable. This is the dominant limitation and it determines every negative result above.
- **Sample size.** 50 eval seeds per split for LLM arms (200 for scripted). At 50 seeds the 95%
  CI on a 0.06 estimate spans roughly 0.01–0.17.
- **Single seed per arm.** No training-seed replication, so no variance estimate on the SFT runs.
- **Simulated, not wrapped.** TaskBoard reimplements ticket-tracker semantics rather than
  driving a real repository. Transactional structure is faithful; a real API's quirks are not.
- **Greedy decoding at eval.** Temperature > 0 with multiple samples might lift success off the
  floor; untested.
- **The scripted ladder uses privileged state.** It establishes that the environment
  discriminates recovery, not that the task is solvable from observations alone — though the
  observation does contain everything needed.

---

## 6. Status of the scale control

A 1.5B SFT run on the `fault_full` arm (400 iters, batch 2) was launched to test whether the
floor effect is model capacity. If 1.5B clears Gate 4 on `clean`, the three-arm comparison
becomes meaningful and should be rerun at that scale. If it does not, the next lever is task
simplification (fewer tickets, shorter episodes) rather than more training.

Result pending at time of writing — see `results/sft_fault_full_15b.log` and
`results/llm_fault_full_15b.json`.

---

## 7. Reproducing

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q tests        # 11 tests
.venv/bin/python run_eval.py               # scripted baselines, ~20s
.venv/bin/python -m train.make_sft         # demo generation
bash train/run_sft.sh                      # ~25 min/arm on M4
.venv/bin/python eval_llm.py fault_full adapters/fault_full 50
```

Raw results: `results/baselines.json`, `results/llm_*.json`, `results/sft_*.log`.
