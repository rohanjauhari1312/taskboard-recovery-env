---
name: build-rl-environment
description: Turn a software repository or stateful API into a validated RL environment that trains and measures one agent capability, with verifiable rewards, injected failures, and held-out evaluation. Use when asked to create an RL environment or training task from a codebase, or to check whether an existing environment produces measurable learning.
---

# Building an RL environment from a codebase

The goal is an environment that **measurably trains one capability**. Most environments that
look fine never demonstrate learning, so the process below front-loads the checks that kill
bad designs cheaply.

Every claim here is tagged:
- **[V]** validated by the TaskBoard build documented in `REPORT.md`
- **[H]** hypothesis, consistent with that build but not tested

Work through the phases in order. Each ends in a gate. **A failed gate means stop and fix,
not proceed.** The expensive phases are last on purpose.

---

## Phase 0 — Name the capability before the repo

Write one sentence: *the agent must be able to ___, and I will know it can because ___.*

A capability is trainable here only if it has an **observable behavioural signature** that a
competent policy shows and an incompetent one does not. "Recovering from failures" works
because recovery requires re-reading state after an error; "writing good code" does not,
because there is no cheap verifier.

Then find the repo. Prefer a target where:
- State is **inspectable and comparable** — you can snapshot it and diff two snapshots. **[V]**
- Operations are **transactional-ish**: a call either changed state or did not, and you can tell. **[V]**
- A task needs **3–15 operations**. Below 3 there is no multistep structure; above ~15, a weak
  model never reaches the end and every reward is zero. **[V]**
- Running it needs no network, no credentials, and under a second per step. You will run
  hundreds of thousands of steps. **[V]**

> **Do not wrap the real repo first.** Reimplement the subset of its semantics you need as an
> in-process simulator, then validate the simulator against the real thing later. Real repos
> bring install friction, nondeterminism and slow steps — all of which cost you far more than
> the fidelity buys. **[V: the TaskBoard service is ~90 lines and runs ~10k steps/s.]**

**Gate 0:** you can state the capability, and name the specific observation a recovering agent
makes that a non-recovering one does not.

---

## Phase 1 — Tasks, state, and a reward that cannot be gamed

### Reward
Score **terminal state equality against a computed expected state**, not the trajectory.

```python
def success(self):
    return (self.svc.snapshot(PROJ) == self.expected
            and self.svc.snapshot(OTHER) == self.other_expected)
```

Rules that matter:
- **Compute `expected` at reset**, from the same generator that wrote the goal text. Never
  hand-write it per task, and never derive it from what the agent did. **[V]**
- **Snapshot excludes identity you do not care about.** TaskBoard compares
  `(title, assignee, status)` and deliberately omits ticket IDs, so an agent that
  deletes-and-recreates a ticket is scored the same as one that updates it. If you include
  incidental fields, you punish valid strategies. **[V]**
- **Include an untouched control region.** TaskBoard has a second project the task never
  mentions and the reward requires it unchanged. Without this, "delete everything and rebuild"
  scores 1.0. **[V — this exploit was live until the control region was added.]**
- **No partial credit, no shaping, in the reward you report.** Shaped rewards get optimised
  instead of the task. If you want shaping for training, keep it strictly separate from the
  reported metric. **[H]**

### Actions
One JSON object per step, validated in layers, each layer returning a distinct error:
parse → unknown tool → missing args → **wrong types** → semantic preconditions.

Type validation is not optional pedantry. A weak model emits `{"id": "1"}` and
`{"titles": [1]}` constantly; without a type layer these reach your service and either crash
it or corrupt state in ways that look like environment bugs. **[V — this crashed a full eval
run partway through.]**

### Failures
Inject faults that are **ambiguous from the error message alone**. The agent must inspect
state to know what happened. TaskBoard's set:

| Fault | What the agent sees | Why it discriminates |
|---|---|---|
| `timeout_not_applied` | `timeout` | retry is correct |
| `timeout_applied` | `timeout` *(identical)* | retry creates a duplicate |
| `rate_limit` | `429, retry after 3s` | must `wait`, not retry immediately |
| `bulk_partial` | `500` after half applied | must diff, not redo the whole batch |
| `silent_drop` | `{"ok": true}` | **no error at all** — only a final verify catches it |
| `external_mutation` | nothing | state changes under the agent between steps |

The first two are the core of the design: **identical observation, opposite correct action.**
A policy cannot pattern-match the error string; it has to look. **[V]**

`silent_drop` is the only fault that forces a *terminal* verification habit rather than a
*reactive* one, and it is the fault that most separates the scripted policies. **[V]**

**Gate 1 — adversarial reward audit.** Write these as tests before any training:
- `finish` immediately → 0
- delete everything → 0
- spam the most common operation → 0
- damage the control region while otherwise solving → 0
- malformed and wrong-typed actions → rejected, no state change, step consumed

If any scores above zero, the reward is broken. **[V — all five are in `tests/test_env.py`.]**

---

## Phase 2 — The scripted ladder (do this before any ML)

Write three scripted policies that **differ only in recovery behaviour**:

1. **naive** — blind retry, up to N times, no state re-read
2. **reactive** — re-list state after any error, recompute remaining work
3. **full** — reactive, *plus* a mandatory verify pass before `finish`

All three share one `reconcile(current_rows, expected) -> ops` function. Keeping the planner
identical is what makes the comparison about recovery and nothing else. **[V]**

These policies may read privileged state (`env.expected`); learned policies never may.

Measured ladder, 200 seeds per split:

| Policy | no faults | train faults | held-out faults |
|---|---|---|---|
| random | 0.00 | 0.00 | 0.00 |
| naive | 1.00 | 0.94 | 0.41 |
| reactive | 1.00 | 1.00 | 0.52 |
| full | 1.00 | 1.00 | 1.00 |

**Gate 2, four conditions:**
- `full` ≈ 1.0 everywhere — the task is solvable and the reward is reachable **[V]**
- `naive` < `reactive` < `full` on held-out faults — recovery is what the gap measures **[V]**
- `random` ≈ 0 — the reward is not accidentally easy **[V]**
- **`naive` is well below 1.0 on the faults you train with** — otherwise there is no headroom
  to learn **[V — this gate failed first: `naive` scored 0.94 on train faults, so training
  could show almost nothing. Fixed by raising fault density from 1–2 to 2–3 per episode.]**

This phase costs minutes and catches design faults that would otherwise surface after hours
of training as an ambiguous flat line. **It is the highest-value step in the playbook.**

---

## Phase 3 — Splits, fixed before training

Three independent axes, each disjoint by construction, not by filtering:

- **seeds** — train `[0, 50k)`, eval `[100000, …)`
- **fault types** — `timeout_not_applied`, `timeout_applied`, `rate_limit` in training;
  `bulk_partial`, `silent_drop`, `external_mutation` **never seen until eval**
- **task templates** — 3 in training, 2 held out

This gives four eval splits: `clean`, `eval_id`, `eval_ood` (unseen faults),
`eval_ood_task` (unseen task shapes). `eval_ood` is the one that answers the research
question; `clean` is the diagnostic that tells you whether a zero means "cannot recover" or
"cannot do the task at all". **[V — `clean` is what identified the floor effect in Phase 5.]**

**Carry the held-out faults in a separate list in code**, so a later edit cannot leak them
into training by accident. **[V]**

---

## Phase 4 — Baseline the model before you train it

Run the untrained model on `clean`. This is cheap and it is the gate that matters most.

**Gate 4: the untrained model must score meaningfully above zero on `clean`.**

If it scores 0, you cannot measure recovery with that model, because the task itself is out of
reach. Your options are to use a larger model, simplify the task, or shorten the episode — but
**do not start training**. The result will be flat and you will not know why. **[V — this is
exactly what happened below.]**

Measured: Qwen2.5-1.5B-Instruct-4bit zero-shot scored **0/10 on `clean`**. It emitted
well-formed JSON but never called `list_tickets`, so it acted on imagined state and ran to the
step limit every episode.

---

## Phase 5 — Train, with controls that can falsify your claim

SFT on scripted demonstrations first; it is far cheaper than RL and it tells you whether the
format and the task are learnable at all.

**Demonstration generation matters more than it looks.** Keep only successful episodes, and
force the demonstrator to open with a state read (`lead_list=True`) — otherwise every demo
starts from privileged knowledge the learned policy will not have, and you teach it to act
blind. **[V]**

Run **three arms**, not one:

| Arm | Demos from | Tests |
|---|---|---|
| `fault_full` | faulted episodes, verifying policy | the full claim |
| `clean_verify` | fault-free episodes, verifying policy | does verification alone suffice? |
| `clean_noverify` | fault-free episodes, reactive policy | does fault exposure matter at all? |

If `fault_full` beats `clean_*` on `eval_ood`, fault exposure is doing the work. If
`clean_verify` matches it, you have learned that the verification *habit* is the whole
capability and the faults are decoration — a cheaper environment would do. **[H — untested;
the measurement below could not resolve it.]**

### What actually happened

LoRA SFT, Qwen2.5-0.5B-Instruct-4bit, 500 iters per arm, 1500 examples. Validation loss fell
from 1.88 to **0.135** — the model learned the demonstration distribution well.

Success rate, 50 eval seeds per split:

| Arm | clean | eval_id | eval_ood | eval_ood_task |
|---|---|---|---|---|
| `fault_full` | 0.04 | 0.00 | 0.00 | — |
| `clean_verify` | 0.00 | 0.00 | 0.00 | 0.00 |
| `clean_noverify` | 0.16 | 0.14 | 0.06 | 0.00 |

**No arm learned the capability, and the ranking is the reverse of the hypothesis.** Low
training loss with near-zero task success is the signature of a model that has learned action
*syntax* without state *grounding*: it reproduces the shape of a `bulk_update` while
hallucinating the IDs, because copying IDs out of a long observation is the hard part and the
loss barely penalises getting them wrong.

The `clean` column is what makes this interpretable. All arms are near zero **with no faults
present at all**, so this is a floor effect in base task competence, not a failure to recover.
The environment cannot measure recovery at this model scale. `clean_noverify` ranking highest
is consistent with that reading: its demos are the shortest, so it has the fewest chances to
hallucinate an ID.

**This is a negative result about the model, not about the environment** — the Phase 2 ladder
shows the environment discriminates recovery cleanly when the policy can execute at all. The
honest conclusion is that the capability claim is **unvalidated**, not refuted.

### RLVR
Not run. With base success near zero on every split, GRPO has no non-zero rewards to form an
advantage from and would produce no gradient signal. **Verifiable-reward RL requires a policy
that already succeeds sometimes** — roughly 10–40% is the usual working range. Getting past
Gate 4 is a precondition for RL, not an optional nicety. **[H — standard practice, not
measured here.]**

---

## Phase 6 — Is it ready to scale?

Ship to a fleet only when **all** hold:

1. Gates 0–4 pass, Gate 4 included. **[V — the one most likely to be skipped.]**
2. Scripted ladder shows a monotone, well-separated gap on held-out faults. **[V]**
3. A trained policy beats its matched control on `eval_ood` by more than seed noise
   (≥3 seeds per arm; report the spread). **[H — not reached here.]**
4. Held-out faults and tasks are disjoint in code, not by convention. **[V]**
5. The adversarial reward tests pass and are in CI. **[V]**

Failing 3 while passing the rest means **a validated environment with an unvalidated capability
claim**. That is a legitimate artefact — it is reusable as an eval — but do not describe it as
a training environment until a model clears Gate 4.

---

## Running this with Optimus

Fits the ArkenOS crew roles; division of labour is **[H]** until run at scale.

| Phase | Lead | Reviewer |
|---|---|---|
| 0 — capability, repo triage | Quintus | Megatron |
| 1 — env, reward, faults | Ratchet | Megatron |
| Gate 1 — adversarial reward audit | Starscream | — |
| 2 — scripted ladder | Ratchet | Starscream |
| Gate 2 / Gate 4 | Starscream | — |
| 4–5 — training, controls | Ratchet | Soundwave |
| 5 — analysis | Soundwave | Megatron |
| 6 — scale decision | Starscream | Megatron |

Megatron's standing brief at Gate 1: *find the policy that scores 1.0 without doing the task.*
Adversarial review is the right shape for a critic role and it is where reward bugs die. **[H]**

**Standardise** (same code every repo): the step/reset loop, JSON action parsing and the
layered validator, the fault-injection wrapper, split management, the scripted ladder harness,
the adversarial reward tests, SFT demo generation, and the eval matrix. **[V — all of these
are repo-independent in this build.]**

**Keep human/agent judgement** (per repo): which capability, which operations to simulate,
what the snapshot compares and what it must ignore, which faults are realistic for *this*
API, and the control region. These encode domain semantics and are where a wrong choice
silently invalidates the reward. **[V — the snapshot-field and control-region decisions were
both load-bearing and neither was mechanical.]**

**Per-repo cost once the harness exists:** roughly a day of work, dominated by Phase 1 and
Phase 2. Training is the long pole in wall-clock but needs little attention. **[H]**

---

## Failure modes, ranked by how much time they cost

1. **Training before Gate 4.** Produces a flat line that could be task design, reward,
   faults, or training — and you cannot tell which. The `clean` split is the cheap
   disambiguator; run it first. **[V]**
2. **No headroom.** If the naive baseline already scores ~0.95 on your training faults, nothing
   can demonstrate improvement. Check the ladder, then raise fault density. **[V]**
3. **A reward with an exploit.** Always recoverable in principle, but it invalidates every run
   made before you found it. The control region is the single highest-value defence. **[V]**
4. **Demos that leak privileged state.** The policy learns to act without looking, which is
   precisely the behaviour you are trying to train away. **[V]**
5. **Missing type validation.** Looks like environment flakiness, surfaces mid-eval. **[V]**
6. **Reading low training loss as learning.** Loss measures imitation of demo tokens; only the
   environment measures capability. Report both, and when they disagree, believe the
   environment. **[V]**
