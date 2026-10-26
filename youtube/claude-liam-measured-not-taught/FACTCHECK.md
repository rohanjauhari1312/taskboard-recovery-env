# FACTCHECK — Measured, Not Taught.

Verified 2026-10-26 against the project's own run artifacts. The source is the
build itself, so every claim is checkable by re-running the commands in
`recovery-env/README.md`. DOUBLE-CHECK LAW: claims are stated at the strength
the evidence supports and no further.

| # | Claim (as narrated) | Verdict | Source | Fix applied |
|---|---|---|---|---|
| 1 | "Reward is binary and it lands once, at the end" | TRUE | `taskboard/env.py` `_success()` — terminal snapshot equality, no shaping | — |
| 2 | "Blind retry gets forty-one percent" (held-out faults) | TRUE | `results/baselines.json` — naive, eval_ood, 200 seeds = 0.41 | — |
| 3 | "Re-reading state after an error buys eleven points" | TRUE | 0.52 − 0.41 = 0.11, same file, same split | — |
| 4 | "Verifying before you finish takes it to one hundred" | TRUE | full, eval_ood = 1.00 | — |
| 5 | "Validation loss falls from 1.88 to 0.135" | TRUE | `results/sft_fault_full.log` — iter 1 val 1.876, iter 500 val 0.135 | narration says "one point eight eight"; 1.876 rounded for speech, exact value on screen in SKILL/REPORT |
| 6 | "Task success never leaves the floor" | TRUE | `results/llm_fault_full.json` — 0.00 on eval_id, eval_ood, eval_ood_task | — |
| 7 | "Four percent, zero, sixteen percent" on the no-fault split | TRUE | `llm_fault_full` 0.04, `llm_clean_verify` 0.00, `llm_clean_noverify` 0.16 | — |
| 8 | "The scripted policy scores one hundred on that same split" | TRUE | `results/baselines.json` — full, clean = 1.00 | — |
| 9 | "The model learned the shape of a tool call and hallucinated the ticket IDs" | SUPPORTED, interpretive | Episode transcript in `REPORT.md §3.3`: model lists tickets, then emits `bulk_update` with invented titles and the wrong schema | stated as the reading of the transcript, not as a measured quantity |
| 10 | "I never ran the reinforcement learning stage… the gradient is zero too" | TRUE as stated | `train/grpo.py` exists but was never run; the zero-advantage argument is standard GRPO mechanics, labelled `[H]` in `SKILL.md` | narration says "because with success at zero every rollout group scores the same" — mechanism, not a measured result |
| 11 | "Delete every ticket, rebuild them… score a perfect one point zero" | TRUE, historical | The exploit was live before the control region was added; `tests/test_env.py::test_delete_everything_and_spam_fail` now pins it at 0 | narration is past tense ("the reward was broken") |
| 12 | "Two timeouts. Byte for byte the same error." | TRUE | `env.py` returns `{"ok": false, "error": "timeout", "detail": "request timed out"}` for both `timeout_applied` and `timeout_not_applied` | — |
| 13 | "Fault types it never saw in training" | TRUE | `TRAIN_FAULTS` and `HELDOUT_FAULTS` are disjoint lists in `taskboard/env.py`; eval_ood draws only from the held-out list | — |
| 14 | "11 tests pass" (verdict card) | TRUE | `pytest -q tests` → 11 passed, re-run against final env code | — |

## De-sensationalised

- The reel never says the environment "doesn't work" or "failed". It says the
  capability claim is **unproven**, which is what a floor effect supports.
- The reel never claims fault exposure beats clean-demo training. All three arms
  sit at the floor and are indistinguishable there.
- `clean_noverify` scoring highest (0.16) is **not** narrated as a finding; at
  50 seeds and 3/50 on eval_ood it is within noise of zero.

## Dated material stripped

Model identifiers stay in the written report, not on screen, except the one
eyebrow line naming the training configuration (`LORA SFT · QWEN2.5-0.5B-4BIT ·
500 ITERATIONS`) — that one is load-bearing, because the entire conclusion is
scoped to that model scale and removing it would overstate the result.

## Not claimed anywhere in the reel

- That training on this environment improves recovery.
- Any RLVR result.
- Any generalisation beyond 0.5B / 50 eval seeds / one training seed per arm.
