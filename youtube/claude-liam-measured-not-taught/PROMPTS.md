# PROMPTS — Measured, Not Taught.

No open slots: every beat renders from a registered Remotion composition, so
there are no request cards for a human to fill.

Recorded for reproducibility — the generation prompt shown on screen at B05 is
the real one, and it is the prompt that produced the B06 figure's data:

```
B05 → B06 (ask → result receipt)
Run the three scripted policies over 200 eval seeds on the held-out fault
types, and chart success rate by policy.
```

Executed as:

```bash
.venv/bin/python run_eval.py     # writes results/baselines.json
```

The B06 composition reads those four measured values as literals
(`random 0.00 · naive 0.41 · reactive 0.52 · full 1.00`); see FACTCHECK.md rows
2–4 for provenance.
