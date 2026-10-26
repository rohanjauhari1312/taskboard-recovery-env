# CHECKS-REPORT — Measured, Not Taught.

Written before the first slate compiled, per PROOF GATE.

## Beat classification

8 SHOW / 0 justified-HOLD / 0 PUNT-flagged

| Beat | Lane | Class | On-screen artifact |
|---|---|---|---|
| B00 | BOOKEND | CARD (UI is the subject) | `ClaudeComposerAsk` — ask lands answered |
| B01 | BOOKEND | SHOW | `BrutalistHesitantWriter` — the correction is the reel's misconception |
| B02 | BODY | SHOW | `RecoveryTaskboard` — goal → action → API, reward check |
| B03 | BODY | SHOW | `RecoveryTwinFault` — twin errors diverge |
| B04 | BODY | SHOW | `RecoveryRewardGuard` — exploit struck through, guard lands |
| B05 | BOOKEND | CARD (ask micro-beat) | `ClaudeComposerAsk` — the chart's actual generation prompt |
| B06 | BODY | SHOW | `RecoveryLadder` — bars grow to 0.41 / 0.52 / 1.00 |
| B07 | BODY | SHOW | `RecoveryFloor` phase=loss — curve falls, success flat |
| B08 | BODY | SHOW | `RecoveryFloor` phase=clean — the no-fault column |
| BVDT | BOOKEND | CARD (verdict artifact) | `ClaudeVerdictArtifact` |
| BHTF | BOOKEND | CARD (handoff) | `ClaudeComposerAsk` |
| BOUT | BOOKEND | CARD (outro lock) | `ClaudeTitleOutro` |

No body beat is a bare CARD. Every claim beat names its artifact in `shot.show`.

## Teaching arc

```
FRAMEWORK ✓   B02 — what the environment is, before any result
WORKED EXAMPLE ✓   B03 (the twin fault in full) and B06 (the ladder measured)
FALSIFIABILITY ✓   B07/B08 — the result that contradicts the hypothesis is
                   shown, and B08 states what would distinguish the two causes
SCAFFOLDED TASK ✓  BHTF — the viewer designs a control condition for their own
                   repo; the narration reads it aloud and says what to look for
BOOKENDS ✓         B00 cold open · B01 BLUF · BVDT verdict · BHTF handoff · BOUT outro
NO-SOURCE-NO-VERDICT ✓  every on-screen number traces to results/*.json or
                        sft_fault_full.log; sources listed in metadata.sources
```

## Number provenance

| On screen | Value | Source |
|---|---|---|
| random / naive / reactive / full, held-out faults | 0.00 / 0.41 / 0.52 / 1.00 | `results/baselines.json`, 200 seeds |
| val loss start → end | 1.876 → 0.135 | `results/sft_fault_full.log` |
| SFT success, no-fault split | 0.04 / 0.00 / 0.16 | `results/llm_*.json`, 50 seeds |
| scripted policy, no-fault split | 1.00 | `results/baselines.json` |

The loss curve's intermediate points are a monotone interpolation between the two
measured endpoints (iter 1 and iter 500); only the endpoints are labelled on
screen. Logged here because the shape between them is drawn, not measured.

## ILLUSTRATE LAW

Claude UI appears at B00, the B05 ask micro-beat, BVDT, BHTF, BOUT — the cold
open, one ask→result receipt, verdict, handoff, outro. All six body beats are
purpose-built C3 concept illustrations. No two consecutive beats share a visual
scheme except B07/B08, which are deliberate phases of one figure and differ in
form (line plot vs. bar column).

## GATE L — library-first

Searched 697 registered scenes before authoring. Spine components found and
reused: `ClaudeComposerAsk`, `BrutalistHesitantWriter`, `ClaudeVerdictArtifact`,
`ClaudeTitleOutro`. Data scenes in the Claude palette were a genuine PUNT —
`BarChart` carries the Codex teal/slate palette, and every Claude-skin data
figure on disk (`TeachClaudeFig*`, `SleeperAgentsResult`, …) is hardcoded to its
own paper with only `sparkLine` as a prop. Built `RecoveryEnv.tsx` (5 scenes)
as the design-card response; registered and indexed so the next reel finds them.
