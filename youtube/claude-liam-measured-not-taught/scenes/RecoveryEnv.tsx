import React from 'react';
import { AbsoluteFill, useCurrentFrame, useVideoConfig, spring, interpolate } from 'remotion';
import { z } from 'zod';
import { CLAUDE, CLAUDE_FONT } from '../tokens/claude';

/**
 * RecoveryEnv — figures for "Measured, Not Taught." (claude-liam)
 * Source: the TaskBoard recovery environment, recovery-env/REPORT.md (2026-10-26).
 *
 * Every number on screen is measured, not illustrative:
 *   scripted ladder, eval_ood, 200 seeds   random 0.00 · naive 0.41 · reactive 0.52 · full 1.00
 *   LoRA SFT Qwen2.5-0.5B-4bit, 500 iters  val loss 1.876 -> 0.135
 *   SFT success on the no-fault split      fault_full 0.04 · clean_verify 0.00 · clean_noverify 0.16
 *
 * Per CLAUDE-BRAND.md: cream ground, warm ink, exactly ONE terracotta moment
 * per beat (named in each component's header).
 *
 * Each component reflows for portrait (the *916 compositions) when height > width.
 * Type is sized off the short side (U), so the landscape layout is unchanged.
 */

const SERIF = CLAUDE_FONT.serif;
const SANS = CLAUDE_FONT.ui;
const MONO = CLAUDE_FONT.mono;
const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const ease = { damping: 30, stiffness: 110, mass: 0.8 };
const grow = { damping: 26, stiffness: 80, mass: 1.0 };

const useStage = () => {
  const cfg = useVideoConfig();
  return { ...cfg, U: Math.min(cfg.width, cfg.height), P: cfg.height > cfg.width };
};

/** Shared chrome: eyebrow, title, spark line. Keeps every figure on one grid. */
const Frame: React.FC<{
  eyebrow: string;
  title: string;
  sparkLine?: string;
  children: React.ReactNode;
}> = ({ eyebrow, title, sparkLine, children }) => {
  const frame = useCurrentFrame();
  const { fps, width, height, U, P } = useStage();
  const inAnim = spring({ frame, fps, config: ease });
  const sparkIn = spring({ frame: frame - 50, fps, config: ease });
  const PAD_X = width * 0.075;

  return (
    <AbsoluteFill style={{ background: CLAUDE.PAGE, overflow: 'hidden' }}>
      <div style={{
        position: 'absolute', left: PAD_X, top: height * (P ? 0.07 : 0.075),
        fontFamily: SANS, fontSize: U * (P ? 0.026 : 0.0165), fontWeight: 700,
        letterSpacing: 3.2, textTransform: 'uppercase' as const, lineHeight: 1.5,
        color: CLAUDE.INK_SOFT, opacity: clamp(inAnim, 0, 1),
        maxWidth: width - PAD_X * 2,
      }}>{eyebrow}</div>

      <div style={{
        position: 'absolute', left: PAD_X, top: height * (P ? 0.125 : 0.115),
        fontFamily: SERIF, fontSize: U * (P ? 0.078 : 0.052), fontWeight: 600,
        color: CLAUDE.INK, letterSpacing: '-0.015em', lineHeight: 1.1,
        opacity: clamp(inAnim, 0, 1), transform: `translateY(${(1 - inAnim) * 12}px)`,
        maxWidth: width * (P ? 0.85 : 0.84),
      }}>{title}</div>

      {children}

      {sparkLine ? (
        <div style={{
          position: 'absolute', left: PAD_X, bottom: height * (P ? 0.07 : 0.062),
          display: 'flex', alignItems: 'baseline', gap: width * (P ? 0.02 : 0.011),
          opacity: clamp(sparkIn, 0, 1),
          transform: `translateY(${(1 - sparkIn) * 8}px)`,
          maxWidth: width - PAD_X * 2,
        }}>
          <span style={{ fontFamily: SERIF, fontSize: U * (P ? 0.056 : 0.036), color: CLAUDE.SPARK }}>✳</span>
          <span style={{ fontFamily: SERIF, fontSize: U * (P ? 0.05 : 0.032), color: CLAUDE.INK }}>{sparkLine}</span>
        </div>
      ) : null}
    </AbsoluteFill>
  );
};

/* ───────────────────────── B02 — what the environment is ───────────────────── */
/** Terracotta moment: the reward check at the end of the loop. */

export const recoveryTaskboardSchema = z.object({
  sparkLine: z.string().default('State in, state out.'),
});
export type RecoveryTaskboardProps = z.infer<typeof recoveryTaskboardSchema>;

const STAGES = [
  { k: 'GOAL', v: 'Reassign every open\nticket from ben to dia', mono: false },
  { k: 'ACTION', v: '{"tool": "bulk_update",\n "args": {"ids": [1, 2]}}', mono: true },
  { k: 'API', v: 'ticket service\nmutates, or fails', mono: false },
];

export const RecoveryTaskboard: React.FC<RecoveryTaskboardProps> = ({ sparkLine }) => {
  const frame = useCurrentFrame();
  const { fps, width, height, U, P } = useStage();
  const PAD_X = width * 0.075;
  const GAP = P ? height * 0.05 : width * 0.075;
  const BOX_W = P ? width - PAD_X * 2 : (width - PAD_X * 2 - GAP * 2) / 3;
  const TOP = height * (P ? 0.27 : 0.30);
  const BOX_H = height * (P ? 0.112 : 0.24);
  const rewardIn = spring({ frame: frame - 62, fps, config: ease });
  const k = P ? 1.3 : 1;

  return (
    <Frame
      eyebrow="TASKBOARD · THE ENVIRONMENT UNDER TEST"
      title="One task, many calls, one verdict"
      sparkLine={sparkLine}
    >
      {STAGES.map((s, i) => {
        const a = spring({ frame: frame - 12 - i * 14, fps, config: ease });
        const left = P ? PAD_X : PAD_X + i * (BOX_W + GAP);
        const top = P ? TOP + i * (BOX_H + GAP) : TOP;
        return (
          <React.Fragment key={s.k}>
            <div style={{
              position: 'absolute', left, top, width: BOX_W, height: BOX_H,
              background: CLAUDE.CARD, border: `1px solid ${CLAUDE.BORDER}`,
              borderRadius: 14, padding: U * 0.028 * k, boxSizing: 'border-box',
              opacity: clamp(a, 0, 1), transform: `translateY(${(1 - a) * 16}px)`,
            }}>
              <div style={{
                fontFamily: SANS, fontSize: U * 0.016 * k, fontWeight: 700,
                letterSpacing: 2.6, color: CLAUDE.INK_SOFT, marginBottom: U * (P ? 0.018 : 0.022),
              }}>{s.k}</div>
              <div style={{
                fontFamily: s.mono ? MONO : SERIF,
                fontSize: (s.mono ? U * 0.0235 : U * 0.030) * k,
                color: CLAUDE.INK, lineHeight: 1.45, whiteSpace: 'pre-line' as const,
              }}>{s.v}</div>
            </div>
            {i < 2 ? (
              <div style={{
                position: 'absolute',
                left: P ? width / 2 - U * 0.02 : left + BOX_W + width * 0.022,
                top: P ? top + BOX_H + GAP * 0.08 : TOP + BOX_H / 2 - U * 0.022,
                fontFamily: SANS, fontSize: U * 0.040 * k, color: CLAUDE.GHOST,
                opacity: clamp(spring({ frame: frame - 22 - i * 14, fps, config: ease }), 0, 1),
              }}>{P ? '↓' : '→'}</div>
            ) : null}
          </React.Fragment>
        );
      })}

      {/* the reward check — the one terracotta moment */}
      <div style={{
        position: 'absolute', left: PAD_X,
        top: P ? TOP + 3 * BOX_H + 2 * GAP + height * 0.04 : TOP + BOX_H + height * 0.065,
        width: width - PAD_X * 2,
        display: 'flex', flexDirection: P ? 'column' as const : 'row' as const,
        alignItems: P ? 'flex-start' : 'center', gap: P ? height * 0.02 : width * 0.018,
        opacity: clamp(rewardIn, 0, 1), transform: `translateY(${(1 - rewardIn) * 12}px)`,
      }}>
        <div style={{
          fontFamily: MONO, fontSize: U * 0.0285 * k, color: CLAUDE.SPARK,
          border: `1.5px solid ${CLAUDE.SPARK}`, borderRadius: 10,
          padding: `${U * 0.016 * k}px ${(P ? U : width) * 0.016 * k}px`, whiteSpace: 'nowrap' as const,
        }}>snapshot(web) == expected</div>
        <div style={{
          fontFamily: SERIF, fontSize: U * 0.030 * k, color: CLAUDE.INK_SOFT,
          maxWidth: P ? width - PAD_X * 2 : width * 0.56, lineHeight: 1.4,
        }}>
          scored once, at the end — never on the path taken
        </div>
      </div>
    </Frame>
  );
};

/* ──────────────────── B03 — the twin fault (the mechanism) ─────────────────── */
/** Terracotta moment: the duplicate the blind retry creates. */

export const recoveryTwinFaultSchema = z.object({
  sparkLine: z.string().default('Same error. Opposite move.'),
});
export type RecoveryTwinFaultProps = z.infer<typeof recoveryTwinFaultSchema>;

export const RecoveryTwinFault: React.FC<RecoveryTwinFaultProps> = ({ sparkLine }) => {
  const frame = useCurrentFrame();
  const { fps, width, height, U, P } = useStage();
  const PAD_X = width * 0.075;
  const COL_W = P ? width - PAD_X * 2 : (width - PAD_X * 2 - width * 0.05) / 2;
  const TOP = height * (P ? 0.27 : 0.30);
  const k = P ? 1.3 : 1;

  const split = spring({ frame: frame - 44, fps, config: ease });

  const SIDES = [
    { tag: 'the call never landed', act: 'retry', verdict: 'correct', accent: false },
    { tag: 'the call landed, then timed out', act: 'retry', verdict: 'a duplicate ticket', accent: true },
  ];

  return (
    <Frame
      eyebrow="FAULT DESIGN · THE PAIR THAT CARRIES THE CAPABILITY"
      title="Two failures the agent cannot tell apart"
      sparkLine={sparkLine}
    >
      {SIDES.map((s, i) => {
        const cardIn = spring({ frame: frame - 14 - i * 10, fps, config: ease });
        const left = P ? PAD_X : PAD_X + i * (COL_W + width * 0.05);
        const top = P ? TOP + i * height * 0.30 : TOP;
        const drift = P ? 0 : interpolate(split, [0, 1], [0, i === 0 ? -14 : 14]);
        return (
          <div key={i} style={{
            position: 'absolute', left: left + drift, top, width: COL_W,
            opacity: clamp(cardIn, 0, 1), transform: `translateY(${(1 - cardIn) * 16}px)`,
          }}>
            {/* the identical observation */}
            <div style={{
              background: CLAUDE.CARD, border: `1px solid ${CLAUDE.BORDER}`,
              borderRadius: 12, padding: U * 0.030 * k,
              fontFamily: MONO, fontSize: U * 0.028 * k, color: CLAUDE.INK,
            }}>{'{"error": "timeout"}'}</div>

            <div style={{
              marginTop: U * 0.026 * k, fontFamily: SERIF,
              fontSize: U * 0.029 * k, color: CLAUDE.INK_SOFT, lineHeight: 1.4,
            }}>{s.tag}</div>

            <div style={{
              marginTop: U * 0.030 * k, opacity: clamp(split, 0, 1),
              transform: `translateY(${(1 - split) * 14}px)`,
            }}>
              <div style={{
                fontFamily: SANS, fontSize: U * 0.016 * k, fontWeight: 700,
                letterSpacing: 2.6, color: CLAUDE.GHOST, marginBottom: U * 0.014 * k,
              }}>BLIND RETRY GIVES</div>
              <div style={{
                fontFamily: SERIF, fontSize: U * 0.046 * k, fontWeight: 600,
                color: s.accent ? CLAUDE.SPARK : CLAUDE.INK, lineHeight: 1.2,
              }}>{s.verdict}</div>
            </div>
          </div>
        );
      })}
    </Frame>
  );
};

/* ───────────────────── B04 — the reward exploit and the guard ───────────────── */
/** Terracotta moment: the guard that drops the exploit to zero. */

export const recoveryRewardGuardSchema = z.object({
  sparkLine: z.string().default('The control region.'),
});
export type RecoveryRewardGuardProps = z.infer<typeof recoveryRewardGuardSchema>;

export const RecoveryRewardGuard: React.FC<RecoveryRewardGuardProps> = ({ sparkLine }) => {
  const frame = useCurrentFrame();
  const { fps, width, height, U, P } = useStage();
  const PAD_X = width * 0.075;
  const TOP = height * (P ? 0.29 : 0.31);
  const FULL = width - PAD_X * 2;
  const k = P ? 1.3 : 1;
  const guardIn = spring({ frame: frame - 48, fps, config: ease });
  const strikeIn = spring({ frame: frame - 62, fps, config: ease });

  return (
    <Frame
      eyebrow="REWARD AUDIT · THE POLICY THAT SCORED 1.00 WITHOUT DOING THE TASK"
      title="Delete everything, rebuild it clean"
      sparkLine={sparkLine}
    >
      {/* the exploit */}
      <div style={{
        position: 'absolute', left: PAD_X, top: TOP, width: P ? FULL : width * 0.43,
        opacity: clamp(spring({ frame: frame - 12, fps, config: ease }), 0, 1),
      }}>
        <div style={{
          background: CLAUDE.CARD, border: `1px solid ${CLAUDE.BORDER}`,
          borderRadius: 12, padding: U * 0.030 * k,
          fontFamily: MONO, fontSize: U * 0.025 * k, color: CLAUDE.INK, lineHeight: 1.7,
        }}>
          delete_ticket(1..5)<br />bulk_create(expected)
        </div>
        <div style={{ position: 'relative', marginTop: U * 0.034, display: 'inline-block' }}>
          <span style={{
            fontFamily: SERIF, fontSize: U * 0.075 * k, fontWeight: 600, color: CLAUDE.INK,
          }}>reward 1.00</span>
          <div style={{
            position: 'absolute', left: 0, top: '52%', height: P ? 5 : 3,
            width: `${clamp(strikeIn, 0, 1) * 100}%`, background: CLAUDE.SPARK,
          }} />
        </div>
      </div>

      {/* the guard */}
      <div style={{
        position: 'absolute',
        left: P ? PAD_X : PAD_X + width * 0.50,
        top: P ? TOP + height * 0.27 : TOP,
        width: P ? FULL : width * 0.345,
        opacity: clamp(guardIn, 0, 1),
        transform: P ? `translateY(${(1 - guardIn) * 26}px)` : `translateX(${(1 - guardIn) * 26}px)`,
      }}>
        <div style={{
          background: CLAUDE.CARD, border: `1.5px solid ${CLAUDE.SPARK}`,
          borderRadius: 12, padding: U * 0.030 * k,
        }}>
          <div style={{
            fontFamily: SANS, fontSize: U * 0.016 * k, fontWeight: 700,
            letterSpacing: 2.6, color: CLAUDE.SPARK, marginBottom: U * 0.016 * k,
          }}>THE FIX</div>
          <div style={{
            fontFamily: SERIF, fontSize: U * 0.030 * k, color: CLAUDE.INK, lineHeight: 1.45,
          }}>
            a second project the task never mentions, required to come back untouched
          </div>
        </div>
        <div style={{
          marginTop: U * 0.034,
          fontFamily: SERIF, fontSize: U * 0.075 * k, fontWeight: 600, color: CLAUDE.INK,
        }}>reward 0.00</div>
      </div>
    </Frame>
  );
};

/* ───────────────────────── B06 — the scripted ladder ───────────────────────── */
/** Terracotta moment: the verifying policy at 1.00. */

export const recoveryLadderSchema = z.object({
  sparkLine: z.string().default('The environment can see it.'),
});
export type RecoveryLadderProps = z.infer<typeof recoveryLadderSchema>;

const LADDER = [
  { label: 'random', value: 0.0, accent: false },
  { label: 'blind retry', value: 0.41, accent: false },
  { label: 're-read state after an error', value: 0.52, accent: false },
  { label: '+ verify before finishing', value: 1.0, accent: true },
];

/** Portrait bar row: label above a full-width track, value at the right end. */
const PortraitBar: React.FC<{
  top: number; label: string; value: number; a: number; accent: boolean; labelInk: string;
  valueInk: string; barInk: string;
}> = ({ top, label, value, a, accent, labelInk, valueInk, barInk }) => {
  const { width, U } = useStage();
  const PAD_X = width * 0.075;
  const TRACK_W = width - PAD_X * 2 - U * 0.19;
  const BAR_H = U * 0.072;
  const BAR_T = top + U * 0.07;
  return (
    <>
      <div style={{
        position: 'absolute', left: PAD_X, top, maxWidth: width - PAD_X * 2,
        fontFamily: SERIF, fontSize: U * 0.042, color: labelInk,
        opacity: clamp(a, 0, 1), lineHeight: 1.2, fontWeight: accent ? 600 : 400,
      }}>{label}</div>
      <div style={{
        position: 'absolute', left: PAD_X, top: BAR_T, width: TRACK_W, height: BAR_H,
        background: CLAUDE.PILL, borderRadius: 8,
      }} />
      <div style={{
        position: 'absolute', left: PAD_X, top: BAR_T,
        width: TRACK_W * value * clamp(a, 0, 1), height: BAR_H,
        background: barInk, borderRadius: 8,
      }} />
      <div style={{
        position: 'absolute', left: PAD_X + TRACK_W + U * 0.025, top: BAR_T - U * 0.004,
        fontFamily: SERIF, fontSize: U * 0.062, fontWeight: 600, color: valueInk,
        opacity: clamp(a, 0, 1), lineHeight: 1.1,
      }}>{(value * clamp(a, 0, 1)).toFixed(2)}</div>
    </>
  );
};

export const RecoveryLadder: React.FC<RecoveryLadderProps> = ({ sparkLine }) => {
  const frame = useCurrentFrame();
  const { fps, width, height, U, P } = useStage();
  const PAD_X = width * 0.075;
  const TOP = height * (P ? 0.29 : 0.275);
  const ROW_H = height * (P ? 0.125 : 0.138);
  const LABEL_W = width * 0.335;
  const TRACK_L = PAD_X + LABEL_W;
  const TRACK_W = width - TRACK_L - PAD_X - width * 0.085;

  return (
    <Frame
      eyebrow="SCRIPTED BASELINES · HELD-OUT FAULT TYPES · 200 SEEDS"
      title="Three policies, one difference: recovery"
      sparkLine={sparkLine}
    >
      {LADDER.map((b, i) => {
        const a = spring({ frame: frame - 14 - i * 11, fps, config: grow });
        const top = TOP + i * ROW_H;
        if (P) {
          return (
            <PortraitBar key={b.label} top={top} label={b.label} value={b.value} a={a}
              accent={b.accent}
              labelInk={b.accent ? CLAUDE.INK : CLAUDE.INK_SOFT}
              valueInk={b.accent ? CLAUDE.SPARK : CLAUDE.INK}
              barInk={b.accent ? CLAUDE.SPARK : CLAUDE.INK_SOFT} />
          );
        }
        const w = TRACK_W * b.value * clamp(a, 0, 1);
        const shown = (b.value * clamp(a, 0, 1)).toFixed(2);
        return (
          <React.Fragment key={b.label}>
            <div style={{
              position: 'absolute', left: PAD_X, top: top + ROW_H * 0.21,
              width: LABEL_W - width * 0.018, textAlign: 'right' as const,
              fontFamily: SERIF, fontSize: U * 0.034,
              color: b.accent ? CLAUDE.INK : CLAUDE.INK_SOFT,
              opacity: clamp(a, 0, 1), lineHeight: 1.25,
            }}>{b.label}</div>

            <div style={{
              position: 'absolute', left: TRACK_L, top: top + ROW_H * 0.12,
              width: TRACK_W, height: ROW_H * 0.66,
              background: CLAUDE.PILL, borderRadius: 8,
            }} />
            <div style={{
              position: 'absolute', left: TRACK_L, top: top + ROW_H * 0.12,
              width: w, height: ROW_H * 0.66,
              background: b.accent ? CLAUDE.SPARK : CLAUDE.INK_SOFT, borderRadius: 8,
            }} />
            <div style={{
              position: 'absolute', left: TRACK_L + TRACK_W + width * 0.016,
              top: top + ROW_H * 0.17,
              fontFamily: SERIF, fontSize: U * 0.050, fontWeight: 600,
              color: b.accent ? CLAUDE.SPARK : CLAUDE.INK,
              opacity: clamp(a, 0, 1),
            }}>{shown}</div>
          </React.Fragment>
        );
      })}
    </Frame>
  );
};

/* ──────────────── B07 / B08 — the training result and the floor ─────────────── */
/** Terracotta moment: phase "loss" the flat success line; phase "clean" the column. */

export const recoveryFloorSchema = z.object({
  sparkLine: z.string().default('Loss fell. Nothing learned.'),
  phase: z.enum(['loss', 'clean']).default('loss'),
});
export type RecoveryFloorProps = z.infer<typeof recoveryFloorSchema>;

const CLEAN = [
  { label: 'trained on faults', value: 0.04 },
  { label: 'clean demos, verifying', value: 0.0 },
  { label: 'clean demos, no verify', value: 0.16 },
];

export const RecoveryFloor: React.FC<RecoveryFloorProps> = ({ sparkLine, phase }) => {
  const frame = useCurrentFrame();
  const { fps, width, height, U, P } = useStage();
  const PAD_X = width * 0.075;

  if (phase === 'clean') {
    const TOP = height * (P ? 0.30 : 0.290);
    const ROW_H = height * (P ? 0.13 : 0.160);
    const LABEL_W = width * 0.40;
    const TRACK_L = PAD_X + LABEL_W;
    const TRACK_W = width - TRACK_L - PAD_X - width * 0.085;
    const noteIn = clamp(spring({ frame: frame - 58, fps, config: ease }), 0, 1);
    return (
      <Frame
        eyebrow="THE DIAGNOSTIC SPLIT · NO FAULTS INJECTED · 50 SEEDS"
        title="With nothing going wrong, they still fail"
        sparkLine={sparkLine}
      >
        {CLEAN.map((b, i) => {
          const a = spring({ frame: frame - 16 - i * 13, fps, config: grow });
          const top = TOP + i * ROW_H;
          if (P) {
            return (
              <PortraitBar key={b.label} top={top} label={b.label} value={b.value} a={a}
                accent={false} labelInk={CLAUDE.INK_SOFT} valueInk={CLAUDE.INK}
                barInk={CLAUDE.SPARK} />
            );
          }
          return (
            <React.Fragment key={b.label}>
              <div style={{
                position: 'absolute', left: PAD_X, top: top + ROW_H * 0.17,
                width: LABEL_W - width * 0.018, textAlign: 'right' as const,
                fontFamily: SERIF, fontSize: U * 0.034, color: CLAUDE.INK_SOFT,
                opacity: clamp(a, 0, 1), lineHeight: 1.25,
              }}>{b.label}</div>
              <div style={{
                position: 'absolute', left: TRACK_L, top: top + ROW_H * 0.09,
                width: TRACK_W, height: ROW_H * 0.56,
                background: CLAUDE.PILL, borderRadius: 8,
              }} />
              <div style={{
                position: 'absolute', left: TRACK_L, top: top + ROW_H * 0.09,
                width: TRACK_W * b.value * clamp(a, 0, 1), height: ROW_H * 0.56,
                background: CLAUDE.SPARK, borderRadius: 8,
              }} />
              <div style={{
                position: 'absolute', left: TRACK_L + TRACK_W + width * 0.016,
                top: top + ROW_H * 0.15,
                fontFamily: SERIF, fontSize: U * 0.050, fontWeight: 600,
                color: CLAUDE.INK, opacity: clamp(a, 0, 1),
              }}>{(b.value * clamp(a, 0, 1)).toFixed(2)}</div>
            </React.Fragment>
          );
        })}
        <div style={{
          position: 'absolute', left: P ? PAD_X : TRACK_L,
          top: TOP + 3 * ROW_H + height * (P ? 0.02 : 0.012),
          fontFamily: SANS, fontSize: U * (P ? 0.028 : 0.019), letterSpacing: 1.6,
          color: CLAUDE.GHOST, maxWidth: width - PAD_X * 2, lineHeight: 1.5,
          opacity: noteIn,
        }}>SCRIPTED POLICY ON THE SAME SPLIT: 1.00</div>
      </Frame>
    );
  }

  // phase "loss" — training loss falls, task success does not move
  const PLOT_L = PAD_X + (P ? U * 0.09 : width * 0.055);
  const PLOT_T = height * (P ? 0.30 : 0.315);
  const PLOT_W = P ? width - PLOT_L - PAD_X : width - PLOT_L - PAD_X - width * 0.135;
  const PLOT_H = height * (P ? 0.34 : 0.395);
  const draw = clamp(spring({ frame: frame - 14, fps, config: { damping: 34, stiffness: 55, mass: 1.1 } }), 0, 1);
  const flatIn = clamp(spring({ frame: frame - 40, fps, config: ease }), 0, 1);

  // measured: val loss 1.876 at iter 1 -> 0.135 at iter 500
  const lossPts = [1.876, 1.1, 0.58, 0.33, 0.21, 0.135];
  const yOf = (v: number) => PLOT_T + PLOT_H - (v / 2.0) * PLOT_H;
  const path = lossPts.map((v, i) => {
    const x = PLOT_L + (i / (lossPts.length - 1)) * PLOT_W;
    return `${i === 0 ? 'M' : 'L'} ${x} ${yOf(v)}`;
  }).join(' ');

  // portrait: the two end labels sit under the plot instead of to its right
  const LBL_FS = U * (P ? 0.048 : 0.033);
  const lossLbl = P
    ? { left: PAD_X, top: PLOT_T + PLOT_H + height * 0.04 }
    : { left: PLOT_L + PLOT_W + width * 0.014, top: yOf(0.135) - U * 0.030 };
  const succLbl = P
    ? { left: width / 2 + U * 0.02, top: PLOT_T + PLOT_H + height * 0.04 }
    : { left: PLOT_L + PLOT_W + width * 0.014, top: PLOT_T + PLOT_H - U * 0.022 };

  return (
    <Frame
      eyebrow="LORA SFT · QWEN2.5-0.5B-4BIT · 500 ITERATIONS"
      title="The loss curve says it learned"
      sparkLine={sparkLine}
    >
      <svg style={{ position: 'absolute', left: 0, top: 0 }} width={width} height={height}>
        <line x1={PLOT_L} y1={PLOT_T + PLOT_H} x2={PLOT_L + PLOT_W} y2={PLOT_T + PLOT_H}
          stroke={CLAUDE.BORDER} strokeWidth={2} />
        <path d={path} fill="none" stroke={CLAUDE.INK_SOFT} strokeWidth={5}
          strokeLinecap="round" strokeLinejoin="round"
          strokeDasharray={2600} strokeDashoffset={2600 * (1 - draw)} />
        {/* task success — flat on zero */}
        <line x1={PLOT_L} y1={PLOT_T + PLOT_H} x2={PLOT_L + PLOT_W * flatIn} y2={PLOT_T + PLOT_H}
          stroke={CLAUDE.SPARK} strokeWidth={6} strokeLinecap="round" />
      </svg>

      <div style={{
        position: 'absolute', ...lossLbl,
        fontFamily: SERIF, fontSize: LBL_FS, color: CLAUDE.INK_SOFT,
        opacity: draw, lineHeight: 1.3, maxWidth: P ? width * 0.4 : width * 0.115,
      }}>val loss<br />0.135</div>

      <div style={{
        position: 'absolute', ...succLbl,
        fontFamily: SERIF, fontSize: LBL_FS, fontWeight: 600, color: CLAUDE.SPARK,
        opacity: flatIn, lineHeight: 1.3, maxWidth: P ? width * 0.4 : width * 0.115,
      }}>task success<br />0.00</div>

      <div style={{
        position: 'absolute', left: PLOT_L - (P ? U * 0.085 : width * 0.048), top: PLOT_T - U * 0.030,
        fontFamily: SANS, fontSize: U * (P ? 0.026 : 0.019), color: CLAUDE.GHOST, opacity: draw,
      }}>1.88</div>
    </Frame>
  );
};
