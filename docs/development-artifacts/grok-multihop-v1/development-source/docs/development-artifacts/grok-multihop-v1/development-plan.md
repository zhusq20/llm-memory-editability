# Three- and four-hop development, before execution

2026-09-30. This is calibration, not independent confirmation. No claim assumes
one hop per layer or that failure within budget establishes an architectural limit.

Reuse the graph/atomic partition and GPT2 training implementation of
`grok-depth-v1`, grounded in *Grokked Transformers are Implicit Reasoners*
(arXiv 2405.15071v3, sections 2, 3.1–3.3 and its public implementation at
734ca654ec7a71dd6737d640407fac14491d538c). The source studies two-hop composition.
This extension asks how depth affects longer-path generalization, in a new small
controlled task. It is not a numerical reproduction of the paper.

## Fixed first development batch

- One new world, seed 144001, 128 entities, 16 relations, degree 8; 1,024 atomics.
- Atomic edge split 95/5 ID/OOD. All atomics, including OOD, enter training.
- Separate k=3 and k=4 tasks. Each trains only atomic rows plus one target length.
  No intermediate answer supervision; only final entity plus generated EOS.
- Enumerate every path of target length. All-ID paths are split by independent
  Bernoulli 10% test reservation before uniformly drawing phi=6 times the ID-edge
  count for training. All-OOD paths are descriptive; mixed paths are excluded.
- Reserve complete queries `[h,r1,...,rk]`. Shared atomics and constituent
  subpaths are intentional and are not complete-query leakage.
- Fixed 1,024 test-probe rows sampled without model outputs from the reserved pool
  (all rows if fewer). Score them at every node; score the whole reserve at 128k.
- Record actual per-edge use counts at each hop position, including first/last;
  cycles, repeated entities, answer equal to an earlier entity, exact split counts,
  and training-pool coverage. These are diagnostics; never filter or regenerate.
- k3 uses depth4; k4 uses depth6 as an initial learnability reference. Width128,
  4 heads, dropout0.1, initialization14401, shuffled-stream seed144011, batch256,
  AdamW lr1e-4/wd0.1, 2,000-step warmup then constant LR. Fixed 128,000 updates.
- One GPU (physical GPU5), runs sequentially. Same FP32+TF32 calculation as the
  parent task, padded length k+2, exact model parameter count and length-aware
  matrix FLOPs. Log exposure and effective input tokens separately.
- Assess atomics (also ID/OOD separately), train compositions, fixed held-out
  probe, full held-out endpoint, all-OOD compositions, and autonomous k calls.
  Autonomous calls receive relation decomposition but carry their own predicted
  intermediate entities and EOS. They use extra calls and are not equivalent
  computation to a single forward pass.
- Nodes 0/100/250/500/1000 then every 2,000 updates. Save weights at
  0/1000/4000/16000/32000/64000/128000 and resumable state every node.
- Record first three consecutive evaluation nodes at or above 90% as descriptive
  T90, right-censored if absent. A single developmental world cannot support a
  population estimate, significance claim or universal depth requirement.

## Development decision

phi6 preserves training count and examples per record but greatly reduces the
fraction of possible paths exposed as k rises. If a deeper reference learns
atomics and training queries but not held-out paths, report this and review
coverage before deciding on a separate, documented phi/budget revision. Do not
automatically launch the complete depth grid or attribute failure to depth.

After calibration, lock the task and compare at least depth2/3/4/6 (depth1 if
feasible) in fresh worlds, with paired seeds and streams. Compare fixed updates,
exposure and matched estimated compute; parameter count is an additional
confound, not evidence for a one-layer-per-hop law. The confirmation matrix is
not yet approved or locked by this development document.

All source, configuration, split evidence and outputs are written only to new
grok-multihop files; historical experiments are preserved.
