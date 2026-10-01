# Two-hop depth extension: 3 and 4 layers

Registered before any new training, 2026-09-30. The user requested actual extension of the existing two-hop experiment to compare how much training 3/4-layer models need to reach the 2-layer accuracy. This is a paired follow-up on three already observed worlds, not an independent new-world confirmation.

## Question and controls

Does adding a third or fourth block at width 128 reduce training exposure, or estimated computation, needed for the same held-out two-hop accuracy under the existing recipe? All six original 2-layer runs are reused read-only. Train twelve new runs: worlds 143011/143012/143013 × initializations 14301/14302 × depths 3/4. Data arrays, sample-stream seeds, initialization seed IDs, width 128, heads 4, dropout 0.1, AdamW lr 1e-4, weight decay 0.1, batch 256, warmup 2000, clipping 1 and 128000-step budget are unchanged. The historical 69 evaluation nodes and seven saved-weight nodes are unchanged. Standard GPT2 residual initialization scales with depth; the same seed does not mean all cross-depth weights are equal.

The unchanged local training/data/model core must match the historical confirmation lock by SHA256. A separate wrapper chooses the new output directory and source lock and checks the regenerated data arrays against each paired historical world before training. Each run keeps its copied inputs, RNG/optimizer checkpoints, per-node predictions and timing. Old outputs are never replaced. All twelve runs use their full fixed budget, without threshold-based early stopping or result-based tuning.

## Methods reused

The paper *Grokked Transformers are Implicit Reasoners*, arXiv:2405.15071v3, sections 2/3.1/3.2 and appendices A/B were reread. Its Appendix B compares model scales and optimization-step convergence. We retain the earlier micro-scale adaptation and only extend depth; this is not an exact reproduction of the paper's scale comparison. The fixed graph and 95/5 atomic partition, all-atomic plus ID-composition training, tail/EOS supervision and GPT2/AdamW recipe are inherited from the already validated implementation. See literature-alignment.json and the original batch's detailed source-method correspondence.

## Preregistered outcomes

Primary score remains the full ID held-out set's answer-plus-model-generated-EOS accuracy. Atomic, training-composition, OOD-composition and externally decomposed two-call scores are reported separately, alongside answer-only scores and NLL. OOD has small denominators; two-call evaluation adds supplied decomposition and inference computation.

For each matched world/initialization, freeze the exact saved 2-layer 128000-step ID accuracy as the paired target. Also use common thresholds 0.90 and 0.95. For every 2/3/4-layer trajectory, the threshold time is the first node of the first sequence of three consecutive registered nodes all at or above that threshold. Save the immediately preceding node and the third (confirmation) node. Do not interpolate, round percent before comparison, pick a best checkpoint or treat two passing final nodes as a confirmed hit. Later dips do not invalidate the registered three-node definition. Missing hits remain 'not reached within budget', never infinity or dropped observations.

The 2-layer endpoint by itself need not satisfy three-node confirmation of its own final threshold. In particular, reaching that endpoint accuracy earlier with a deeper model permits a budget comparison with the fixed 128000-step baseline, not an exact learning-speed ratio against an unobserved 2-layer threshold time. Common T90/T95 compare learning times directly. Ratios of threshold times or threshold computation are shown only for pairs where both times are observed, with missing denominators explicitly retained.

At every threshold record updates, examples, mean per-atomic and per-training-composition exposure, dataset passes, effective input tokens, supervised tokens and estimated training FLOPs. FLOPs use the unchanged actual padded length 4 and forward-plus-backward matrix count; exclude elementwise and optimizer overhead. Also retain fixed 128000-step endpoint results and complete learning curves. For an equal-compute descriptive comparison, take the last registered node no larger than each paired 2-layer full-budget FLOP cap; never interpolate or choose by accuracy.

Statistical unit is the world (three), with two initialization repeats inside each world. Report all runs and per-world averages/ranges. A censored world's aggregate threshold time remains unavailable instead of averaging only its reached initialization. No fitted depth law, significance claim or inference of one-hop-per-layer is authorized by this design. Parameter count and per-step computation grow with depth; this is not a matched-parameter comparison. No new mechanism intervention is included.

## Execution and completion

Assigned device: physical GPU 2 only. GPU 5 belongs to the separate multi-hop development effort; no other GPU is used. Parallel work may share CPU/filesystem resources, so report recorded training/evaluation time separately from elapsed UTC span and do not infer hardware efficiency from wall time.

Before launch, run the inherited data/training/report contract tests plus the new pairing and threshold/censoring tests; freeze the new config, wrapper, helper, tests, this contract, literature alignment, test log and unchanged scientific core in execution-lock.json and source/. Do not amend locked execution code while a run is active. Analysis/report code can be completed while training runs, and its final hash is recorded separately.

After all twelve runs complete, independently reload each new endpoint on the assigned GPU using the original auditor's snapshot import, identical TF32 setting and original GPU tolerances (discrete equality, NLL atol=rtol=1e-5). Verify all 828 new nodes' counts/FLOPs and complete paired data/exposure equality against the six immutable baselines. Recompute new per-node accuracy from saved predictions independently; retain failed attempts and numerical exceptions without weakening tolerances. Only then mark the extension complete. Summaries, CSVs, plots, source comparison and audits stay within this batch's artifact directory; the coordinating agent updates shared overview documents.
