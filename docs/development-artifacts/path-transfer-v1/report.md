# v2.13 fixed-batch machine summary

{
  "created_at": "2026-09-27T20:30:50.531538+00:00",
  "lock_sha256": "2f2a1c592d29f0d7f44421f7ce6a7b17175afd9c98fe497fdf187c1722e0d190",
  "states": 12,
  "cases": 216,
  "single_step_conditions": 5184,
  "probe_records": 31104,
  "gradient_arms": 864,
  "maximum_factor_kernel_error": 2.2271593408306473e-09,
  "maximum_sham_error": 0.0,
  "maximum_forward_error": 0.0,
  "maximum_gradient_reconstruction_error": 6.528658983532494e-08,
  "maximum_no_cross_unsupervised_norm": 0.0,
  "all_frozen_parameters_unchanged": true,
  "all_parents_restored": true,
  "all_full_duplicate_controls_identical": true,
  "source_updates_with_increased_loss": 0,
  "analysis_source_sha256": "e1b6a552a9092691d3d523a580f857bef694fa58c26527e03f140624a65d5e74"
}

Mature checkpoints, matched norm, fraction 0.0003. Negative loss change is improvement.

| Source kind | Arm | Source Δloss | Derived same Δloss | Derived other Δloss | Local retention Δloss |
|---|---|---:|---:|---:|---:|
| root | full | -0.01403899 | -0.00592739 | -0.00647882 | -0.00003187 |
| root | no_cross | -0.01257699 | -0.00533928 | -0.00592229 | -0.00003260 |
| root | no_mlp | -0.01112887 | -0.00541550 | -0.00604780 | +0.00000811 |
| root | fixed_qk | -0.01371054 | -0.00610034 | -0.00660911 | -0.00003521 |
| coherent | full | -0.02085531 | -0.01058786 | -0.00541255 | -0.00005345 |
| coherent | no_cross | -0.01892889 | -0.01005754 | -0.00515945 | -0.00005405 |
| coherent | no_mlp | -0.01518163 | -0.00916466 | -0.00535691 | +0.00002960 |
| coherent | fixed_qk | -0.01995615 | -0.01075549 | -0.00564347 | -0.00006365 |
| conflict | full | -0.02205180 | -0.00562825 | -0.00278093 | -0.00010236 |
| conflict | no_cross | -0.01988137 | -0.00532302 | -0.00265826 | -0.00008052 |
| conflict | no_mlp | -0.01657324 | -0.00414387 | -0.00221702 | -0.00005430 |
| conflict | fixed_qk | -0.02132179 | -0.00567430 | -0.00289414 | -0.00010892 |

## Gradient geometry

| Step | Arm | Cosine to full | Gradient norm | Supervised norm | Other-position norm |
|---|---|---:|---:|---:|---:|
| 0 | fixed_qk | 0.999829 | 16.347276 | 15.402228 | 4.453989 |
| 0 | full | 1.000000 | 16.354120 | 15.408978 | 4.457025 |
| 0 | no_cross | 0.952686 | 15.324884 | 15.324884 | 0.000000 |
| 0 | no_mlp | 0.858486 | 14.081949 | 13.218869 | 3.839456 |
| 5120 | fixed_qk | 0.963157 | 9.742752 | 9.893167 | 3.245370 |
| 5120 | full | 1.000000 | 10.273846 | 10.441503 | 3.383125 |
| 5120 | no_cross | 0.943172 | 10.441502 | 10.441502 | 0.000000 |
| 5120 | no_mlp | 0.761885 | 8.182802 | 8.296045 | 2.886769 |
| 15360 | fixed_qk | 0.969039 | 10.415418 | 11.136582 | 4.690232 |
| 15360 | full | 1.000000 | 10.785524 | 11.528135 | 4.859864 |
| 15360 | no_cross | 0.904902 | 11.528135 | 11.528135 | 0.000000 |
| 15360 | no_mlp | 0.760175 | 9.832339 | 10.630635 | 4.706006 |

## Taylor validation

| Step | Scale | Fraction | Relative RMSE | Resolved sign agreement |
|---|---|---|---:|---:|
| 0 | matched_norm | 0.0001 | 0.000081 | 1.000000 |
| 0 | matched_norm | 0.0003 | 0.000231 | 1.000000 |
| 0 | matched_norm | 0.001 | 0.000768 | 1.000000 |
| 0 | shared_lr | 0.0001 | 0.000080 | 1.000000 |
| 0 | shared_lr | 0.0003 | 0.000229 | 1.000000 |
| 0 | shared_lr | 0.001 | 0.000759 | 1.000000 |
| 5120 | matched_norm | 0.0001 | 0.000821 | 1.000000 |
| 5120 | matched_norm | 0.0003 | 0.002379 | 1.000000 |
| 5120 | matched_norm | 0.001 | 0.007938 | 1.000000 |
| 5120 | shared_lr | 0.0001 | 0.000817 | 1.000000 |
| 5120 | shared_lr | 0.0003 | 0.002349 | 1.000000 |
| 5120 | shared_lr | 0.001 | 0.007845 | 1.000000 |
| 15360 | matched_norm | 0.0001 | 0.000816 | 1.000000 |
| 15360 | matched_norm | 0.0003 | 0.002268 | 1.000000 |
| 15360 | matched_norm | 0.001 | 0.007502 | 1.000000 |
| 15360 | shared_lr | 0.0001 | 0.000807 | 1.000000 |
| 15360 | shared_lr | 0.0003 | 0.002264 | 1.000000 |
| 15360 | shared_lr | 0.001 | 0.007484 | 1.000000 |
