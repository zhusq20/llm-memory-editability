# Two-layer common-error / competition follow-up

{
  "finished_at": "2026-09-27T20:41:35.560460+00:00",
  "lock_sha256": "8ed7122de8848da4a0f24799d4b8d2c7d1c8363584f743f09fb07a70c58fa556",
  "parents": [
    {
      "seed": 0,
      "step": 1024,
      "loss": 0.0016226809239014983,
      "accuracy": 1.0
    },
    {
      "seed": 1,
      "step": 1024,
      "loss": 0.0014907550066709518,
      "accuracy": 1.0
    }
  ],
  "trajectories": 48,
  "training_steps": 2048,
  "editing_steps": 24576,
  "measurement_points": 3120,
  "max_factor_kernel_error": 6.340029312374216e-08,
  "frozen_parameters_unchanged": true,
  "all_update_norm_controls_passed": true,
  "analysis_source_sha256": "5c03a9b052c79a4a0b4f991dc79fe1a9b7fce744ddb0a3c0543a4f9ae27762fb"
}

| Kind | Arm | Initial derived CE | Final derived CE | Min derived CE | Final source P | Final a-b margin | D correct | U correct |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| coherent | fixed_qk | 8.394567 | 0.002514 | 0.002514 | 0.997738 | 8.806423 | 1.000000 | 1.000000 |
| coherent | full | 8.394567 | 0.002443 | 0.002443 | 0.997806 | 8.726620 | 1.000000 | 1.000000 |
| coherent | no_cross | 8.394567 | 0.002212 | 0.002212 | 0.997804 | 8.956659 | 1.000000 | 1.000000 |
| coherent | no_mlp | 8.394567 | 0.002324 | 0.002324 | 0.997927 | 8.584177 | 1.000000 | 1.000000 |
| conflict | fixed_qk | 8.394567 | 8.175363 | 7.352053 | 0.997661 | -8.173036 | 0.000000 | 0.500000 |
| conflict | full | 8.394567 | 8.394656 | 7.976839 | 0.997759 | -8.392420 | 0.000000 | 0.666667 |
| conflict | no_cross | 8.394567 | 8.126964 | 7.044924 | 0.997800 | -8.124781 | 0.000000 | 0.500000 |
| conflict | no_mlp | 8.394567 | 8.371241 | 7.576796 | 0.997540 | -8.368799 | 0.000000 | 0.666667 |

## Sustained negative transfer onset

| Kind | Arm | Trajectories with onset | First observed steps |
|---|---|---:|---|
| coherent | full | 0/6 | [None, None, None, None, None, None] |
| coherent | no_cross | 0/6 | [None, None, None, None, None, None] |
| coherent | no_mlp | 0/6 | [None, None, None, None, None, None] |
| coherent | fixed_qk | 0/6 | [None, None, None, None, None, None] |
| conflict | full | 5/6 | [None, 0, 0, 0, 32, 136] |
| conflict | no_cross | 5/6 | [None, 0, 0, 8, 8, 488] |
| conflict | no_mlp | 5/6 | [None, 0, 0, 0, 16, 0] |
| conflict | fixed_qk | 5/6 | [None, 0, 0, 0, 0, 0] |

## Secondary mature-model replay

| Kind | Arm | Desired CE change | a-b margin change |
|---|---|---:|---:|
| coherent | fixed_qk | -0.01075549 | +0.00950231 |
| coherent | full | -0.01058786 | +0.00946276 |
| coherent | no_cross | -0.01005754 | +0.00871785 |
| coherent | no_mlp | -0.00916466 | +0.01041818 |
| conflict | fixed_qk | -0.00567430 | -0.01216569 |
| conflict | full | -0.00562825 | -0.01199483 |
| conflict | no_cross | -0.00532302 | -0.01159866 |
| conflict | no_mlp | -0.00414387 | -0.01226147 |
| root | fixed_qk | -0.00610034 | +0.00805735 |
| root | full | -0.00592739 | +0.00790253 |
| root | no_cross | -0.00533928 | +0.00700977 |
| root | no_mlp | -0.00541550 | +0.00753427 |
