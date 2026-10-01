# H5 冻结输入/输出 token 参数

{'complete': True, 'models': 24, 'new_cases': 48, 'reference_cases': 96, 'all_checkpoints': 576, 'frozen_readout_checks': 48, 'missing': [], 'all_predictions_rescored': True, 'full_model_weights_hashed': False, 'frozen_weights_and_optimizer_steps_checked': True, 'source_sha256': '5e30b0211c0262edcd39e5ff01f13335d83a6f7b704a17ece81c722f53c24a8c'}

输入词嵌入与输出读出绑定，因此此干预同时冻结两者；位置嵌入及其余参数仍训练。这不是等参数量比较，也不能单独把差异归于输出读出；可训练梯度集合改变还会影响总范数裁剪。所有比较使用同一原始父节点、E93/R和采样流，固定512步，不选择最好检查点。

| Width | Scope | Cases | E | Heldout conflict D | Full U damage |
|---|---|---:|---:|---:|---:|
| 256 | mlp | 24 | 100.000% | 16.018% | 0.377% |
| 256 | all | 24 | 100.000% | 9.428% | 0.479% |
| 256 | all-freeze-embedding | 24 | 100.000% | 12.667% | 0.403% |
| 768 | mlp | 24 | 100.000% | 13.561% | 0.262% |
| 768 | all | 24 | 100.000% | 4.503% | 0.230% |
| 768 | all-freeze-embedding | 24 | 100.000% | 6.194% | 0.209% |
