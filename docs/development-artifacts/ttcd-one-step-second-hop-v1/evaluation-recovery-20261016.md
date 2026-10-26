# 一步写入与merge评测恢复（2026-10-16）

用户明确授权修复报错、完成剩余一步写入和merge评测并汇总，已经完成的训练不得重跑。

## 原因与修正

原一步评测的Docker create在模型执行前失败：NFS上的批次父目录为700，daemon无法遍历到输出bind源。2026-10-16实际重放创建命令得到`invalid mount config ... permission denied`，没有创建容器。DiscoLoop依赖曾未齐全，现三架构stream10000检查点、独立训练审计与成功退出记录均已齐全，按原契约重新核验确切checkpoint SHA即可；不降低审计要求。

新增宿主恢复入口为`scripts/resume_ttcd_evaluations.py`。冻结科学源码、配置、数据、32事实、32旋转、查询、精度、写入公式、步数和评分全部复用。仅将输出bind改为持久根下单独创建的`evaluation-recovery-20261016/<batch>/<arm>/evaluation`；旧目录权限不改。完成阶段的常规文件以同文件系统硬链接发布到原canonical输出，避免复制大型快状态；已有不同证据不得覆盖。merge父评测bind及汇总comparison bind指向同一份已核验输出。

恢复只允许prepare/preflight/calibrate/evaluate/audit/cloud-audit/aggregate；显式拒绝train、common-train和foundation-train。每条架构先完成原四步训练终点的一步评测及本地/云端审计，再做原merge两跳位比较；merge已通过的纯图prepare直接复用。每阶段只申请实际独占空闲GPU与原CPU集合，不抢占、共享或停止其它项目。原失败claims与状态归档，不改成成功。成功阶段须同时有退出0、原result receipt与canonical JSON SHA吻合才复用。

## 验证与收尾

原固定镜像下，恢复/原一步控制器/原merge控制器共65项CPU测试与Ruff通过。启动前逐文件验证两批冻结源码，逐一核对三份父检查点与独立训练审计；收尾再次核对这些哈希。新增基础及外层训练更新预算为0，评测快状态重写按原契约执行，不冒称零评测计算。

两批完成后，另用已完成一步训练批次的原冻结aggregate入口生成独立control-report，补入四步训练、一步部署的对照；已存在的主报告和所有训练/评测不重跑。结果与状态最终写回`docs/results.md`及`docs/roadmap.md`。恢复记录与源哈希保存在持久根`evaluation-recovery-20261016`及各批同名目录。

## 实测后的调度修订

首个正式评测GPU7约占1.6GiB、GPU利用率约1%，主要受状态I/O影响。按项目尽可能并行且不抢占其它任务的要求，新增宿主`scripts/share_ttcd_evaluation_recovery.py`只接管原宿主调度，不重启正在执行的DiscoLoop评测容器；两条尚未启动的架构评测也仅使用原本空闲的GPU7。最多三个本次评测模型、各保留原独立CPU与输出，启动前留至少12GiB显存；逐一核验容器命令、镜像、冻结manifest环境和实际CUDA PID，外部任务拒绝共享。五项新调度契约与Ruff通过。原科学预算、完整矩阵及审计要求不变；共享耗时不作为独占架构吞吐比较。旧独占启动记录保留。

DiscoLoop正式1313状态及W&B同步完成后，宿主在大量NFS路径的重复mkdir/stat发布中耗时较长。`output-publish-repair-20261016`仅把证据发布改为逐目录一次scandir与硬链接，已有同inode直接保留、不同内容仍拒绝覆盖；七项发布/共享契约与Ruff通过。宿主再次接管仍在运行的评测及已退出但尚待发布的同一容器，不重算任何已完成评测。原共享入口完整保留，最新入口为持久根`evaluation-recovery-20261016/controller-process-publish.json`及两批`controller-process-sharing.json`。

随后接管DiscoLoop已退出容器时，旧通用`handoff-monitor.log`已存在，宿主在创建新日志前报FileExistsError，未重新执行模型。`publish-log-resume-20261016`保留此错误，改用修订号隔离日志，只允许这个明确诊断的宿主错误恢复；八项发布/共享/日志契约及Ruff通过。两个仍在执行的容器继续、DiscoLoop已存1313状态直接发布后进入审计。**最新宿主入口**为持久根`evaluation-recovery-20261016/controller-process-publish-log.json`及两批`controller-process-sharing.json`；前两轮宿主记录及失败全部保留。

三架构正式状态生成均已完成后，`publication-cache-repair-20261016`进一步省去重复的宿主遍历：只有已提交的成功阶段收据、对应完整JSON SHA及两路径同inode全部成立，才省去反向重新播种；正式evaluate完整发布收据成立后，后续阶段不重复遍历已硬链接的immutable states树，仍发布新增audit/cloud JSON。冻结科学审计仍逐状态执行全部检查，缓存不用于跳过科学重放。九项发布/共享/日志/收据完整性契约及Ruff通过，原容器继续。**当前恢复入口优先读**`evaluation-recovery-20261016/controller-process-publication-cache.json`及两批`controller-process-sharing.json`；全部先前版本与失败保留。

## GPU占用变化后的审计接续

三架构一步评测各1313状态、441002查询的独立重放和完整云端JSON核验全部通过。DiscoLoop与Loop的merge各2081正式状态已生成；Transformer的merge仍执行时，GPU7新增其他用户的两个CUDA进程，原保守准入停止提交新审计，现有模型不停止、不重启。登记的PID、UID、进程启动时间及GPU UUID见持久根`resource-admission-repair-20261016/foreign-processes.json`。

新增`scripts/resume_ttcd_audit_resources.py`只接管宿主监控。默认不启动跨用户共享；新阶段可申请无CUDA进程、显存占用不超过1024MiB、利用率不超过5%的空闲卡，并持有独占GPU锁。只有用户明确选择继续限额共享时，才可用`--allow-existing-neighbors`在原GPU7上恢复最多三路本批模型；登记进程身份须相同，任何新增未知进程仍拒绝，新阶段启动前保留至少32GiB空余显存，PyTorch分配器限制为整卡6%。已运行模型维持原容器/PID，源码、写入与读出参数不变，其他用户任务不停止或搬移。六项资源契约与既有九项共享契约、Ruff通过，内存限额入口的原脚本/参数传递已实际模拟核对；不是新的科学评测。

**最新接续入口**是持久根`evaluation-recovery-20261016/controller-process-resource-admission.json`及两批`controller-process-sharing.json`。原一步训练批次的control-report汇总进程继续等待；另排队`scripts/finish_ttcd_evaluation_report.py`，只在全部原审计、云端核验及control-report通过后生成18条件表并更新结果/roadmap的本次进度段落。尚未通过的merge审计不得写成完成。

## 用户授权限额共享后的恢复

用户随后明确要求“限额共享GPU7继续，把剩余merge评测、审计和结果汇总完成，已完成的训练不要重跑”。此时三架构merge正式评测均退出0、2081状态及结果收据完整；未再执行正式评测。原资源入口曾在空闲卡上启动三个审计，但限额封装通过runpy运行脚本时没有恢复原脚本目录到sys.path，均在模型执行前因同目录`run_ttcd_one_step`无法导入而退出。修正仅补齐Python直接执行脚本原有的导入路径；新增真实新进程、同目录导入和原参数传递回归检查，与原资源/共享检查共16项及Ruff通过。三个失败容器改名保留，失败阶段完整归档，原科学源码未修改。

`approved-gpu7-audit-repair-20261016`冻结修正入口、用户授权及当前邻居PID/UID/启动时间。宿主首启因归档目录与接续目录重名退出，发生在提交新模型前；原归档改到独立目录后接续，该宿主错误日志同样保留。三路正式审计已在GPU7实际运行，每路分配器上限整卡6%、最多三路，启动前至少32GiB空余显存。已运行邻居不停止、不搬移，原正式评测/训练新增更新0。

**本次最新入口优先读**持久根`evaluation-recovery-20261016/controller-process-approved-gpu7-resume.json`，两批`controller-process-sharing.json`指向同一新源码；三个审计及云端验证仍须完成。原汇总与报告等待进程随前次失败正常退出，其日志保留；原冻结汇总入口和已核验报告入口现已接续，分别见`summary-controller-process-approved-gpu7.json`与`report-publication-controller-process-approved-gpu7.json`。此前各入口均为历史，不重复启动。
