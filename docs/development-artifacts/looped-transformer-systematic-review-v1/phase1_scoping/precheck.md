# 执行可行性预检说明（§3.2.1）

- 执行：2026-10-12 02:55–03:22 UTC，检索执行者子代理代主会话执行。完整记录见同目录 [`precheck.json`](precheck.json)。
- 对象：协议 v0.2 草案（未冻结，预检未修改；预检时 SHA-256 为 `51a74597…61d4f2`）。
- 范围：每条检索式只读总数；召回预演只取附录E候选本身的记录。没有筛选或提取。

## 1. 检索式记录数

| 检索式 | 来源 | 记录数 | HTTP | 审查者参考值 | 说明 |
| --- | --- | ---: | --- | --- | --- |
| AX-1a | arXiv | 503 | 200 | v0.1第一组 351 | 新增短语，预期内；见问题4 |
| AX-1b | arXiv | 97 | 200 | — | |
| AX-2 | arXiv | 2,787 | 200 | v0.1拆三段 218/1,268/1,585（未去重） | 与并集相符 |
| AX-3 | arXiv | 1,098 | 200 | 1,098 | 完全相同 |
| AX-4 | arXiv | 313 | 200 | 示例检索 265 | 区块词更多，预期内 |
| AX-3b（第2轮） | arXiv | 551 | 200 | v0.1整体放宽 8,758（已废弃） | |
| SS-1 | Semantic Scholar | 1,160 | 200（先429×3） | — | |
| SS-2 | Semantic Scholar | 2,109 | 200 | 2,109 | 完全相同 |
| SS-4 | Semantic Scholar | 527 | 200（先429×3） | — | |
| SS-3b（第2轮） | Semantic Scholar | 944 | 200（先429×4） | — | |
| R1 / R2 / R3 / R4（并集） | ACL Anthology本地正则 | 12 / 28 / 42 / 42（107） | 200 | — | 2024–2026年共37,803条 |
| R3b（第2轮） | ACL Anthology本地正则 | 20 | 200 | — | 其中7条不在第0轮并集 |

所有检索式都没有超长报错，不需要拆分。arXiv的计数用 `max_results=1`：`max_results=0` 固定返回HTTP 500（问题1）。Semantic Scholar 分页实取的ID数与 `total` 完全一致。

**数量保护**：第0轮合并去重前共 **8,718** 条（arXiv 4,798 + Semantic Scholar 3,796 + ACL各组之和 124），单条最高 AX-2 的2,787条。两项阈值（单条5,000、合计10,000）都没有触发，合计为阈值的87%。第2轮扩展另计（551 + 944 + 20），按§3.3.4第6条单独看新增量。

与参考值相比，差异较大的只有 AX-1a（+43%）和 AX-4（+18%），都来自v0.2新增的检索词。AX-3、SS-2与参考值完全相同。

## 2. 来源可达性

| 来源 | 结果 |
| --- | --- |
| arXiv API | 可用。`http://` 被301重定向到 `https://` |
| Semantic Scholar 批量检索、`/references`、`/citations` | 可用。没有密钥时频繁429，按退避规则都成功 |
| arXiv HTML（引文追踪替补） | 可用 |
| ACL Anthology 全量BibTeX | 可用。Last-Modified 2026-10-10 22:05:40 GMT，42,548,978字节，SHA-256 `85a53292…46d38242`。文件放在会话临时目录，正式检索须重新下载存档 |
| OpenReview API v2 与论坛页 | API返回403 `ChallengeRequiredError`；论坛页307重定向到验证页。没有绕过 |
| 第4号候选PDF | SHA-256复核一致（`ef8aa94a…19bbc81`） |
| 通用网页检索 | 预检没有执行，因为执行就会看到记录内容 |

## 3. 召回预演（不是正式召回检查）

- **核实**：23篇有arXiv记录的候选全部存在，标识与描述全部相符，首次公开日期都在时间窗内（最早2024-05-23，最晚2026-10-01）。没有"标识不符""不存在"或"不在时间窗"的情况。
  - 第8、9、12、13号的标题与提供的简称不同，但摘要中写明了STARS、Fully Looped Transformer、OperLoop、MELT，判为相符。
  - 第22号：审查者给出的2405.15071正确。
  - 第4号：PDF标题一致，页眉为ICLR 2027在审、匿名。按§3.1.4的用户PDF规则判为在时间窗内。PDF中没有投稿号34319，OpenReview ID无法核实。
  - 核心判据和子问题3类型要读全文，留到第二阶段。
- **命中**（分母为除第4号外的23篇）：
  - arXiv与Semantic Scholar合并：23/23（100%，Clopper–Pearson 95%区间85.2–100%）。调试组12/12，留出组11/11（区间71.5–100%）。
  - 分来源：arXiv 23/23；Semantic Scholar 22/23；ACL 0/23（这些论文都不在ACL论文集中）；本地正则在arXiv文本上23/23。
  - 第一组漏掉第2号和第22号，这两篇只由第四组命中（AX-4、SS-4、R4）。这与审查者的观察一致，说明第四组必须保留。
  - **漏检**：数据库合并后没有漏检。Semantic Scholar漏掉第24号（TRM），原因是它没有收录这篇论文（问题6）。

## 4. 附录G核实

19条全部存在，作者和年份正确。需要更正的有两条：

- **第8条（Cochrane第12章）**：缺章节作者，版本不具体。应改为 McKenzie JE, Brennan SE. Chapter 12 [last updated October 2019]. In: Higgins JPT, et al. (eds). *Cochrane Handbook* version 6.5. Cochrane, 2024。
- **第19条（Lipton & Steinhardt）**："2018/2019"不明确。应改为 *Queue* 2019;17(1):45–77, doi:10.1145/3317287.3328534（arXiv:1807.03341）。

其余条目多数只缺卷页或DOI，补全内容见 `precheck.json` 的 `appendix_G_verification`。例如 PRISMA-S 为 *Syst Rev* 2021;10:39，PRISMA-trAIce 为 *JMIR AI* 2025;4:e80247，Pineau 为 *JMLR* 22(164):1–20。

## 5. 需要协议修改或决定的问题

1. **arXiv计数参数**：§3.2.1写明计数用 `max_results=1`，附录A.1的接口改为https。
2. **R4k漏复数**：`generali[sz]ation` 改为 `generali[sz]ations?`。第21号摘要写的是"compositional generalizations"，arXiv命中而本地正则没有命中。
3. **arXiv会去掉短语中的停用词**：`"in depth"` 实际等同于 `depth`（两者都是7,540条）。所以AX-3b比SS-3b和R3b宽得多（候选命中19篇对11篇）。§3.3.2应写明这一点，并把"逐项对应"改为"近似对应"。这只影响精度，不影响召回。
4. **AX-1a噪声（待用户决定）**：AX-1a不限分类，`looped model(s)/network(s)` 经词干化后会匹配 closed-loop model、电网的 looped network 等。AX-1a的503条中有190条不在四个分类内。可以把这两组短语移到AX-1b，或要求同时满足Transformer区块。预演中所有候选都另有其他短语命中，这一改动不影响预演召回。
5. **零结果短语**：`depthwise recurrence` 和 `shared-parameter transformer` 在arXiv上为0条。正式检索时按§3.3.5记录。
6. **Semantic Scholar的覆盖缺口**：它没有收录TRM这篇知名论文。§3.2中"由Semantic Scholar覆盖DBLP"的理由应注明，arXiv仍是主来源。
7. **Semantic Scholar限流**：没有密钥时可以完成检索，但引文追踪请求多、耗时长。是否申请免费密钥由用户决定。
8. **ACL归一化**：BibTeX没有关键词字段，标题中有大括号和LaTeX命令。A.3应补上"先去掉大括号和LaTeX命令"。预检已经这样处理。
9. **冻结与git**：`.gitignore` 的 `/docs/*` 规则忽略了本目录，所以 protocol.md 和 precheck.json 目前不受git跟踪。§1.2的提交和推送需要 `git add -f` 或为本目录加例外。这由主会话决定。
