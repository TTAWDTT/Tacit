# Agent C：机制审查与集成合同（2026-09-30）

状态：独立离线审查；M1 可收窄验收，M2 未放行。没有模型调用、模型下载、付费 API 或外部 GPU，没有正式 approve 或远端 merge。
计划：[TACIT 研究目标与执行计划](https://chatgpt.com/space/page_bd3d1298bd5c81919e4101089077ae2e)。

## 决策先行

- **A：停止“新增组合语言/表达力/线长收益”路线；保留并修订验证器。** 有效 wire 与既有 compact fields 完全相同。相同 payload、说明、模型配置和随机性下，换实现名字没有可辨识干预。有限值域、作用域校验是软件性质，不是 LLM 机制。原 head 的 DEL/C1 控制字符契约缺口已在最终 head 修正并复审，负结果不变。
- **B：继续收窄的接收者条件选择假设；修订后可作为 M1 离线交付，不能作为 M2 可执行选择器。** 新 head 正确固定发送消息、只扰动接收端关联，且明确标记无约束排序。真正的预算可行性与预调用限额尚未实现；break-even 只是给定成本的算术。
- **不扩基础设施。** #1/#2 保留为针对资格失败的最小工具；#3 是可复用的统计设计草案，当前假设、门槛及缺失策略不能直接拼入新实验。五包能离线共存，不等于模型端到端已就绪。
- **新阻塞：跨 split 发现支持隔离。** seed 17 训练支持与 seed 23 held-out 支持交集为 52/64。逐 split 内无泄漏不代表汇总训练后仍有未见组合。不要把模型先见过的其他 split 组合称为未见。

## 固定来源与依赖

共同 base/main：`5dd260c5824b280b5323a1707ab132524afa8e4c`。

| 包 | 固定审查 head | 处置与证据范围 |
|---|---|---|
| [#1](https://github.com/TTAWDTT/Tacit/pull/1) Linux launcher | `6de34bfd821aa8656d9063378087afc362aa9252` | 离线监管逻辑保留；真实 CUDA、Uvicorn FD、模型加载及生产遥测未验证 |
| [#2](https://github.com/TTAWDTT/Tacit/pull/2) response diagnostics | `05d2921ead5ac8f555b7f776974f907de5bbc5a4` | 保留元数据诊断；不修复 strict 输出，不授予资格 |
| [#3](https://github.com/TTAWDTT/Tacit/pull/3) confirmatory design | `ab06295aa5e0331c414f002b077dd71293c9c6f9` | 区间机械检查有效；研究方案需重新冻结 |
| [#4](https://github.com/TTAWDTT/Tacit/pull/4) B 原版 | `c6e0e3e80b43523b78aa273e0a6c6f9005827042` | 固定原版已复现，再逐项复审下列新版 |
| #4 B 修订 | `04129b8dcefedbaf1738f0851f9cd940ba863eff` | 4 文件变化；21 测试；M1 收窄通过，M2 预算门槛未通过 |
| [#5](https://github.com/TTAWDTT/Tacit/pull/5) A 原版 | `b438b8d07aae5e148b223141203599ccd6eef54c` | 8 新测试 + 38 回归，wire 等价；下列最终 head 已复审 |

| #5 A 修订 | `e0dc30806886caacdcd44dcbc33a151c17c8af92` | 4 文件变化；9 新测试 + 38 回归；65 个 Cc 控制字符拒绝验证通过 |

C 分支 `codex/tacit-research-integration` 直接从共同 base 建立，只新增本目录。没有复制或依赖未合并包来运行主仓库；审计脚本需显式传入包含上述固定源的临时集成 checkout。五包只在本地 `audit/tacit-five-pack` 合并验证，不发布该合并分支，不改别人分支/main。当前 checkout、base 和相关 heads 没有 AGENTS.md / `.agents/skills`。

独立审计对每个固定提交及临时集成树检查：**base 的全部既有文件无修改/删除/重命名/类型变化**，因此 strict scorer、runner、protocol、协议卡与冻结 preregistration 的 blob 均保留。并非只靠文件名抽样。五包原版共新增 23 文件，无路径冲突；新版 A/B 各只改自身四文件。C 新增文件不注册 runner 条件。

## 证据矩阵与缺口

| 审查维度 | A | B | 集成要求 |
|---|---|---|---|
| 强基线 | 已承认 compact 等价；必须有相同 typed card 的 compact | 适配/非适配在同一 family 内、相同 inventory/search 比较 | 优化短 NL、JSON、compact、任意码本、组合码本；FULL/无消息/缺源只用于相应对照 |
| 信息公平 | public schema + own fields；不能赠送 oracle decode | primitive support 一致仅是必要条件，card 可含任意文本 | 同事实/序列化边界/轮次/接收表/说明/推理预算；信息选择另配同选择 NL/JSON |
| 搜索公平 | 无新代码机制可搜索 | 所有失败候选费用计入，但 supplied costs 不验证真实性 | 同等训练数据、候选数、反馈、实际调用及 tokens；列实际使用量，不只列上限 |
| 完整成本 | 13,568 是每 seed payload bytes，非全链路 | affine 单轴 `F/N+c`，未知为未知 | 多维原始事件账本；重复说明为 recurring；未知维度不当零 |
| 可行性 | 预算梯度尚无真实验证 | v2 明确不评估预算，100 单位赢家可超 10 单位 cap | 先可行再比较，逐请求上限不能用平均摊销抵销；所有拒绝仍留分母与成本 |
| 角色 | codec scope 由调用方提供，不认证发送者 | 信任 stage、episode/support、receiver 声明 | 可信路由与来源 adapter；hidden-field 干预不改变 sender prompt/message；真实接口尚未审计 |
| split | 3 seed ×256 是同一有限全集的重复枚举 | stage guard 阻止标作 test 的反馈，不证明事实来源 | 逐 split 支持隔离和冻结，禁止跨 split 汇总发现污染；无需读封存模型测试来验证 |
| 统计 | split 聚类建议未定 n | paired split 设计未冻结 | 按 receiver 分层；不可把 episode、tuple 次数或 8,000 算术核对作独立效应样本 |
| 可辨识性 | 相同 prompt/wire 下根本没有新语言干预 | 适配效应可与 family 效应正交 | 先单 receiver、单 family 的最小对比，后再加迁移；不启动全交叉巨型实验 |
| 理论 | 可逆性要求 delimiter-free 原子、固定 schema、可信 disjoint scope | break-even 要稳定工作量、固定成本/每次成本及可比单位 | 性质只在前提下成立；无接收者行为或优越性推论 |

### 理论边界

A 的乘积消息数下界要求完整 support 及接收端上下文需要区分全部意义；受限候选共现或 side information 会弱化下界。固定宽度 bits 不等于 UTF-8/token/真实费用。作用域拼接不能恢复遗漏关系，也不能发现合法但错误的值。当前 codec 即使接受调用方提供的 x/y scope，也没有验证该调用方身份。

B 的 `F <= N*s` 以及正/零/负 saving 区间复现正确，等价点不是严格节省；负 saving 可为暂时收益，未知成本不能确定阈值。成本漂移、reset/cache 改变、失败率改变或分段价格需要累计账本，不适用固定 affine 参数。成本优势与成功率分别需要证据，向量优势需要各轴共同可行；观察到小 payload 不证明端到端更便宜。

### B 修订复审

原版 `shuffled_mapping` 保持的是字典标签集合，频率不均的实际消息总长可从 7 变 13。新版 `decoder_corruption_control` 保持确切发送消息序列，只改变接收关联；文档承认不同说明卡可能仍改变 tokens、推理成本。因此原始混淆已作为 M1 控制构造修正，完整成本匹配仍待真实测量。

新版 freeze schema 为 v2，含 `selection_scope=unconstrained_inventory_ranking`、`budget_feasibility=not_evaluated`、`deployment_authorized=false`。这避免错称可部署，却未实现受限优化。候选特有 deployment/context、N、单次/cumulative 预算、未知量和执行前限制均为 M2 硬条件。审查评论：[C 复审](https://github.com/TTAWDTT/Tacit/pull/4#issuecomment-5913061368)。

### A 修订复审

已独立复现原版接受内部 DEL/C1 的缺口；最终 `e0dc30806886caacdcd44dcbc33a151c17c8af92` 改为 Unicode Cc 拒绝，定义、实现及测试一致。独立复跑 47/47（9 新 + 38 回归），全部65个 C0/DEL/C1 拒绝，Unicode 合法边界保留；falsify 报告逐字节一致，更新的 source hash 正确。最小修改没有改变 compact 等价结论、冻结文件或 runner。M1 作为负结果和验证器交付足够，新增组合机制仍停止。[C最终评论](https://github.com/TTAWDTT/Tacit/pull/5#issuecomment-5913139635)。

### 旧 #3 与新合同的冲突

1. #3 的 A/B 指“组合卡/优化英语”，不等于新 Agent A/B；正式冻结必须使用 arm_id 和完整定义，不能混用字母。
2. 原 H1/H2 使用 0.05 实际重要性门槛，只是候选设计判断；计划明确不得无依据沿用。必须基于应用与验证证据重新论证，或者只报告区间。
3. 原“两臂×三 cap、768*n calls”不包含当前强基线/适配/迁移和 setup，不能作为本实验预算。
4. #3 以固定卡条件化、Hoeffding+Bonferroni 为主，A/B 草案考虑 cluster bootstrap/其他 family；M2 只能选择并冻结一个主 estimand、比较 family 和区间方法，不能试后择优。
5. #3 缺失真实结果时阻断主分析；A/B 提出 intention-to-evaluate 中失败计零。统一为：已尝试且违反格式/预算/拒答是失败；未采集或损坏的结果是行政缺失，不假造。主分析须完整，或在事前合理缺失设计下处理；耗尽预算则报告 incomplete。
6. 64 balanced episodes/split 与 generator 默认 16 validation/48 test meanings 的采样计划要明确各自含义，不能把 meanings 数当 episodes 数。独立 split 数由验证精度/功效和资源决定。

## 最小、真正可区分的下一实验（目前只设计）

**E0：已完成的模型无关证伪。** A 与 compact 全量 wire 等价，足以停止“新 wire 机制”。再做相同 prompt 的真实模型 A/compact 对照没有科学价值。先复核 schema/控制字符修订即可。

**E1：若有具体格式失败，测说明或校验干预。** 用同一 compact payload，generic card vs domain/scope card，给优化 NL/JSON 同等 schema。若测 validator，向全部格式注入同一、冻结的损坏集合；包含合法错值。把拒绝计失败，禁止免费修复/重试。确定性拒绝率已能离线得到；只有问题指向模型解码或有授权故障修复时才申请真实 pilot。结果只能归因说明/验证，不能恢复新语言主张。

**E2：B 最小辨识 pilot。** 一名合格目标接收者、同一 format family、固定候选 inventory：目标 receiver train/validation 选择 vs 预先指定 source receiver 同预算选择，另列 zero-search 和等量额外示例/推理基线。所有分支固定消息事实与推理配置；每 split 独立发现并 freeze，不共享跨 split feedback/card/examples。验证集用于选一个主 cap/arm 和估计方差，测试封存。预算可行性先通过后才比较 strict success。若优势只来自额外搜索，结论缩为搜索；若等量示例或推理消除优势，停止协议专有解释。NL 也能适配，优先与适配 NL 比较，不能只赢未经优化 NL。

**E3：条件迁移与机制控制。** 只有 E2 有信息量才做 source-frozen protocol 在未参与发现的 receiver 上的 zero-onboarding、固定卡、源训练示例、target re-adaptation；后两者分别收费，target feedback 后不叫 zero-shot。decoder-only 错映射检验关联使用；两端一致重标记检验词汇先验，不能混称错误映射。分别记录实际 full prompts/tokens；若成本不同，则估计是联合干预而非纯语义效应。

组合泛化 estimand 必须二选一并明确：逐 split 独立 train/select/freeze/evaluate，推断的是整个学习/选择流程；或完全事前公共 schema 的固定手工协议，推断条件化的固定表示效果。若卡从其他 split 的同一 ontology tuple 学得，不再声称全局未见组合。当前公开 fixture 审计不是对尚不存在真实 run 的泄漏指控，也不证明新 ontology 迁移。

停止/收窄规则：qualification 不过先停止；可行性/哈希/角色/账本失败即停相应批次；固定资源或样本用尽即停，不按显著性追加。两路线都无可识别干预时结束，不为保留路线制造新环境。

## M2 未决清单与最小资源申请模板

| 冻结项 | 当前状态 / 必填内容 |
|---|---|
| 研究问题 | 优先 E2 的单 family 选择效应；待验证前确定 primary contrast；A 新语言主张已否定 |
| 数据/角色 | 每 split train/validation/test 来源、seed namespace、sealed hash、生成器 revision；逐 split 隔离 discovery、hidden-field 审计 |
| 模型 | receiver/sender/model immutable revision、tokenizer hash、endpoint/build/template、温度/seed、context/output/reasoning cap；全未选择或资格未过 |
| 协议 | inventory/card/mapping/prompt/hash、允许反馈、调用次数与候选数、source receiver、实际搜索计费；无免费失败候选 |
| 成本 | 主轴及单位、请求限额和累计上限、候选特有固定/重复成本、缓存/reset、N（拟展示1/10/100/1000并列极限，不是择优N）；未知量处理 |
| 预算点 | 至少一宽松、两真正触发限制的共同点；仅由验证数据选定；FULL 不可行时保留标记，不插值 |
| 统计 | receiver strata、paired split estimand、独立 split 数、精度/功效依据、不等效/实用门槛依据、区间方法、family/multiplicity、完整性/停止规则 |
| 准入 | INDEX_m 4 FULL 后8消息总≤12；OOD独立12/12；Private Match独立校准，彼此不替代；修接口后新资格 |
| 执行 | separately reviewed cost/role adapter；来源/提交与输出 manifest；先模型资格再 train/validation pilot，确认性 test 暂不申请 |

资源申请（提交时把每个空白填为数值/摘要；本文件**不构成批准**）：

```text
目的/阻塞：E__ 的具体资格或验证问题；不打开确认性测试。
冻结代码/模型/tokenizer/config/prompt 哈希：__。
位置：已有本地资源 __；新增下载 __ bytes；外部 GPU __ hours。
预算阶段：资格 __ 次；proposal __ 次；全部候选 train __ 次；validation __ 次；
           source/target receiver 对照 __ 次；失败/重试上限 __ 次；总尝试上限 __ 次。
每次 input/output/reasoning 上限：按 endpoint 分列 __；现有每批限制保持不变。
CPU/GPU/RAM/磁盘、墙钟上限：__；费用币种及硬上限 __（未知不能批准执行）。
成本估算：sum(各类最多尝试数 × 对应最大单次费用) + 固定租赁等 __；
          明示供应方无 reasoning 上限等任何无法封顶项。
缓存/reset：__；candidate-specific feasibility 与 N：__；原始账本去向/manifest __。
退出：资格/角色/哈希/成本/资源失败或上限达到立即停止；不自动扩大预算。
请求授权范围：只资格 + train/validation pilot；确认性测试、额外下载/支出另批。
```

在这些空白解决前，授权量仍为 **0 模型请求、0 模型下载、0 付费 API、0 外部 GPU**。资源不足则保留负结果/不确定性，不转成“已验证”。

## 离线复现与观察结果

环境 Python stdlib。没有安装依赖、打开模型文件或启动真实 endpoint。原固定五包在隔离树 octopus merge 无冲突；再合并 A/B 修订均无冲突。

| 检查 | 本次独立观察 |
|---|---|
| A 原版8 + B原版18 + diagnostics7 + frozen OOD runner26 + runtime24 | 83/83 通过 |
| A focused existing regressions | 38/38 通过 |
| role ledger / induction / NL feedback（mocked） | 14/14 通过 |
| Linux fake process suite | 14/14 方法通过；报告25 lifecycle scenarios |
| #3 synthetic helper | 9/9 通过；JSON 逐字节复现 |
| A falsify | JSON 逐字节复现；3×256 tuple、其中3×64 held-out；每seed13,568 bytes双方相同；Private Match8种消息相同 |
| A revised | 47/47；falsify报告逐字节复现；C独立审计再检查65个Cc |
| B revised | 21/21，通过8,000整数不等式检查；原demo equality N30、strict N31、重复上下文 never |
| C独立审计 | 所有base blobs不变；52/64跨seed overlap；7→13；decoder payload固定；ranker未评估预算；合法错值/scope不认证反例 |
| compileall / diff --check | 通过 |

测试有重叠，不把重跑计数累加成科学样本。全仓库 suite 未跑；生产模型/真实完整成本/泛化/端到端兼容未验证。CI：已查询五个原 heads 的 PR workflow runs 和 commit statuses，全部为空，树无 `.github/workflows`。这是**没有报告的 CI**，不是通过。最终 A `e0dc308` 与 B `04129b8` 已再查，PR workflow runs / commit statuses 仍全为空；C 发布状态见其 PR。

复现（全部在本地；不要推送临时 merge 分支）：

```sh
git fetch origin '+refs/pull/*/head:refs/remotes/origin/pr/*'
git worktree add -b audit/tacit-five-pack-repro /tmp/tacit-five-pack 5dd260c5824b280b5323a1707ab132524afa8e4c
cd /tmp/tacit-five-pack
git merge --no-edit 6de34bfd821aa8656d9063378087afc362aa9252 05d2921ead5ac8f555b7f776974f907de5bbc5a4 ab06295aa5e0331c414f002b077dd71293c9c6f9 c6e0e3e80b43523b78aa273e0a6c6f9005827042 b438b8d07aae5e148b223141203599ccd6eef54c
git merge --no-edit 04129b8dcefedbaf1738f0851f9cd940ba863eff
git merge --no-edit e0dc30806886caacdcd44dcbc33a151c17c8af92
python -m unittest experiments.compositional_protocol.test_codec experiments.receiver_adaptation.test_reference tests.test_response_diagnostics tests.test_emergent_ood_v04_runner tests.test_tacit_runtime -v
python -m unittest tests.test_compact_labeled_fields tests.test_private_match_v03 tests.test_emergent_ood_v04_split tests.test_emergent_ood_v04_episodes tests.test_emergent_ood_v04_induction tests.test_emergent_ood_v04_nl_feedback -v
TACIT_LAUNCHER_TEST_REPORT=/tmp/tacit-launcher-offline.json python tests/test_index_linux_launcher.py -v
python research/confirmatory_ood_synthetic_v0_1.py --self-test
python research/confirmatory_ood_synthetic_v0_1.py > /tmp/tacit-synthetic.json
diff -u research/data/CONFIRMATORY_OOD_SYNTHETIC_V0_1.json /tmp/tacit-synthetic.json
python -m experiments.compositional_protocol.falsify > /tmp/tacit-falsify.json
diff -u research/compositional_protocol/offline_report.json /tmp/tacit-falsify.json
python -m experiments.receiver_adaptation.demo
# C audit remains in the independent C checkout; replace /path/to/C with that checkout.
python /path/to/C/research/integration/audit.py --source-root /tmp/tacit-five-pack > /tmp/tacit-c-audit.json
diff -u /path/to/C/research/integration/offline_audit.json /tmp/tacit-c-audit.json
```

`offline_audit.json` 是公开fixture可重复结果，不含私有角色账本或模型结果。审计不会 fetch/push/run models；未取得固定对象时失败，不偷偷换最新分支。更新任何 head 后按 diff 复审受影响项，旧观察保留来源。
