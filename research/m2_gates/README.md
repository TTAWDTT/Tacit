# M2 离线预算门槛与逐 split 发现隔离

2026-09-30。状态：已根据独立审查的两项P1提交修订候选，**等待固定新head独立复审，不称fail-closed已验证**；不放行真实模型，不改变冻结实验。这是 [PR #6 的集成审查](https://github.com/TTAWDTT/Tacit/pull/6) 后续最小工程工作，不是协议优势或模型资格证据。

## 行为与复现

```sh
python -m unittest tests.test_m2_gates tests.test_m2_snapshot_races -v
python -m experiments.m2_gates.demo > /tmp/tacit-m2-demo.json
diff -u research/m2_gates/offline_demo.json /tmp/tacit-m2-demo.json
python -m unittest tests.test_emergent_ood_v04_split tests.test_emergent_ood_v04_episodes tests.test_emergent_ood_v04_induction tests.test_emergent_ood_v04_nl_feedback -v
```

无需安装依赖，Python 3.10+ 标准库；demo 只生成公开合成 sender 行，使用临时 SQLite 与假回调，随后清理。没有模型客户端、endpoint、权重读取、下载或真实执行参数。新入口是 `experiments.m2_gates`，没有往旧 runner 注册条件。API 示例就是 `demo.py`，不是伪代码。

Demo 的两个候选各有一个训练调用和一个验证调用，合计 discovery=4。成功候选每次需要11单位、超过单次上限10，因此不可选；失败候选仍保留 discovery 成本，其 deployment=2、每次 work=1/context=1、N=2，全链路预留=10，最多7次操作。选择后必须先记录部署费用；两次执行后第三次在回调前拒绝。这些数值是**发明的单元测试单位**，不是模型费用或研究结果。

## 预算规则

`Policy` 明确逐请求向量、累计向量和总调用/操作数上限。轴名称必须体现单位/端点/tokenizer；不把异构 tokens、bytes、钱或延迟相加。`Charge` 显式分开 work 与每次 context；完整值必须覆盖全部注册轴，缺轴、多轴、None、NaN、负数、布尔和浮点输入均失败，精确整数或十进制字符串可用。未观测的 reasoning/费用不能填写0以通过门槛。

`frontier` 对每个候选计算：**所有候选的全部 discovery + 该候选 deployment + N×(每次 work+context)**。失败候选也在 discovery；缺失候选直接失败。同一全局账本中其他split的已花预留也占用剩余额度，单列 `prior_reserved`；平均候选成本排除这些不属于当前协议的先前支出，但总预算可行性不能排除它们。单次上限独立检查，不能靠大N摊销掩盖单次超限。不同候选有不同部署和每次成本；超大N用精确算术计算，不展开海量列表。返回每候选可行/不可行及原因，`Scope.freeze` 只在可行候选中按严格验证成功数、稳定名字排序。未知必要成本会阻断整个冻结，而不是默认为可行。

`Ledger` 是持久化、single-flight、调用前预留控制器：

1. SQLite `BEGIN IMMEDIATE` 内检查政策指纹、重复ID、未完成/停机状态、逐请求/累计/次数上限及并发快照。
2. 在进入任何回调之前同步提交最坏情况预留，保存阶段、候选、scope binding、work/context、完整预留向量。
3. 成功、严格失败均保留预留；实际消耗另存，**不退款**，因此账本是保守预算上界，不把未使用预留称为实际花费。
4. 实际成本未知、缺收据、异常或超过已声明上界，保留费用并 halt；越界时保存已观察实际数。其他回调不再运行。崩溃遗留 inflight 同样阻断重开；不自动重试、不提供清零/退款/恢复API。
5. 新进程打开同一路径保留历史；更换政策不能重置。新空账本无法执行已有冻结产物，因为缺其 discovery receipts。并发回调/陈旧预检被阻断。对同一获批实验必须固定账本路径，不能用另一个文件规避全局预算。

这不是 OS sandbox：预算上界须由未来可信 adapter 从**完整序列化输入、固定最大输出/reasoning、工具调用、重试、费用与超时**得出，并由端点/子进程执行这些硬上限。一个谎报上界的任意 Python 回调仍可能先超支；本控制器只能记录并停止后续调用。因此目前没有把它接入模型或宣称生产费用已封顶。无法封顶的必要维度必须保持未知、拒绝真实执行。部署失败仍收费，但不会被当作已完成上手而放行执行。费用为0只可在确认没有外部按量收费时标明该维度；本地能源/计算机会成本仍单列未知。

`max_calls` 在这个小扩展中保守计每个事件/回调（包含部署/本地工具），不只计模型请求。一个回调不得偷偷包含多次模型/工具操作。计划的实际模型调用数还须单列；不得将已计入 token 轴的 payload 再重复加进同一 token 轴。

## split 和发现控制

Scope 绑定生成器验证过的 `split_sha256`、完整 role manifest 摘要、train/validation/test support 划分、receiver/config 标识。固定绑定不是 seed 字符串；更换任一项都会失效。`read_development` 只读当前 manifest 的 `sender_train.jsonl` 或 `sender_validation.jsonl`，检查文件哈希、行数、路径边界、严格字段白名单与每行 meaning 的 split/stage 成员资格。test/gold/receiver 路径在打开文件前拒绝；私有答案/候选ID/额外元数据不进入返回的 sender 投影。

Inventory 必须在验证前持久化封存，不能验证后添加新候选；所有候选训练行完成才可开始验证，验证后不再训练。每个候选/阶段必须按相同的 row index 完整执行，重复/跳行/越界被拒绝。冻结时同时核对：所有候选的真实持久化预留、失败记录、精确行ID/顺序、严格布尔结果与验证行数；不能省略失败候选，不能把失败改写成功。

每个冻结产物包含 inventory、训练/验证 provenance、discovery receipts 摘要、预算政策、候选特有部署/执行计划和N。执行前重新检查绑定、哈希、历史、部署计费及执行计划次数。跨 split、manifest 或 receiver 产物不会进入预算预留/回调。冻结后再添加 discovery 会使产物失效。

公开反例回归：seed17 的 train 与 seed23 的 held-out 相交 **52/64**。不仅拒绝直接拿 seed17 协议到 seed23，还用“给数据重贴 seed23 train 标签并重算文件哈希”的反例，验证 meaning-support 检查仍拒绝 held-out 行。

边界：manifest、分区和 receiver/config 来自可信 setup；散列不是签名，不能证明操作者没有把别的 split 信息手工抄入任意 card/callback，也不能清除模型/会话历史。每 split 必须隔离生成上下文、候选/卡、feedback、示例与 stateful session；真实服务 adapter、stateless reset 证明、card内容审计和台账权限尚待独立审查。当前只支持固定inventory、固定receiver、确定性sender、每阶段一次完整行遍历；多轮自适应提案/跨receiver重新适配不在此小入口中。跨receiver零样本研究需要另行明确 transfer manifest，不能绕过当前拒绝。

## 准确依赖与远端状态

新分支 `codex/tacit-m2-offline-gates` 直接从 main/base `5dd260c5824b280b5323a1707ab132524afa8e4c` 建立。仅新增本目录、`experiments/m2_gates/`、`tests/test_m2_gates.py`。运行时只复用 base 的公开 split validator；不复制/隐式堆叠其他PR，不依赖它们被合并。预算选择是 #6 要求的新受限入口，**不是修改 #4 v2 无约束 selector**，两者的结果不能互换。

本次开工核验六PR均 open、draft、未合并，main 未变：

| PR | 固定 head | 与本包关系 |
|---|---|---|
| #1 | `6de34bfd821aa8656d9063378087afc362aa9252` | 未来获批 Linux INDEX screen 可能使用；当前不启动 |
| #2 | `05d2921ead5ac8f555b7f776974f907de5bbc5a4` | 未来资格输出诊断；当前无 runtime hook |
| #3 | `ab06295aa5e0331c414f002b077dd71293c9c6f9` | 统计草案仍需重冻结，不继承5pp门槛或样本量 |
| #4 | `04129b8dcefedbaf1738f0851f9cd940ba863eff` | 接收者适配假设与成本定义；无代码依赖，无修改 |
| #5 | `e0dc30806886caacdcd44dcbc33a151c17c8af92` | compact等价负结果保留；无代码依赖，无新语言主张 |
| #6 | `84846ec8b132e9e2cd02bbf71535063ea863ba40` | 本包的审查合同、52/64反例与M2门槛来源 |

发布时精确新head、测试及CI状态在本PR描述与Space记录。冻结 runner/scorer/protocol/preregistration 没有修改；他人分支及 main 没有写入。

## 最小真实验证选项（未授权）

只读缓存检查范围：当前 `/workspace/Tacit/.cache/models/Qwen3-1.7B`、新checkout同路径、`/home/agent/.cache/huggingface/hub` 均不存在。**只代表这些路径**，没有扫描别的项目、SSH主机或全部磁盘；没有读权重、哈希模型文件或启动 GPU。历史就绪报告记载另一主机曾有缓存，但不能当作当前可用证明。

固定 Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`，仓库冻结 runner 声明两权重 shard 共 **4,063,515,592 bytes**。官方固定 revision 文件目录显示整个仓库约 **4.08 GB**，含 tokenizer/config 等；精确必需附件和传输额外开销仍待白名单核定。[官方固定文件目录](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e)。本次只读了目录网页，没有下载模型文件。

| 选项 | 下载/资源 | 调用与费用边界 |
|---|---|---|
| 优先：用户指定已有缓存的自有主机，沿用上述固定Qwen | 若缓存完整且授权检查通过，新增互联网权重下载0；若跨主机复制，约4.08GB传输仍须授权。实际缓存完整性/可用硬件未知 | 首先只申请INDEX_m最多12次（4 FULL门槛后8消息），无重试；旧24新token/次和资源阈值保持。无按量外部API费前提须确认；电力/本地计算成本未知，不能称总成本0 |
| 同revision在本环境重新准备 | 本环境未发现缓存，至少4,063,515,592 bytes权重，加附件（整仓约4.08GB）；依赖安装和实际CPU/GPU/RAM缺口未知 | 同12次screen；下载、安装、GPU及费用未批准，不能执行。没有用量报价便不填虚构价格 |
| 另选托管endpoint/模型 | 本地模型下载可为0，但模型/版本/tokenizer/推理预算不可默认等于冻结Qwen | provider、可强制输出/reasoning上限、单价和硬费用帽均未知；不是即插即用冻结runner选项，当前不推荐启动。须新配置审查与单独费用授权 |

若 INDEX screen 通过且接口已独立验证，可**另批** OOD 的独立12/12校准（不能由INDEX替代）。再另批工程 pilot：一个development split、同family两个事前手工候选、各4训练+4验证、确定性sender、0模型proposal、0重试、0确认性test，共16接收调用。三阶段最多 **12+12+16=40** 模型请求，分阶段 stop/go，不是一次性许可；每批仍≤12。不同校准/开发split保持独立。单split pilot只验证新门槛、序列化和成本覆盖，不估计优势/方差充分性/迁移。

冻结INDEX保留24输出token/次，OOD校准沿用既有48输出token配置；新的16调用pilot输入/输出/reasoning帽与完整提示tokens尚待可信adapter确定。费用上界应为 `Σ(每类请求最大输入×单价 + 最大输出/推理×单价) + 固定租赁/传输`；未知单价、不可封顶reasoning、工具或请求时长会阻断审批。当前授权仍为 **0真实模型请求、0模型下载、0付费API、0外部GPU**。

## 剩余决策

本包完成的是**可执行离线门槛**，不是整体M2放行。还需确定：使用哪个已有主机/模型；可信adapter的完整序列化、token计量/最大值、provider硬上限与实际收据；role manifest来源与stateless session隔离；独立资格和有效紧预算梯度；主假设/成本轴/N/split样本量/区间与多重比较；资源和费用授权。缺任何必要成本或资格，继续fail closed。无需再扩新平台或运行所有基线来掩盖这些空白。

## 本次验证证据

Python 3.12.14。最终新门槛27/27测试通过；相关split/episode/induction/NL-feedback回归20/20，组合命令47/47通过。另在**只本地**的隔离树整合六个固定PR与本包，最终114/114（新27、A9、B21、diagnostics7、冻结OOD runner26、runtime24）通过；没有发布该merge分支。#6独立审计JSON和本包demo JSON均逐字节复现，全部base既有blob不变。compileall及diff检查通过。测试重叠，不累计作科学样本；完整仓库suite未运行，没有模型端到端测试。六个既有PR heads 的PR workflow runs和commit statuses本次再查均为空；这是无报告CI，不是CI通过。新PR发布后的CI查询见其描述/评论。


## 两项 P1 的修订候选（等待独立复审）

[固定旧head 8fb5419 的独立评论](https://github.com/TTAWDTT/Tacit/pull/7#issuecomment-5914001217)发现真实绕过，旧测试通过不能抵消它们。新增最初7项针对性测试在未修改旧head上全部失败：原11单位请求在等待SQLite锁时被改成1，出现检查1/预留11/执行11；dispatch在验证旧discovery后采用新快照，接受了已经变化的历史。旧head不得当作工作中的预算/隔离放行版本。

修订：Charge构造时复制为只读mapping和精确Fraction，Ledger.run再固定同一规范化Charge及Policy；检查、总预留、work/context和actual比较全部使用该副本，不跨锁等待重读调用方字典。dispatch一次读取账本，discovery、deployment、执行计数和事务内CAS哈希全部派生于该次读取。freeze也从同一份all_rows派生history/prior，在返回前比对账本是否变化；freeze仍是某一时刻的计划验证而不是资源预留，后续改变由dispatch重新验证/拒绝。

同类审计修正了verified/sealed、inventory/source和freeze计划的浅层别名：哈希验证和实际使用现在共享脱离调用方嵌套字典/列表的副本。额外回归覆盖等待锁期间合法请求成本增大、异常后的原始context计费、锁超时无记录且连接可复用、事务前写入、freeze写入交错、验证后修改调用方schedule及可行性后修改计划。

修订候选观察：新增10项对抗/并发测试，加原27项门槛和20项相关回归，共57/57通过；demo JSON逐字节不变，compileall/diff检查通过。以上是作者测试，**不是独立复审通过**。单SQLite连接仍要求按线程使用；两个连接用数据库锁串行预留。操作者篡改数据库、绕过入口、伪造trusted来源、任意callback谎报上界、真实provider硬限额和外部session污染仍不由这些测试证明安全。资源资格/完整端点费用/统计冻结/用户支出授权仍是独立阻塞。
