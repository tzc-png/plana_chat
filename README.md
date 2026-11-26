# plana_chat — 普拉娜 Telegram 二次元角色语音聊天机器人

让《蔚蓝档案》的「普拉娜」进驻你的 Telegram，用她的声音和你聊天。

本项目是一个**自包含**的常驻服务：本地 GPT-SoVITS 语音合成 + 大模型角色扮演（云端或本地），全程跑在你自己的机器上。

## 功能

- **角色扮演**：普拉娜人格（系统 prompt 可换），回复带情感标签
- **本地语音**：GPT-SoVITS 日语模型本地合成，Telegram 语音消息（OGG/Opus）
- **长期记忆**：每个用户独立的 SQLite 记忆，含「滚动历史 + 总结压缩 + 事实抽取 + 语义检索」四层，能跨天记住你的喜好、未完成的事，并能按语义精准回忆过往对话细节
- **动态认知系统**：好感度 + 关系阶段（陌生→熟悉→喜欢→恋爱→结婚），好感度随情感标签增减、阶段自动升降，喜欢→恋爱/恋爱→结婚需你主动申请
- **主动问候**：早上 8 点、中午 11 点、晚上 9 点主动发问候（上下文感知、情境演绎，节日/纪念日/未完成事项会带一句关心）
- **重新提醒**：对问候反应「要加班/要贪睡」时挂起提醒任务，20 分钟后自动再提醒（期间你发消息则取消）
- **话题系统**：FIFO 逐条处理，识别话题切换、挂起未聊完话题（含上下文）、合适时按三层优先级拾起；好感度按话题结算；吃醋/黑化挂起后切走即结算
- **三种模式**：`default`（默认）/ `active`（总是续接）/ `quiet`（免打扰），聊天 `/quiet` `/active` `/default` 或 PC `plana quiet/active/default` 切换
- **发送前消息检查**：回复发送前拉取新消息，无营养则忽略、同话题则合并、不同话题则放弃当前回复重处理
- **回复自检 + 情感自检**：发送前分别自检文本与情感是否符合上下文/关系阶段，有问题则修正
- **情感表情图**：按回复情感自动发对应立绘表情图（可自备图片，默认关闭）
- **智能搜索**（可选）：问实时/事实性问题时联网搜索再回答
- **引擎可切换与自动降级**：云端 DeepSeek（默认）或本地 Ollama；云端余额不足/失败时自动切本地，日志红色高亮提示
- **命令行管理**：`plana on/off/system_on/...` 一条命令启停，可做成开机自启

## 架构

```
Telegram 消息 → 角色对话(LLM) → 情感标签解析/文本清洗 → 中日翻译 → 本地 TTS(GPT-SoVITS) → OGG 语音
                                      ↑
                         长期记忆（摘要 + 事实 + 未完成项）
```

核心（对话/翻译/语音）是**通道无关**的，未来接其他聊天平台只需加一个适配器。

## 前置要求

| 类别 | 要求 |
|---|---|
| 操作系统 | Linux（作者在 Ubuntu 24.04 验证） |
| GPU | NVIDIA 显卡（语音引擎用 CUDA 版 torch，约 1GB 显存；本地 LLM 需更多） |
| Python | 两个独立环境：语音引擎用 **3.10**（conda），机器人用 **3.12**（venv） |
| 软件 | [miniconda](https://docs.conda.io/en/latest/miniconda.html)、`ffmpeg`（带 libopus）、`git`、[Ollama](https://ollama.com)（用于本地 embedding 语义记忆） |
| 账号 | Telegram 账号；[DeepSeek](https://platform.deepseek.com) API key（可选，可改用本地 Ollama） |
| 网络 | 代理（Telegram 国内需代理；机器人服务器端也要能连上 Telegram） |

## 目录结构

```
plana_chat/
├── run_telegram.py            # Telegram 机器人入口
├── bin/plana                  # 命令行管理（on/off/system_on/system_off/status/help）
├── config.yaml                # 主配置
├── .env                       # 密钥（不提交，需自建）
├── .env.example               # .env 模板
├── plana_chat/                # 领域逻辑（不依赖 nonebot）
│   ├── telegram.py            # Telegram 适配器（httpx 长轮询）
│   ├── pipeline.py            # 完整链路编排
│   ├── memory.py              # SQLite 记忆（滚动历史 + 摘要 + 事实）
│   ├── search.py              # 联网搜索
│   ├── character/             # 人格 prompt（plana.txt / plana_compromise.txt）
│   ├── llm/  translator.py  tts/  audio.py  postprocess.py  config.py
├── voice/                     # 本地语音引擎（自包含）
│   ├── plana_tts_server.py    # GPT-SoVITS 推理服务
│   ├── plana_engine/          # 引擎代码（含 chinese-hubert-base，见下）
│   └── models/                # 语音模型（见下，不进 git）
├── assets/emotions/           # 情感表情图（按情绪分子文件夹）
├── scripts/                   # setup.sh / verify_*.py / simulate_persona.py
└── docs/                      # 调研报告（记忆方案 / 人格）
```

## 快速开始

> ⚠️ 两个最容易卡住的地方先说明：
> 1. **语音模型文件不进 git**（太大），`git clone` 后需要自行下载（见第 1 步）。
> 2. **语音引擎需要一个独立的 conda 环境**（见第 2 步）。

### 1. 获取语音模型文件

克隆仓库后，`voice/` 里缺少以下大文件，需自行下载放入对应位置：

| 文件 | 放入位置 | 来源 |
|---|---|---|
| `Plana-e15.ckpt`（GPT 模型） | `voice/models/` | 普拉娜专属音色模型，作者 bilibili@SLNeil |
| `Plana_e16_s208.pth`（SoVITS 模型） | `voice/models/` | 同上，作者 bilibili@SLNeil |
| `ref.wav`（参考音频） | `voice/models/` | 同上，作者 bilibili@SLNeil |
| `chinese-hubert-base` 模型 | `voice/plana_engine/pretrained_models/chinese-hubert-base/` | [HuggingFace: TencentGameMate/chinese-hubert-base](https://huggingface.co/TencentGameMate/chinese-hubert-base) |

> 💡 上面 3 个普拉娜专属模型是「普拉娜」这个音色的来源，由 **bilibili@SLNeil** 制作。个人备用存储链接：
> 链接: https://pan.baidu.com/s/1uW2DfQESnsDdDdXeMpMq3A?pwd=72e8 提取码: 72e8
>
> 如果你想要**别的角色**音色，可替换成你自己的 GPT-SoVITS 模型（GPT 的 `.ckpt`、SoVITS 的 `.pth`、参考音频 `ref.wav`），并在 `config.yaml` 里改 `ref_txt` 为该参考音频对应的文本。

### 2. 创建语音引擎的 conda 环境（Python 3.10）

```bash
conda create -n plana python=3.10 -y
conda activate plana
# CUDA 版 torch（版本需匹配你的显卡驱动/系统 CUDA）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy soundfile librosa transformers einops pytorch-lightning \
    matplotlib pandas torchmetrics typeguard regex pyopenjtalk gruut
```

验证环境可用：

```bash
conda activate plana
python -c "import torch; print(torch.cuda.is_available())"   # 应输出 True
```

### 3. 创建机器人的 Python 环境（Python 3.12，无需 torch）

```bash
cd ~/developer/plana_chat          # 换成你的项目路径
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

系统还需 `ffmpeg`（含 libopus 编码器）：`sudo apt install ffmpeg`（或确认 `ffmpeg -encoders | grep libopus` 有输出）。

还需本地 embedding 模型（用于「语义记忆」，让机器人能按语义回忆过往对话细节）：

```bash
ollama pull bge-m3    # 中文/多语种语义检索模型，约 1.2GB
```

### 4. 配置 `.env`

```bash
cp .env.example .env
# 编辑 .env：
#   TELEGRAM_BOT_TOKEN=123456789:xxxxxx    # 第 5 步创建机器人后拿到
#   DEEPSEEK_API_KEY=sk-xxxx               # 从 platform.deepseek.com 申请；用本地 Ollama 可留空
```

### 5. 配置 `config.yaml`

最少需要确认/修改这几处：

- `tts.python`：改成你的 conda 环境 python 路径（`which python` 查看，默认是 miniconda 下的 `plana` 环境）
- `tts.*` 路径：默认指向本项目的 `voice/`，一般无需改
- `llm.model`：DeepSeek 模型名（报「模型不存在」可改为 `deepseek-chat`）
- `telegram.proxy`：填**你自己的**代理地址（`http://127.0.0.1:7897` 是作者 Clash Verge 的默认端口，你的可能不同）
- `greeting.chat_ids`：改成你自己的 Telegram chat_id（见下文「主动问候」）

完整配置项说明见下方「配置说明」。

### 6. 创建 Telegram 机器人

1. 在 Telegram 搜索 **@BotFather**，发 `/newbot`
2. 起个名字（如「普拉娜」），用户名以 `bot` 结尾（如 `plana_mybot`）
3. 拿到 token，填入 `.env` 的 `TELEGRAM_BOT_TOKEN`

### 7. 验证

```bash
source .venv/bin/activate
python scripts/test_core.py        # 核心逻辑回归测试（不依赖外部服务，秒级）
python scripts/verify_llm.py       # 对话引擎：DeepSeek / Ollama 各回一条
python scripts/verify_tts.py "おはようございます"   # 语音：直接合成一句日语，产出 wav
```

`test_core.py` 覆盖情感标签清洗、记忆读写、好感度/关系阶段等纯逻辑；`verify_tts.py` 输出 wav 路径和时长即成功（首次会加载模型，约 1 分钟）。

> ⚠️ 顺序注意：`verify_llm.py` 的降级测试会加载 Ollama `qwen2.5:14b`（约 7GB）到显存，紧接着跑 `verify_tts.py` 可能因显存不足失败——这是 8GB 显存下的正常争抢，先 `ollama stop qwen2.5:14b` 释放即可。

### 8. 安装 `plana` 命令并启动

```bash
bash scripts/setup.sh      # 把 bin/ 加入 PATH（写入 ~/.bashrc）
source ~/.bashrc           # 或重开终端

plana on                   # 后台启动机器人
plana trace               # 观察后台日志（非侵入：Ctrl+C 只停跟随，不停后台）
plana status               # 查看运行状态
plana off                  # 停止并释放语音引擎
plana quiet                # 切换 quiet 模式（免打扰）
plana active               # 切换 active 模式（总是追问）
plana default              # 切换 default 模式（追问动态决策）
plana edit                 # 编辑记忆（需先 plana off；与 bot 互斥运行）
plana test                 # 测试网络与各依赖服务连通性
plana system_on            # 设为 systemd 服务（开机自启，需 sudo）
plana system_off           # 取消开机自启（需 sudo）
plana help                 # 查看所有指令
```

启动后在 Telegram 给机器人发消息，即可收到**文字 + 语音**回复。

## 配置说明（config.yaml）

| 配置段 | 关键项 | 说明 |
|---|---|---|
| `llm` | `provider` | `deepseek`（默认）或 `ollama` |
| | `model` / `base_url` | 云端模型名 / API 地址 |
| | `fallback_to_local` | 云端失败时是否自动降级本地 Ollama |
| `ollama` | `model` | 本地模型（需先 `ollama pull`） |
| `translator` | `provider` / `ollama_model` | 中日翻译用哪个引擎 |
| `tts` | `python` / `server_script` | 语音引擎的 python 路径与脚本 |
| | `gpt_path` / `vits_path` / `ref_wav` / `ref_txt` | 语音模型与参考音频 |
| `character` | `persona_file` / `holidays_file` | 人格 prompt 文件 / 节日配置（早上问候时提示） |
| `memory` | `rounds` | 每用户保留最近 N 轮 |
| | `compact_threshold` | 未整理消息数超该值触发「总结压缩 + 事实抽取」 |
| | `embedding_model` / `retrieve_top_k` | 语义检索用的 embedding 模型 / 注入的历史条数 |
| | `topic_fresh_hours` / `topic_idle_minutes` / `resume_topic_limit` / `topic_max_chars` | 挂起话题「新鲜」时间窗（小时，默认 1）/ 话题静默超时（分钟，默认 60，超时自动结算）/ 每次拾起最多考察几条（默认 8）/ 话题总结长度上限（字，默认 80） |
| | `open_item_short_hours` / `open_item_max_hours` | 未完成事项的短时/长时完成时长（小时，默认 2 / 72） |
| `reply` | `mode` | `text_voice`（文字+语音，默认）/ `voice` / `text` |
| `telegram` | `token_env` / `proxy` / `allow_user_ids` / `send_retries` | token 环境变量名 / 代理地址 / 用户白名单（空=不限制）/ 发送失败重试次数（默认 3） |
| `greeting` | `enabled` / `times` / `chat_ids` / `reminder_minutes` / `meal_duration_hours` / `recent_minutes` | 主动问候开关 / 时刻 / 目标用户 / 重新提醒间隔（分钟）/ 吃饭动作持续时间（小时，默认 2，超过则视为已吃完）/ 最近多少分钟内有对话则跳过问候（默认 60） |
| `behavior` | `mode` / `follow_up` / `settle_delay` / `self_review` / `affinity_scale` | 运行模式 / 话题续接开关 / 发送前等待（秒）/ 文本与情感自检开关 / 好感度缩放系数（默认1） |
| `emotion` | `enabled` / `image_dir` | 表情图开关（默认关）/ 表情图目录（见下） |
| `search` | `enabled` | 智能搜索开关（默认关） |

## 功能详解

### 主动问候

`greeting.enabled: true` 时，每天在 `times` 列出的时刻（北京时间，默认 8:00/11:00/21:00）主动向 `chat_ids` 里的用户发问候。问候是**上下文感知**的：最近 `recent_minutes`（默认 60）分钟内有进行中对话则跳过；否则把最近对话喂给 LLM，按情境演绎（已说晚安→轻声自语「睡着了」、生病→关心等）。还会尊重「同居状态」与时间：未同居时问候是发消息、不写成同处一室的肢体动作；老师已「下班/到家」后，晚上问候改用「辛苦了/到家了吗」而非回顾「今天过得怎么样」。文案不固定，若有「未完成的事」会顺带关心；**早上若遇到节日/纪念日，会带一句应景祝福（纪念日优先于节日）。**

**重新提醒**：若你对问候的反应是「要加班/要贪睡」，普拉娜会挂起一个提醒任务，默认 `reminder_minutes`（20）分钟后自动再提醒一次；期间你发任意消息则取消提醒，等待期间她不会主动发起话题。

`chat_ids` 是你和机器人的聊天 ID——先正常给机器人发一条消息，然后看日志（`data/plana.log` 里的 `chat=xxxx`）就能拿到。

### 情感表情图

机器人每次回复会带情感标签（如 `[emotion:sheepish]`）。开启 `emotion.enabled: true` 后，若 `assets/emotions/<情感名>/` 文件夹里有图片，就随机抽一张发给你；没有则不发（**默认关闭**）。

已预建好 11 个情绪文件夹：`neutral` `sheepish` `love` `sulky` `jealous` `expectant` `dumbfounded` `delighted` `cheerful` `puzzled` `like`。把图片（png/jpg/webp）直接丢进对应文件夹即可。

### 话题系统（详细架构）

机器人把对话组织成「话题」来管理。下面是完整的状态、执行流、每个分支与 LLM 请求。

#### 核心状态

| 状态 | 存储 | 说明 |
|---|---|---|
| `FIFO` | 内存队列 | 老师发来、尚未处理的消息，按 `update_id` 顺序 |
| `current_topic` | SQLite | 当前话题（一句话）+ 类型（`normal` / `emotion`）|
| `topic_emotions` | SQLite（current_topic.emotions）| 当前话题累积的情感标签，随话题重置；**持久化，意外退出不丢** |
| `topics` | topics 表 | 挂起话题：普通话题（可反复挂起）+ 吃醋事件（一次性）|
| 记忆 | messages/summaries/facts | 不变 |
| `relationship` | relationship 表 | **已结算**好感度 + 阶段 |

#### 主循环（FIFO 逐条）

```
长轮询拉取 updates → 按 update_id 顺序 push 进 FIFO → 确认 offset（持久化到 data/offset）
while FIFO 非空:
    取一条 → 处理单条（见下）
```

#### 单条消息处理流程（阶段 0~7）

**阶段 0 · 落库**：老师消息立即落库（只存文本，embedding 延迟到阶段 2 计算后回填），不依赖后续是否发送。

> 落库在前，因此阶段 1（话题判断）、阶段 2（主回复生成）和回复自检在拼装「历史上下文」时，会把**本轮刚写入的老师消息从末尾剔除**，只作为「当前消息」单独传入一次，避免同一条消息在上下文中重复出现（否则短句会被模型误读成“老师又说了一遍”）。

**阶段 0.5 · 静默超时结算**（`memory.topic_idle_minutes`，默认 60 分钟）：
- 老师发消息时，检查当前话题距最后一条消息是否超过阈值
- 超时则自动结算：普通话题 → 挂起 + 结算好感度 + 清空；吃醋话题 → 安抚结算（+2/-10）+ 清空（不挂起）
- 未超时 → 跳过

**阶段 1 · ①判断**（一次 LLM，输出四字段 `has_content / same / topic / old_context`）：
- 输入：`user_text` + `current_topic`
- `topic` 是**当前正在聊的「这一件事」**的一句话总结（起因经过结果，`topic_max_chars` 字以内）；`same` 判断「是否还在聊同一件事」，不是「同一个大主题」——中午吃饭和晚上散步应判成两件事
- 分支：
  - `has_content = false`（无营养「嗯嗯」）→ **静默跳过**，不生成、不回复
  - `same = false`（切话题）→ 处理旧话题：
    - 普通话题 → 挂起（含 `old_context` 起因经过结果 + 未完成部分）；**已自然收束、无未完成部分则不挂起**
    - 吃醋话题 → 安抚结算（+2/-10）+ 移除
    - 结算旧话题好感度
  - 无论是否切话题 → `current_topic = topic`（**当场更新**，这是「连发切话题不丢话题」的关键）

**阶段 2 · 生成回复**（纯生成，零状态副作用）：
- 主回复生成（LLM，云端失败降级本地）
- 回复自检 `self_review`（LLM）：矛盾 / 不合时宜 / 不符性格 / 重复·时态错乱
- 情感自检 `_emotion_review`（LLM）：标签不符阶段则修正
- **申请判断**（LLM，除结婚外每轮）：是否明确表白/求婚
- 命中 `reject / 恋爱 / 结婚` → **特殊时刻重生成**（LLM，reject 也重生成）
- 此阶段不碰任何数值状态

**阶段 3 · 发送前查 FIFO**（短轮询拉新消息，分叉）：
- 无新消息 → 继续
- 新消息无营养 → 忽略，继续发送当前回复
- 新消息同话题 → **合并**（`pending` 累积，标记「尚未回复」只加一次、不嵌套），回到阶段 2 重新生成
- 新消息不同话题 → **放弃当前回复**（状态未动），回主循环处理新消息

**阶段 4 · 状态修改**（确认发送后一次性执行）：
- 累积情感（立即持久化到 `current_topic.emotions`）
- 吃醋挂起（sulky/jealous → 挂起，带老师原话缘由，去重）
- 申请升级（阶段 2 命中升级 → `set_stage` + milestone）
- 提醒检测（加班/睡 → 挂起 reminder）
- 话题过期（>3 天普通话题关闭）

**阶段 5 · 语音 + 存储 + 发送**：生成语音 → 存储回复 → 发送文字/语音/表情。

**阶段 6 · 续接判断**（一次 LLM，二态 `finished / continuing`）：
- `continuing` → 续一句（发送前也查 FIFO，有新消息则放弃续接回主循环）；续句会避开与主回复重复、也避免在已问过老师问题后再反问「你呢 / 老师呢」
- `finished` → 结算好感度 + 拾起挂起话题（发送前也查 FIFO）

**阶段 7 · 好感度结算**（话题级）：
- 触发：阶段 6 判 `finished`，或下一轮阶段 1 判切话题，或静默超时
- 公式：`delta = (情感平均分·绝对值向上取整 + 阶段基础分) × affinity_scale`
  - 例：一个话题 like + neutral，平均 1 → 情感分 1 → 陌生阶段 (1+1)×1 = +2

#### 申请检测（判断与升级分离）

- **判断**（阶段 2，纯判断 + 重生成）：除结婚外每轮判断是否表白/求婚，达标用「已结算好感度」；命中则重生成特殊回复
- **升级**（阶段 4，状态修改）：`set_stage` + 拉到 70（恋爱）等，推迟到确认发送后

#### 挂起话题的三层拾起优先级

`memory.topic_fresh_hours`（默认 1 小时），SQL 排序：

| 层 | 条件 | 内部排序 |
|---|---|---|
| 第 1 层 | 话题发生时间在 N 小时内（无论类型）| 时间倒序 |
| 第 2 层 | 超过 N 小时的吃醋事件 | 时间倒序 |
| 第 3 层 | 超过 N 小时的普通话题 | 时间倒序 |

> 话题的 `created_at` 记录的是**事件实际发生时间**：切话题挂起时=当前时刻；静默超时挂起时=该话题最后一条消息的时间。拾起判断里把「当前时间 vs 话题发生时间」作对比，据此识别话题是否已过时（`suitable`/`stale`），避免把「中午吃饭」这类早已过去的事误判成「刚刚发生、适合提起」。

拾起时逐条判断三态 `suitable / stale / not_now`（一次 LLM 每条，最多 `resume_topic_limit` 条，默认 8）：
- `suitable` → 生成提起文案 → 成为 current_topic → 关闭该话题（普通话题的 current_topic 记为「提起文案 + 原话题」，兼顾回应提起文案或回应原话题；抽取原始话题作根，避免多层「原话题」嵌套）
- `stale` → 关闭（已过时）
- `not_now` → 保留（氛围不对，等下次）

#### 吃醋/黑化事件生命周期

```
触发(sulky/jealous) → 挂起(带老师原话缘由)
  → finished 时拾起成为 current_topic(kind=emotion)
  → 老师切走或静默超时 → 安抚结算(+2/-10) + 移除（不挂起）
对话期间不产生新吃醋事件（has_active_emotion 去重）
```

#### 每轮 LLM 调用

**固定 6 次**：①判断 + 主回复 + 自检 + 情感自检 + 申请判断 + 续接判断
**条件触发**：翻译 / 搜索 / 提醒检测 / 申请重生成 / 记忆整理 / 拾起合适性 / 合并重生成，最多再加 ~4 次

### 三种运行模式

机器人有三种运行模式（`behavior.mode`，默认 `default`）：

| 模式 | 主动问候 | 话题续接 |
|---|---|---|
| `default`（默认）| ✅ | 由 LLM 判断话题是否结束（结束则结算好感度并拾起、未结束续一句）|
| `active` | ✅ | 每轮都续接 |
| `quiet`（免打扰）| ❌ | 不续接 |

切换方式（控制指令，机器人不会回复）：
- 聊天中：`/quiet` `/active` `/default`
- PC 端：`plana quiet` / `plana active` / `plana default`（约 30 秒内生效）
- `plana status` 可查看当前模式、好感度、关系阶段、挂起话题

另外两个相关机制：

- **回复自检 + 情感自检**：`self_review: true` 时，发送前让 LLM 分别审查「文本是否与上下文矛盾」与「情感标签是否符合关系阶段」，有问题则修正（各多一次轻量 LLM 调用）
- **拾起挂起话题**：话题 `finished` 后，先判断老师是否方便，再按三层优先级对挂起话题逐条判断「是否仍合适」（综合时效性 + 当前氛围，最多 `resume_topic_limit` 条）——合适则提起并关闭、过时则关闭、暂不合适则保留

### 智能搜索

`search.enabled: true` 时，问「今天天气/最新新闻/股价」这类实时或事实性问题，会先联网搜索（DuckDuckGo，走 `telegram.proxy`）再把结果喂给 LLM 回答。默认关闭。

### 动态认知系统（好感度 + 关系阶段）

普拉娜对你有好感度（0 起），**按话题结算**：每个话题结束时（聊完或切走），把话题内累积的情感标签取平均，加上阶段基础分，再乘缩放系数 `affinity_scale`（默认 1）：

| 情感 | 情感分 |
|---|---|
| love | +3 |
| like / delighted | +2 |
| sheepish / cheerful / expectant | +1 |
| neutral / puzzled / sulky | 0 |
| dumbfounded / jealous | -1 |

阶段基础分：**陌生 +1、熟悉 +0**（每话题一次，仅陌生阶段轻微升温；喜欢/恋爱/结婚阶段基础分为 0）。

> 每话题结算 = (情感平均分·绝对值向上取整 + 阶段基础分) × `affinity_scale`
> 例：一个话题里 like + neutral，平均 1 → 情感分 1 → 陌生阶段 (1 + 1) × 1 = +2

好感度决定关系阶段：

```
陌生(0) → 熟悉(10) → 喜欢(30) → 恋爱(60) → 结婚(100)
```

- 陌生→熟悉、熟悉→喜欢：**自动升级**
- 喜欢→恋爱、恋爱→结婚：需你**主动申请**（好感度达标后，额外判断你的话是否明确表达意愿；恋爱会直接升到 70「热恋期」）
- **喜欢阶段好感度封顶 60、恋爱阶段封顶 100**：确认关系（申请成功）前可降、不可再升过阈值，申请成功后才突破
- **任何阶段（除结婚）你表白/求婚都会被识别**：好感度不足时，普拉娜会委婉地「还没准备好」回应，而不是含糊带过；结婚后关系锁定，不再判断
- 除结婚外各阶段都可升降；**结婚后好感度锁定不降**
- 每个阶段有独立的 prompt（语气、动作、彩蛋、吃醋梯度都不同）
- 好感度**数值不进 prompt**（只有阶段与大致感觉）；`plana status` 可查看好感度、阶段、挂起话题

### 长期记忆

四层记忆（都在 `data/history.db`），解决「跨天记住你」+「精准回忆细节」：

1. **滚动历史**：最近 N 轮原始对话（当前上下文）
2. **总结压缩**：历史超阈值后，把旧消息压成「长期记忆摘要」常驻上下文；注入时带上「最近整理时间」，让模型能区分摘要里的内容是「今天」还是「以前」
3. **事实抽取**：抽出「关于你的事实」和「未完成事项」，每次对话和主动问候时注入。未完成事项按完成时长分三类：**短时**（吃饭/睡觉等，`open_item_short_hours` 小时后视为已完成）、**长时**（写论文/待办等，`open_item_max_hours` 小时后兜底过期）、**定时**（周五出差等有明确时间点，按估算 `due_hours` 到期），到期后不再注入
4. **语义检索**：所有原文**永久保留**并用本地 embedding（`bge-m3`）向量化；你问起过去某件事时，按语义检索出最相关的历史原文注入上下文，细节不失真

### 记忆编辑（手动修正）

记忆存在 `data/history.db`，可用 `plana edit` 手动查看/修改/删除：

- `plana edit show` — 记忆概览（关系 / 摘要 / 事实 / 话题 / 当前话题）
- `plana edit messages [N]` — 最近 N 条对话原文（带 id）
- `plana edit edit-message <id> <新文本>` / `del-message <id>` — 改 / 删原文
- `plana edit facts` / `del-fact <id>` — 列出 / 删除事实
- `plana edit summary` / `set-summary <文本>` — 查看 / 重写摘要
- `plana edit relationship` / `set-stage <阶段>` / `set-affinity <n>` — 关系
- `plana edit topics` / `close-topic <id>` / `clear-current` — 话题

**互斥**：编辑与 bot 不能同时进行。bot 在跑时 `plana edit` 会拒绝；编辑期间（持有 `data/edit.lock`）`plana on` 也会拒绝。修改/删除原文会连带清空其语义向量（embedding），该条不再参与语义检索，但仍留在原文历史里。

## 注意事项

- **代理**：Telegram 国内被墙，机器人服务器端和手机端都要能连上（`telegram.proxy` 填你自己的代理）
- **显存**：语音引擎约占 1GB 显存；若切本地 Ollama，注意与语音引擎的显存争抢，必要时 `ollama stop` 释放
- **容错降级**：DeepSeek 余额不足/不可用时，自动切本地 Ollama（对话用 qwen2.5、翻译用 translategemma），并在 `data/plana.log` 中以红色高亮提示（`tail -f data/plana.log` 可见）；`plana status` 可查看运行方式（systemd/后台进程）
- **清除记忆**：想重置普拉娜的记忆（相当于第一次见面），运行 `bash scripts/clear_memory.sh`——会先停机器人、需输入 `yes` 确认，然后清除记忆与日志，其余配置不变
- **音色**：本项目语音模型仅支持日语，中文回复会先翻译成日语再合成语音
- **账号风险**：用真实 Telegram 账号注册机器人（BotFather）本身无风险，但第三方代理需自行合规使用

## 致谢

- 角色与设定源自《蔚蓝档案》（Blue Archive）
- 语音引擎：GPT-SoVITS
- 普拉娜音色模型：bilibili@SLNeil
- 中文音频特征提取：TencentGameMate/chinese-hubert-base
- shittim（其实就是我，没想到吧）
