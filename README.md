# plana_chat — 普拉娜 Telegram 二次元角色语音聊天机器人

让《蔚蓝档案》的「普拉娜」进驻你的 Telegram，用她的声音和你聊天。

本项目是一个**自包含**的常驻服务：本地 GPT-SoVITS 语音合成 + 大模型角色扮演（云端或本地），全程跑在你自己的机器上。

## 功能

- **角色扮演**：普拉娜人格（系统 prompt 可换），回复带情感标签
- **本地语音**：GPT-SoVITS 日语模型本地合成，Telegram 语音消息（OGG/Opus）
- **长期记忆**：每个用户独立的 SQLite 记忆，含「滚动历史 + 总结压缩 + 事实抽取 + 语义检索」四层，能跨天记住你的喜好、未完成的事，并能按语义精准回忆过往对话细节
- **主动问候**：早上 8 点、晚上 10 点主动发问候（文案不固定，并会带一句你之前未完成事项的关心）
- **情感表情图**：按回复情感自动发对应立绘表情图（可自备图片）
- **智能搜索**（可选）：问实时/事实性问题时联网搜索再回答
- **引擎可切换**：云端 DeepSeek（默认）或本地 Ollama，云端失败自动降级
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
| `Plana-e15.ckpt`（GPT 模型） | `voice/models/` | 普拉娜专属，来自 shittim 项目（百度网盘，链接可能失效，见其 README） |
| `Plana_e16_s208.pth`（SoVITS 模型） | `voice/models/` | 同上 |
| `ref.wav`（参考音频） | `voice/models/` | 同上 |
| `chinese-hubert-base` 模型 | `voice/plana_engine/pretrained_models/chinese-hubert-base/` | [HuggingFace: TencentGameMate/chinese-hubert-base](https://huggingface.co/TencentGameMate/chinese-hubert-base) |

> 💡 这 3 个普拉娜专属模型是「普拉娜」这个音色的来源。如果你想要**别的角色**音色，可替换成你自己的 GPT-SoVITS 模型（GPT 的 `.ckpt`、SoVITS 的 `.pth`、参考音频 `ref.wav`），并在 `config.yaml` 里改 `ref_txt` 为该参考音频对应的文本。

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
python scripts/verify_llm.py        # 对话引擎：DeepSeek / Ollama 各回一条
python scripts/verify_tts.py "おはようございます"   # 语音：直接合成一句日语，产出 wav
```

`verify_tts.py` 输出 wav 路径和时长即成功（首次会加载模型，约 1 分钟）。

### 8. 安装 `plana` 命令并启动

```bash
bash scripts/setup.sh      # 把 bin/ 加入 PATH（写入 ~/.bashrc）
source ~/.bashrc           # 或重开终端

plana on                   # 后台启动机器人
plana status               # 查看运行状态
plana off                  # 停止并释放语音引擎
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
| `character` | `persona_file` | 人格 prompt 文件（可换成 `plana_compromise.txt`） |
| `memory` | `rounds` | 每用户保留最近 N 轮 |
| | `compact_threshold` | 未整理消息数超该值触发「总结压缩 + 事实抽取」 |
| | `embedding_model` / `retrieve_top_k` | 语义检索用的 embedding 模型 / 注入的历史条数 |
| `reply` | `mode` | `text_voice`（文字+语音，默认）/ `voice` / `text` |
| `telegram` | `token_env` / `proxy` | token 的环境变量名 / 代理地址 |
| `greeting` | `enabled` / `times` / `chat_ids` | 主动问候开关 / 时刻 / 目标用户 |
| `emotion` | `image_dir` | 表情图目录（见下） |
| `search` | `enabled` | 智能搜索开关（默认关） |

## 功能详解

### 主动问候

`greeting.enabled: true` 时，每天在 `times` 列出的时刻（北京时间）主动向 `chat_ids` 里的用户发问候。文案由 LLM 即时生成（不固定），若你有「未完成/没聊完的事」，会顺带自然关心一句。

`chat_ids` 是你和机器人的聊天 ID——先正常给机器人发一条消息，然后看日志（`data/plana.log` 里的 `chat=xxxx`）就能拿到。

### 情感表情图

机器人每次回复会带情感标签（如 `[emotion:sheepish]`），若 `assets/emotions/<情感名>/` 文件夹里有图片，就随机抽一张发给你；没有则不发。

已预建好 11 个情绪文件夹：`neutral` `sheepish` `love` `sulky` `jealous` `expectant` `dumbfounded` `delighted` `cheerful` `puzzled` `like`。把图片（png/jpg/webp）直接丢进对应文件夹即可。

### 智能搜索

`search.enabled: true` 时，问「今天天气/最新新闻/股价」这类实时或事实性问题，会先联网搜索（DuckDuckGo，走 `telegram.proxy`）再把结果喂给 LLM 回答。默认关闭。

### 长期记忆

四层记忆（都在 `data/history.db`），解决「跨天记住你」+「精准回忆细节」：

1. **滚动历史**：最近 N 轮原始对话（当前上下文）
2. **总结压缩**：历史超阈值后，把旧消息压成「长期记忆摘要」常驻上下文
3. **事实抽取**：抽出「关于你的事实」和「未完成事项」，每次对话和主动问候时注入
4. **语义检索**：所有原文**永久保留**并用本地 embedding（`bge-m3`）向量化；你问起过去某件事时，按语义检索出最相关的历史原文注入上下文，细节不失真

## 注意事项

- **代理**：Telegram 国内被墙，机器人服务器端和手机端都要能连上（`telegram.proxy` 填你自己的代理）
- **显存**：语音引擎约占 1GB 显存；若切本地 Ollama，注意与语音引擎的显存争抢，必要时 `ollama stop` 释放
- **音色**：本项目语音模型仅支持日语，中文回复会先翻译成日语再合成语音
- **账号风险**：用真实 Telegram 账号注册机器人（BotFather）本身无风险，但第三方代理需自行合规使用

## 致谢

- 角色与语音源自 shittim 项目（蔚蓝档案「什亭之匣」主题）
- 语音引擎：GPT-SoVITS
- 普拉娜音色模型：bilibili@SLNeil
