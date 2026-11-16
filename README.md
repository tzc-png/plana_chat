# plana_chat — 普拉娜 Telegram 二次元角色语音聊天机器人

让《蔚蓝档案》的「普拉娜」进驻你的 Telegram，用她的声音和你聊天。基于 shittim 已有的本地 GPT-SoVITS 日语语音合成，核心为常驻 Python 服务。

## 功能

- **角色扮演**：普拉娜人格（内向、依赖老师的少女），支持情感标签
- **本地语音**：复用 shittim 的 GPT-SoVITS 日语模型，本地合成，Telegram 语音消息（OGG/Opus）
- **多用户记忆**：每个 Telegram 用户独立的历史与上下文（SQLite）
- **引擎可切换**：云端 DeepSeek（默认，快/效果好）或本地 Ollama（隐私/免费），云端失败自动降级

## 架构

```
Telegram 消息 → 角色对话(LLM) → 情感标签解析/文本清洗 → 中日翻译 → 本地 TTS(GPT-SoVITS) → OGG 语音
```

核心（对话/翻译/语音）是**通道无关**的，未来接其他通道只需加一个适配器。

## 目录

```
plana_chat/
├── run_telegram.py           # Telegram 机器人入口
├── bin/plana                 # 命令行管理（on/off/system_on/system_off/status/help）
├── config.yaml               # 主配置
├── .env                      # 密钥（不提交）
├── plana_chat/
│   ├── telegram.py           # Telegram 适配器（httpx 长轮询）
│   ├── pipeline.py           # 完整链路编排
│   ├── character/plana.txt   # 人格 prompt
│   ├── llm/                  # DeepSeek / Ollama
│   ├── translator.py         # 中日翻译
│   ├── tts/                  # GPT-SoVITS 客户端 + 进程管理
│   ├── audio.py              # WAV→OGG(Opus)
│   ├── postprocess.py        # 情感标签 / 清洗
│   ├── memory.py             # SQLite 每用户历史
│   └── config.py             # 配置加载
└── scripts/                  # setup.sh / verify_llm.py / verify_tts.py
```

## 快速开始

### 1. 依赖

需要一个含 GPT-SoVITS 的 conda 环境（即 shittim 用的 `plana` 环境），以及一个轻量 Python 环境跑机器人（无需 torch）：

```bash
cd ~/developer/plana_chat
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

系统还需 `ffmpeg`（带 libopus）用于 WAV→OGG 转换。

### 2. 配置

- `.env` 填 `TELEGRAM_BOT_TOKEN`（见下）和 `DEEPSEEK_API_KEY`
- `config.yaml` 里 `tts.*` 路径默认指向 shittim；`telegram.proxy` 填本机代理（Clash Verge 默认 `http://127.0.0.1:7897`）

### 3. 创建 Telegram 机器人

1. 在 Telegram 搜索 **@BotFather**，发 `/newbot`
2. 起名（如「普拉娜」），用户名以 `bot` 结尾（如 `shittim_plana_bot`）
3. 拿到 token 填入 `.env` 的 `TELEGRAM_BOT_TOKEN`

### 4. 验证

```bash
python scripts/verify_llm.py     # DeepSeek + Ollama 各回一条
python scripts/verify_tts.py     # 中文→翻译→日语→WAV
```

### 5. 安装 `plana` 命令

```bash
bash scripts/setup.sh   # 把 bin/ 加入 PATH（写入 ~/.bashrc）
source ~/.bashrc        # 或重开终端
```

### 6. 用 `plana` 命令管理

```bash
plana on          # 后台启动机器人
plana status      # 查看运行状态
plana off         # 停止并释放语音引擎
plana system_on   # 安装并启用 systemd 服务（开机自启，需 sudo）
plana system_off  # 停止并禁用 systemd 服务（需 sudo）
plana help        # 查看所有指令
```

启动后在 Telegram 里给机器人发消息，即可收到文字 + 语音回复。也可以不经 `plana` 直接 `source .venv/bin/activate && python run_telegram.py` 前台运行。

## 注意事项

- **代理**：Telegram 国内被墙，机器人和手机端都需代理（本机 Clash Verge 已配置 `127.0.0.1:7897`）
- **显存**：8GB 显卡下默认云端 LLM 时 GPU 主要留给 TTS；若切本地 Ollama 需先 `ollama stop` 释放显存
- **音色**：普拉娜模型仅支持日语，文字回复（中文）会先翻译成日语再合成语音
- **QQ 已弃用**：原 NapCatQQ + NoneBot2 方案因 NapCatQQ 于 2026-11 停更、QQ 第三方协议生态崩塌而移除，核心代码保留通道无关设计，后续可加其他通道适配器
