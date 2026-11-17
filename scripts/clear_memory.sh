#!/usr/bin/env bash
# 清除普拉娜的记忆与日志（相当于第一次见到你）
#
# 这不是 plana 的子命令，需手动到项目 scripts/ 目录运行，防止误删。
# 会先停止机器人，再清除记忆与日志；config、人格、模式设置、表情图等都不动。

set -u

PLANA_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PLANA_ROOT/data"

echo "将清除："
echo "  记忆：$DATA_DIR/history.db"
echo "  日志：$DATA_DIR/plana.log"
echo "（config.yaml、人格、模式设置、表情图等均保留）"
echo
read -r -p "确认清除？输入 yes 继续： " confirm
if [ "$confirm" != "yes" ]; then
    echo "已取消。"
    exit 0
fi

# 停止机器人与语音引擎
"$PLANA_ROOT/bin/plana" off >/dev/null 2>&1
pkill -f "run_telegram.py" 2>/dev/null || true
pkill -f "plana_tts_server.py" 2>/dev/null || true
rm -f /tmp/plana_tts.sock
sleep 1

# 清除记忆与日志
rm -f "$DATA_DIR"/history.db "$DATA_DIR"/history.db-* "$DATA_DIR"/plana.log

echo
echo "已清除。下次 plana on 启动后，普拉娜将不记得之前的对话，相当于第一次见到你。"
