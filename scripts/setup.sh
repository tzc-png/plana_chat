#!/usr/bin/env bash
# 把 <项目根>/bin 加入 PATH（写入 ~/.bashrc），使 `plana` 命令全局可用。
#
# 用法：bash scripts/setup.sh
set -e

BIN_DIR="$(cd "$(dirname "$0")/../bin" && pwd)"
MARK="# >>> plana-chat bin >>>"

if grep -qF "$MARK" "$HOME/.bashrc" 2>/dev/null; then
    echo "~/.bashrc 已配置过 plana-chat，跳过。"
else
    {
        echo ""
        echo "$MARK"
        echo "export PATH=\"$BIN_DIR:\$PATH\""
        echo "# <<< plana-chat bin <<<"
    } >> "$HOME/.bashrc"
    echo "已把 $BIN_DIR 加入 ~/.bashrc 的 PATH。"
fi

echo ""
echo "让命令立即生效，请执行以下任一操作："
echo "  source ~/.bashrc"
echo "  # 或直接在当前终端："
echo "  export PATH=\"$BIN_DIR:\$PATH\""
echo ""
echo "之后即可使用：plana help"
