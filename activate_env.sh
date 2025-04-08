#!/bin/bash
# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# 切换到项目目录
cd "$SCRIPT_DIR"
# 激活虚拟环境
source "$SCRIPT_DIR/.venv/bin/activate"
echo "统一项目环境已激活！"
echo "使用 deactivate 命令退出虚拟环境"
