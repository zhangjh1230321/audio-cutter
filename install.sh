#!/bin/bash
# AutoCut 安装脚本

set -e

echo "=== AutoCut 安装脚本 ==="
echo ""

# 检查操作系统
if [[ "$(uname)" != "Darwin" ]]; then
    echo "警告: 此脚本针对 macOS 优化，其他系统可能需要调整"
fi

# 安装 Homebrew (如果未安装)
if ! command -v brew >/dev/null 2>&1; then
    echo "需要安装 Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

echo "1/3 安装系统依赖..."
brew install ffmpeg sox

echo ""
echo "2/3 安装 Python 依赖 (可选，但推荐)..."
echo "   需要 moviePy 进行高级功能 (批量处理、可视化等)"
read -p "   安装 Python 依赖? [y/N] " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # 检查 Python
    if ! command -v python3 >/dev/null 2>&1; then
        echo "需要安装 Python 3"
        brew install python
    fi
    
    pip3 install moviePy pydub matplotlib
    echo "   Python 依赖安装完成"
else
    echo "   跳过 Python 依赖，仅使用 Shell 脚本版本"
fi

echo ""
echo "3/3 创建快捷命令..."

# 创建别名
ALIAS_LINE="alias autocut='$(dirname "$0")/autocut.sh'"
PROFILE_FILE=""

# 检测 shell 配置文件
for file in ~/.zshrc ~/.bashrc ~/.bash_profile; do
    if [[ -f "$file" ]]; then
        PROFILE_FILE="$file"
        break
    fi
done

if [[ -z "$PROFILE_FILE" ]]; then
    PROFILE_FILE="$HOME/.zshrc"
fi

# 检查是否已存在别名
if ! grep -q "alias autocut=" "$PROFILE_FILE" 2>/dev/null; then
    echo "" >> "$PROFILE_FILE"
    echo "# AutoCut" >> "$PROFILE_FILE"
    echo "$ALIAS_LINE" >> "$PROFILE_FILE"
    echo "已添加别名到 $PROFILE_FILE"
    echo "请运行: source $PROFILE_FILE"
else
    echo "别名已存在"
fi

chmod +x "$(dirname "$0")/autocut.sh"
chmod +x "$(dirname "$0")/autocut.py" 2>/dev/null || true

echo ""
echo "=== 安装完成 ==="
echo ""
echo "使用方法:"
echo "  autocut video.mp4                    # 基础剪辑"
echo "  autocut video.mp4 --analyze-only     # 仅分析气口"
echo "  autocut video.mp4 -t -40 -g 1.0       # 自定义参数"
echo ""
