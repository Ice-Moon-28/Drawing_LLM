#!/bin/bash

# 设置环境变量（macOS 下用来定位 cairo 库）
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:/opt/homebrew/share/pkgconfig:$PKG_CONFIG_PATH"

export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_FALLBACK_LIBRARY_PATH"

# 指定文件名和下载 URL
FILE="sac+logos+ava1-l14-linearMSE.pth"
URL="https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/refs/heads/main/sac+logos+ava1-l14-linearMSE.pth"

# 检查文件是否存在，如果存在则跳过下载
if [ -f "$FILE" ]; then
    echo "文件 $FILE 已存在，跳过下载。"
else
    echo "文件 $FILE 不存在，开始下载..."
    wget "$URL"
fi

poetry run python main.py