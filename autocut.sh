#!/bin/bash
# AutoCut - 自动剪辑气口脚本

set -e

# 默认参数
THRESHOLD="-30"
MIN_GAP="0.1"
MERGE="0.3"
OUTPUT=""
ANALYZE_ONLY=false
VERBOSE=false

show_help() {
    cat << EOF
AutoCut - 自动剪辑气口

用法: autocut <视频路径> [选项]

选项:
    -t, --threshold <dB>    静音阈值 (默认: -30, 范围: -60 到 -20)
    -g, --min-gap <秒>     最小气口时长 (默认: 0.5)
    -m, --merge <秒>        合并间隔 (默认: 0.3)
    -o, --output <路径>     输出目录 (默认: 同目录)
    -a, --analyze-only     仅分析，不剪辑
    -v, --verbose          显示详细输出
    -h, --help             显示帮助

示例:
    autocut video.mp4
    autocut video.mp4 --threshold -40 --min-gap 1.0
    autocut video.mp4 --analyze-only
EOF
    exit 0
}

log() {
    echo "[AutoCut] $1"
}

error() {
    echo "[AutoCut] ERROR: $1" >&2
    exit 1
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -g|--min-gap)
            MIN_GAP="$2"
            shift 2
            ;;
        -m|--merge)
            MERGE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -a|--analyze-only)
            ANALYZE_ONLY=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        -*)
            error "未知参数: $1"
            ;;
        *)
            VIDEO_PATH="$1"
            shift
            ;;
    esac
done

# 检查依赖
command -v ffmpeg >/dev/null 2>&1 || error "需要安装 ffmpeg: brew install ffmpeg"
command -v sox >/dev/null 2>&1 || log "提示: 安装 sox 可获得更精确的检测: brew install sox"

# 检查视频文件
[[ -z "$VIDEO_PATH" ]] && error "请提供视频文件路径"
[[ ! -f "$VIDEO_PATH" ]] && error "视频文件不存在: $VIDEO_PATH"

# 获取绝对路径
VIDEO_PATH="$(cd "$(dirname "$VIDEO_PATH")" && pwd)/$(basename "$VIDEO_PATH")"
BASENAME="${VIDEO_PATH%.*}"
EXTENSION="${VIDEO_PATH##*.}"

# 设置输出目录
if [[ -z "$OUTPUT" ]]; then
    OUTPUT="${BASENAME}_autocut"
fi
mkdir -p "$OUTPUT"

log "开始处理: $VIDEO_PATH"
log "静音阈值: ${THRESHOLD}dB, 最小气口: ${MIN_GAP}s"

# 提取音频
AUDIO_PATH="${OUTPUT}/audio.wav"
log "提取音频..."
ffmpeg -y -i "$VIDEO_PATH" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$AUDIO_PATH" 2>/dev/null

# 检测静音区域
SILENCE_FILE="${OUTPUT}/silence.txt"
if command -v sox >/dev/null 2>&1; then
    log "使用 SoX 检测气口..."
    sox "$AUDIO_PATH" -n silence 1 0.1 "$THRESHOLD" 0.1 "$THRESHOLD" > "$SILENCE_FILE" 2>&1
else
    log "使用 FFmpeg 检测气口..."
    ffmpeg -y -i "$AUDIO_PATH" -af "silencedetect=noise=${THRESHOLD}dB:d=${MIN_GAP}" -f null - 2>&1 | grep "silence_end" > "$SILENCE_FILE"
fi

if [[ "$VERBOSE" == true ]]; then
    log "静音检测结果:"
    cat "$SILENCE_FILE"
fi

# 分析气口位置
log "分析气口位置..."

# 解析 silence_end 时间戳
declare -a GAP_STARTS
declare -a GAP_ENDS

while IFS= read -r line; do
    if [[ "$line" == *"silence_end"* ]]; then
        # 提取时间戳 (格式: silence_end: 123.45)
        END_TIME=$(echo "$line" | sed -E 's/.*silence_end: ([0-9]+\.?[0-9]*).*/\1/')
        GAP_ENDS+=("$END_TIME")
    elif [[ "$line" == *"silence_start"* ]]; then
        START_TIME=$(echo "$line" | sed -E 's/.*silence_start: ([0-9]+\.?[0-9]*).*/\1/')
        GAP_STARTS+=("$START_TIME")
    fi
done < "$SILENCE_FILE"

# 计算总气口时长
TOTAL_GAP=0
for i in "${!GAP_ENDS[@]}"; do
    START="${GAP_STARTS[$i]}"
    END="${GAP_ENDS[$i]}"
    DURATION=$(awk "BEGIN {printf \"%.3f\", $END - $START}")
    TOTAL_GAP=$(awk "BEGIN {printf \"%.3f\", $TOTAL_GAP + $DURATION}")
done

log "检测到 ${#GAP_ENDS[@]} 个气口，总时长: ${TOTAL_GAP}s"

if [[ "$ANALYZE_ONLY" == true ]]; then
    log "分析完成（仅分析模式）"
    echo ""
    echo "=== 气口列表 ==="
    for i in "${!GAP_STARTS[@]}"; do
        START="${GAP_STARTS[$i]}"
        END="${GAP_ENDS[$i]}"
        DURATION=$(awk "BEGIN {printf \"%.3f\", $END - $START}")
        printf "  气口 %2d: %6.2fs - %6.2fs (%.3fs)\n" $((i+1)) "$START" "$END" "$DURATION"
    done
    rm -rf "$OUTPUT"
    exit 0
fi

# 生成剪辑时间戳（保留气口前后的小缓冲）
log "生成剪辑片段..."

# 创建片段列表
declare -a CUT_STARTS
declare -a CUT_ENDS

PREV_END=0
for i in "${!GAP_STARTS[@]}"; do
    GAP_START="${GAP_STARTS[$i]}"
    GAP_END="${GAP_ENDS[$i]}"
    
    # 计算片段边界（去除气口，留出 0.1s 缓冲）
    CLIP_START="$PREV_END"
    CLIP_END="$GAP_START"
    
    if (( $(awk "BEGIN {print ($CLIP_END - $CLIP_START > 0.3)}") )); then
        CUT_STARTS+=("$CLIP_START")
        CUT_ENDS+=("$CLIP_END")
    fi
    
    PREV_END="$GAP_END"
done

# 添加最后一个片段
DURATION=$(ffmpeg -i "$VIDEO_PATH" 2>&1 | grep "Duration" | cut -d' ' -f4 | tr -d , | awk -F: '{print $1*3600+$2*60+$3}')
if (( $(awk "BEGIN {print ($DURATION - $PREV_END > 0.3)}") )); then
    CUT_STARTS+=("$PREV_END")
    CUT_ENDS+=("$DURATION")
fi

log "将生成 ${#CUT_STARTS[@]} 个片段"

# 拼接视频
if [[ ${#CUT_STARTS[@]} -eq 0 ]]; then
    error "未检测到有效片段"
fi

# 创建 concat 文件
CONCAT_FILE="${OUTPUT}/concat.txt"
> "$CONCAT_FILE"

for i in "${!CUT_STARTS[@]}"; do
    CLIP_PATH="${OUTPUT}/clip_$i.mp4"
    log "提取片段 $((i+1)): ${CUT_STARTS[$i]}s - ${CUT_ENDS[$i]}s"
    
    ffmpeg -y -ss "${CUT_STARTS[$i]}" -to "${CUT_ENDS[$i]}" -i "$VIDEO_PATH" \
        -c:v libx264 -c:a aac -avoid_negative_ts make_zero \
        "$CLIP_PATH" 2>/dev/null
    
    echo "file '$CLIP_PATH'" >> "$CONCAT_FILE"
done

# 合并片段
FINAL_OUTPUT="${OUTPUT}/${BASENAME##*/}_cut.${EXTENSION}"
log "合并片段..."

ffmpeg -y -f concat -safe 0 -i "$CONCAT_FILE" \
    -c:v libx264 -c:a aac \
    -movflags +faststart \
    "$FINAL_OUTPUT" 2>/dev/null

# 清理临时文件
rm -rf "$AUDIO_PATH" "$SILENCE_FILE" "$CONCAT_FILE" "${OUTPUT}"/*.mp4

log "完成！输出: $FINAL_OUTPUT"
log "原始时长: ${DURATION}s → 剪辑后: $(ffmpeg -i "$FINAL_OUTPUT" 2>&1 | grep Duration | cut -d' ' -f4 | tr -d ,)"
