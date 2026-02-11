# AutoCut - 自动剪辑气口

## 功能描述
自动检测视频中的静音/气口区域，并进行智能剪辑。适用于：
- 直播切片
- 口播视频优化
- 去除过长停顿
- 视频素材整理

## 前置依赖
```bash
# 核心依赖
brew install ffmpeg

# 音频分析（推荐）
brew install sox

# 可选：Python 增强
pip install moviepy pydub
```

## 核心逻辑
1. **音频分析** - 检测音量低于阈值的区域（气口/停顿）
2. **智能合并** - 将短间隔的气口合并，避免过度碎片化
3. **精确剪辑** - 基于时间戳进行视频切割
4. **自动合成** - 将片段拼接成新视频

## 使用方法
### 基础用法（默认参数）
```bash
autocut /path/to/video.mp4
```

### 自定义参数
```bash
# 设置静音阈值 (-60dB 到 -20dB，默认 -30dB)
autocut /path/to/video.mp4 --threshold -40

# 设置最小气口时长（秒，默认 0.5s）
autocut /path/to/video.mp4 --min-gap 1.0

# 设置合并间隔（秒，默认 0.3s）
autocut /path/to/video.mp4 --merge 0.5

# 输出目录
autocut /path/to/video.mp4 --output /path/to/output

# 仅分析，不剪辑
autocut /path/to/video.mp4 --analyze-only
```

## Python 增强版（可选）
需要 moviePy：
```bash
pip install moviePy
```

功能：
- 更精确的音频波形可视化
- 批量处理目录
- 自定义输出格式
- 气口位置预览

## 注意事项
- 气口检测基于音量阈值，不是语义分析
- 建议先用 `--analyze-only` 预览效果
- 音乐/背景音视频可能误判
