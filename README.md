# ✂️ AutoCut - 智能气口剪辑

AutoCut 是一款智能音视频气口剪辑工具，能够自动检测并去除音视频中的气口停顿，让你的内容更加流畅专业。

## ✨ 功能特点

- 🎯 **智能检测** - 自动识别音频中的气口/静音区域
- 🎚️ **参数可调** - 支持自定义静音阈值、最小气口时长、合并间隔等参数
- 📊 **可视化预览** - 波形图、增益曲线、片段总览三面板时间轴
- ▶️ **实时预览** - 播放时自动跳过气口区域，预览剪辑效果
- 🔀 **灵活控制** - 可单独切换每个气口的剪除/保留状态
- 📦 **一键导出** - 支持音频和视频格式的导出
- 🎵 **多格式支持** - MP3、WAV、FLAC、AAC、MP4、MOV、MKV 等

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
python app.py
```

服务启动后访问 http://localhost:8765

### 命令行模式

```bash
# 分析视频气口
python autocut.py video.mp4 -a

# 自动剪辑导出
python autocut.py video.mp4 -o output.mp4

# 自定义参数
python autocut.py video.mp4 -t -35 -g 0.1 -m 0.3 -o output.mp4

# 批量处理
python autocut.py --batch ./videos -o ./output
```

## 📸 界面预览

Web 界面提供了完整的可视化操作体验：

1. **上传** - 拖拽或点击上传音视频文件
2. **参数设置** - 调整检测参数
3. **分析预览** - 查看波形、增益和片段时间轴
4. **导出** - 一键导出剪辑后的文件

## 🛠️ 技术栈

- **后端**: Python + FastAPI + Uvicorn
- **音视频处理**: MoviePy + NumPy
- **前端**: 原生 HTML/CSS/JavaScript (Canvas 绘制时间轴)

## 📝 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 静音阈值 | -35 dB | 低于此值判定为静音 |
| 最小气口时长 | 0.1 秒 | 短于此值的静音不计为气口 |
| 合并间隔 | 0.3 秒 | 两个气口间距小于此值则合并 |
| 边界保留 | 0.08 秒 | 气口两端保留的缓冲时间 |

## 📄 License

MIT License
