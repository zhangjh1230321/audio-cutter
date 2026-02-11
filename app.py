#!/usr/bin/env python3
"""
AutoCut Web - FastAPI 后端
"""

import os
import uuid
import json
import shutil
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# ──────────────────────────────────────────────────────────────
#  配置
# ──────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="AutoCut Web", version="1.0")

# 静态文件
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 任务状态存储
tasks = {}


# ──────────────────────────────────────────────────────────────
#  工具函数
# ──────────────────────────────────────────────────────────────

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".opus"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".ts", ".wmv"}
ALL_EXTS = AUDIO_EXTS | VIDEO_EXTS


def is_audio_file(path: str) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXTS


def compute_audio_data(file_path: str, threshold: float = -35,
                       min_gap: float = 0.1, merge_gap: float = 0.3,
                       padding: float = 0.08):
    """分析音频/视频文件，返回完整分析数据"""
    import numpy as np
    from moviepy import VideoFileClip, AudioFileClip

    audio_only = is_audio_file(file_path)
    if audio_only:
        clip = AudioFileClip(file_path)
        audio_obj = clip
        duration = clip.duration
        video_fps = None
    else:
        clip = VideoFileClip(file_path)
        audio_obj = clip.audio
        duration = clip.duration
        video_fps = clip.fps

    if audio_obj is None:
        clip.close()
        raise ValueError("文件没有音频轨道")

    audio_array = np.array(list(audio_obj.iter_frames()))
    sample_rate = audio_obj.fps

    # ── 检测静音 ──
    chunk_duration = 0.1
    chunk_samples = int(chunk_duration * sample_rate)

    silence_regions = []
    gain_data = []
    in_silence = False
    silence_start = 0

    for i in range(0, len(audio_array), chunk_samples):
        chunk = audio_array[i:i + chunk_samples]
        if len(chunk) == 0:
            continue

        t = round(i / sample_rate, 3)
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        peak = float(np.max(np.abs(chunk)))

        rms_db = round(20 * np.log10(rms + 1e-10), 2)
        peak_db = round(20 * np.log10(peak + 1e-10), 2)

        gain_data.append({"time": t, "rms_db": rms_db, "peak_db": peak_db})

        if rms_db < threshold:
            if not in_silence:
                in_silence = True
                silence_start = t
        else:
            if in_silence:
                dur = t - silence_start
                if dur >= min_gap:
                    silence_regions.append([silence_start, t])
                in_silence = False

    if in_silence:
        end_t = round(len(audio_array) / sample_rate, 3)
        dur = end_t - silence_start
        if dur >= min_gap:
            silence_regions.append([silence_start, end_t])

    # 合并相近气口
    merged = []
    for start, end in silence_regions:
        if merged and start - merged[-1][1] < merge_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    # ── 应用边界保留（padding）──
    # 在每个静音区域两端保留 padding 秒的缓冲，让过渡更自然
    padded = []
    for start, end in merged:
        ps = start + padding
        pe = end - padding
        if pe > ps:  # 只有缩减后仍有剩余才算有效静音
            padded.append([round(ps, 3), round(pe, 3)])

    # ── 计算保留片段 ──
    segments = []
    prev_end = 0.0
    for start, end in padded:
        if start - prev_end > 0.05:
            segments.append([round(prev_end, 3), round(start, 3)])
        prev_end = end
    if duration - prev_end > 0.05:
        segments.append([round(prev_end, 3), round(duration, 3)])

    # ── 波形数据（降采样） ──
    mono = audio_array[:, 0] if audio_array.ndim > 1 else audio_array
    ds = max(1, len(mono) // 2000)
    waveform_samples = mono[::ds].tolist()
    waveform_times = (np.arange(len(mono))[::ds] / sample_rate).tolist()

    total_silence = sum(e - s for s, e in padded)

    result = {
        "duration": round(duration, 3),
        "sample_rate": sample_rate,
        "is_audio": audio_only,
        "video_fps": video_fps,
        "silence_regions": merged,
        "silence_count": len(merged),
        "total_silence": round(total_silence, 3),
        "cut_duration": round(duration - total_silence, 3),
        "segments": segments,
        "gain_data": gain_data,
        "waveform": {"times": waveform_times, "samples": waveform_samples},
    }

    clip.close()
    return result


def export_media(file_path: str, segments, output_path: str, video_fps=None):
    """根据片段列表导出剪辑后的音频/视频"""
    from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips

    audio_only = is_audio_file(file_path)

    if audio_only:
        clip = AudioFileClip(file_path)
        sub_clips = [clip.subclipped(s, e) for s, e in segments]
        if not sub_clips:
            clip.close()
            raise ValueError("没有可导出的片段")
        final = concatenate_audioclips(sub_clips)
        final.write_audiofile(output_path, logger=None)
    else:
        clip = VideoFileClip(file_path)
        sub_clips = [clip.subclipped(s, e) for s, e in segments]
        if not sub_clips:
            clip.close()
            raise ValueError("没有可导出的片段")
        final = concatenate_videoclips(sub_clips)
        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=video_fps or clip.fps or 24,
            preset="medium",
            logger=None,
        )

    for c in sub_clips:
        c.close()
    final.close()
    clip.close()


# ──────────────────────────────────────────────────────────────
#  API 路由
# ──────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传音频或视频文件"""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALL_EXTS:
        raise HTTPException(400, f"不支持的文件格式: {ext}，支持: {', '.join(sorted(ALL_EXTS))}")

    file_id = str(uuid.uuid4())[:8]
    save_name = f"{file_id}{ext}"
    save_path = UPLOAD_DIR / save_name

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    file_size = save_path.stat().st_size
    return {
        "file_id": file_id,
        "filename": file.filename,
        "save_name": save_name,
        "size": file_size,
        "size_mb": round(file_size / (1024 * 1024), 2),
    }


@app.post("/api/analyze")
async def analyze_file(
    save_name: str = Form(...),
    threshold: float = Form(-35),
    min_gap: float = Form(0.1),
    merge_gap: float = Form(0.3),
    padding: float = Form(0.08),
):
    """分析音频/视频气口"""
    video_path = UPLOAD_DIR / save_name
    if not video_path.exists():
        raise HTTPException(404, "视频文件不存在")

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, compute_audio_data, str(video_path), threshold, min_gap, merge_gap, padding
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"分析失败: {str(e)}")


@app.post("/api/export")
async def export_cut_file(
    save_name: str = Form(...),
    segments: str = Form(...),
    video_fps: float = Form(None),
    background_tasks: BackgroundTasks = None,
):
    """导出剪辑后的音频/视频"""
    file_path = UPLOAD_DIR / save_name
    if not file_path.exists():
        raise HTTPException(404, "文件不存在")

    seg_list = json.loads(segments)
    task_id = str(uuid.uuid4())[:8]
    ext = Path(save_name).suffix.lower()
    # 音频文件统一输出为 mp3
    if ext in AUDIO_EXTS:
        out_ext = ".mp3"
    else:
        out_ext = ext
    output_name = f"{task_id}_cut{out_ext}"
    output_path = OUTPUT_DIR / output_name

    tasks[task_id] = {"status": "processing", "progress": 0, "output": None}

    def do_export():
        try:
            export_media(str(file_path), seg_list, str(output_path), video_fps=video_fps)
            tasks[task_id] = {
                "status": "done",
                "progress": 100,
                "output": output_name,
            }
        except Exception as e:
            tasks[task_id] = {"status": "error", "progress": 0, "error": str(e)}

    background_tasks.add_task(do_export)

    return {"task_id": task_id, "message": "导出任务已开始"}


@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """查询导出任务状态"""
    if task_id not in tasks:
        raise HTTPException(404, "任务不存在")
    return tasks[task_id]


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """下载导出的音频/视频"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "文件不存在")

    ext = Path(filename).suffix.lower()
    content_types = {
        ".mp4": "video/mp4", ".mov": "video/quicktime", ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo", ".webm": "video/webm",
        ".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
        ".aac": "audio/aac", ".ogg": "audio/ogg", ".m4a": "audio/mp4",
    }
    media_type = content_types.get(ext, "application/octet-stream")

    return FileResponse(
        str(file_path),
        media_type=media_type,
        filename=filename,
    )


@app.get("/api/media/{save_name}")
async def stream_media(save_name: str):
    """流式播放上传的原始音频/视频"""
    file_path = UPLOAD_DIR / save_name
    if not file_path.exists():
        raise HTTPException(404, "文件不存在")

    ext = Path(save_name).suffix.lower()
    content_types = {
        ".mp4": "video/mp4", ".mov": "video/quicktime", ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo", ".webm": "video/webm",
        ".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
        ".aac": "audio/aac", ".ogg": "audio/ogg", ".m4a": "audio/mp4",
        ".wma": "audio/x-ms-wma", ".opus": "audio/opus",
    }
    media_type = content_types.get(ext, "application/octet-stream")
    return FileResponse(str(file_path), media_type=media_type)


# ──────────────────────────────────────────────────────────────
#  启动
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8765, reload=True)
