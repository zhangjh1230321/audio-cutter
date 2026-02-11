#!/usr/bin/env python3
"""
AutoCut Python - 自动剪辑气口 (增强版)

需要: pip install moviePy pydub matplotlib numpy

功能:
- 精确音频波形可视化
- 批量处理目录
- 自定义输出格式
- 气口位置预览
- 剪辑后预览时间轴（气口 + 音频增益）
- GUI 预览（可选）
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict

try:
    from moviePy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
    from pydub import AudioSegment
    import matplotlib
    matplotlib.use('Agg')  # 非交互后端，保证无 GUI 环境也能运行
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import BrokenBarHCollection
    import numpy as np
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════

def format_time(seconds: float) -> str:
    """将秒数格式化为 MM:SS.ms"""
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


def db_to_bar(db: float, min_db: float = -60, max_db: float = 0) -> float:
    """将 dB 映射到 0-1 区间"""
    return max(0.0, min(1.0, (db - min_db) / (max_db - min_db)))


# ═══════════════════════════════════════════════════════════════
#  时间轴预览
# ═══════════════════════════════════════════════════════════════

class TimelinePreview:
    """
    剪辑后预览时间轴
    - 可在终端显示 ASCII 时间轴
    - 可生成 matplotlib 多面板高清预览图
    """

    # 终端颜色
    C_RESET   = "\033[0m"
    C_BOLD    = "\033[1m"
    C_DIM     = "\033[2m"
    C_GREEN   = "\033[92m"
    C_RED     = "\033[91m"
    C_YELLOW  = "\033[93m"
    C_CYAN    = "\033[96m"
    C_MAGENTA = "\033[95m"
    C_BLUE    = "\033[94m"
    C_BG_RED  = "\033[41m"
    C_BG_GREEN = "\033[42m"

    def __init__(self, total_duration: float,
                 silence_regions: List[Tuple[float, float]],
                 segments: List[Tuple[float, float]],
                 gain_data: List[Dict],
                 threshold_db: float = -30):
        """
        Parameters
        ----------
        total_duration : 原始视频总时长 (s)
        silence_regions : [(start, end), ...] 气口区域
        segments : [(start, end), ...] 保留片段区域
        gain_data : [{"time": float, "rms_db": float, "peak_db": float}, ...]
                   按时间排序的音频增益采样
        threshold_db : 静音判定阈值 (dB)
        """
        self.total_duration = total_duration
        self.silence_regions = silence_regions
        self.segments = segments
        self.gain_data = gain_data
        self.threshold_db = threshold_db

    # ── 终端 ASCII 预览 ────────────────────────────────────────

    def print_terminal_preview(self, width: int = 80):
        """在终端输出 ASCII 时间轴预览"""
        C = self  # shorthand for color constants

        total_silence = sum(e - s for s, e in self.silence_regions)
        cut_duration = self.total_duration - total_silence

        print()
        print(f"{C.C_BOLD}{C.C_CYAN}{'═' * width}{C.C_RESET}")
        print(f"{C.C_BOLD}{C.C_CYAN}  📐 剪辑后预览时间轴{C.C_RESET}")
        print(f"{C.C_BOLD}{C.C_CYAN}{'═' * width}{C.C_RESET}")

        # ── 摘要 ──
        print(f"\n  {C.C_BOLD}📊 总览{C.C_RESET}")
        print(f"  ┌──────────────────────────────────────┐")
        print(f"  │  原始时长    {format_time(self.total_duration):>10s}             │")
        print(f"  │  气口数量    {len(self.silence_regions):>5d} 个                │")
        print(f"  │  气口总长    {format_time(total_silence):>10s}             │")
        print(f"  │  剪辑后时长  {format_time(cut_duration):>10s}             │")
        ratio = (total_silence / self.total_duration * 100) if self.total_duration > 0 else 0
        print(f"  │  节省比例    {ratio:>5.1f}%                  │")
        print(f"  └──────────────────────────────────────┘")

        # ── 时间轴条 ──
        bar_width = width - 6
        print(f"\n  {C.C_BOLD}🎬 时间轴（绿=保留 红=气口）{C.C_RESET}")
        print(f"  {'─' * bar_width}")
        bar = self._build_ascii_bar(bar_width)
        print(f"  {bar}")
        # 刻度尺
        self._print_ruler(bar_width)

        # ── 气口明细 ──
        print(f"\n  {C.C_BOLD}✂️  气口剪辑明细{C.C_RESET}")
        print(f"  {'─' * 58}")
        print(f"  {'序号':>4s}  {'开始':>10s}  {'结束':>10s}  {'时长':>8s}  {'状态'}")
        print(f"  {'─' * 58}")
        for i, (s, e) in enumerate(self.silence_regions, 1):
            dur = e - s
            status = f"{C.C_RED}✘ 已剪除{C.C_RESET}"
            print(f"  {i:>4d}  {format_time(s):>10s}  {format_time(e):>10s}  {dur:>6.2f}s  {status}")
        print(f"  {'─' * 58}")

        # ── 保留片段 ──
        print(f"\n  {C.C_BOLD}🎞️  保留片段{C.C_RESET}")
        print(f"  {'─' * 60}")
        print(f"  {'序号':>4s}  {'原始起止':>22s}  {'片段时长':>8s}  {'新起始':>10s}")
        print(f"  {'─' * 60}")
        new_start = 0.0
        for i, (s, e) in enumerate(self.segments, 1):
            dur = e - s
            orig_range = f"{format_time(s)} → {format_time(e)}"
            print(f"  {i:>4d}  {orig_range:>22s}  {dur:>6.2f}s  {format_time(new_start):>10s}")
            new_start += dur
        print(f"  {'─' * 60}")

        # ── 音频增益预览 ──
        if self.gain_data:
            self._print_gain_preview(bar_width)

        print(f"\n{C.C_BOLD}{C.C_CYAN}{'═' * width}{C.C_RESET}\n")

    def _build_ascii_bar(self, width: int) -> str:
        """根据时间轴生成彩色 ASCII 条"""
        bar_chars = []
        for col in range(width):
            t = (col / width) * self.total_duration
            in_silence = any(s <= t < e for s, e in self.silence_regions)
            if in_silence:
                bar_chars.append(f"{self.C_BG_RED} {self.C_RESET}")
            else:
                bar_chars.append(f"{self.C_BG_GREEN} {self.C_RESET}")
        return "".join(bar_chars)

    def _print_ruler(self, width: int):
        """打印时间刻度尺"""
        num_ticks = min(10, max(4, width // 10))
        ruler = [' '] * width
        labels = []
        for i in range(num_ticks + 1):
            pos = int(i / num_ticks * (width - 1))
            t = (i / num_ticks) * self.total_duration
            ruler[pos] = '|'
            labels.append((pos, format_time(t)))

        print(f"  {''.join(ruler)}")
        label_line = [' '] * width
        for pos, lbl in labels:
            start = max(0, pos - len(lbl) // 2)
            for j, ch in enumerate(lbl):
                if start + j < width:
                    label_line[start + j] = ch
        print(f"  {''.join(label_line)}")

    def _print_gain_preview(self, width: int):
        """打印音频增益的 ASCII 电平表"""
        print(f"\n  {self.C_BOLD}🔊 音频增益电平{self.C_RESET}")
        print(f"  {'─' * width}")

        # 将 gain_data 按时间映射到 width 列
        height = 8  # ASCII 电平高度
        cols = width
        grid = [[' '] * cols for _ in range(height)]

        for col in range(cols):
            t = (col / cols) * self.total_duration
            # 找最近的 gain 采样
            closest = min(self.gain_data, key=lambda g: abs(g["time"] - t))
            level = db_to_bar(closest["rms_db"], min_db=-60, max_db=0)
            filled = int(level * height)
            for row in range(filled):
                r = height - 1 - row
                if level > 0.8:
                    grid[r][col] = f"{self.C_RED}█{self.C_RESET}"
                elif level > 0.5:
                    grid[r][col] = f"{self.C_YELLOW}█{self.C_RESET}"
                else:
                    grid[r][col] = f"{self.C_GREEN}█{self.C_RESET}"

        # 标注阈值线
        threshold_level = db_to_bar(self.threshold_db, min_db=-60, max_db=0)
        threshold_row = height - 1 - int(threshold_level * height)
        if 0 <= threshold_row < height:
            for col in range(cols):
                if grid[threshold_row][col] == ' ':
                    grid[threshold_row][col] = f"{self.C_DIM}·{self.C_RESET}"

        for row in grid:
            label = ""
            if row is grid[0]:
                label = "  0dB"
            elif row is grid[-1]:
                label = " -60dB"
            elif row is grid[threshold_row] if 0 <= threshold_row < height else False:
                label = f" {self.threshold_db:.0f}dB"
            print(f"  {''.join(row)}{label}")

        self._print_ruler(width)

    # ── Matplotlib 图形预览 ────────────────────────────────────

    def save_timeline_image(self, output_path: str, audio_array=None,
                            sample_rate: int = 44100):
        """
        生成高清多面板时间轴预览图
        
        面板 1: 音频波形 + 气口标记
        面板 2: 音频增益 (dB) 曲线 + 阈值线
        面板 3: 剪辑片段总览（保留 vs 剪除）
        """
        if not MOVIEPY_AVAILABLE:
            print("[AutoCut] 图形预览需要 matplotlib / numpy")
            return

        fig, axes = plt.subplots(3, 1, figsize=(18, 10),
                                 gridspec_kw={'height_ratios': [3, 2, 1]},
                                 sharex=True)
        fig.patch.set_facecolor('#1a1a2e')

        colors = {
            'waveform': '#00d4aa',
            'silence': '#ff4757',
            'segment': '#2ed573',
            'gain_line': '#ffa502',
            'threshold': '#ff6b81',
            'text': '#f1f2f6',
            'grid': '#2f3542',
            'bg': '#1a1a2e',
            'panel_bg': '#16213e',
        }

        for ax in axes:
            ax.set_facecolor(colors['panel_bg'])
            ax.tick_params(colors=colors['text'], labelcolor=colors['text'])
            ax.spines['top'].set_color(colors['grid'])
            ax.spines['bottom'].set_color(colors['grid'])
            ax.spines['left'].set_color(colors['grid'])
            ax.spines['right'].set_color(colors['grid'])

        # ── 面板 1: 波形 + 气口 ──
        ax1 = axes[0]
        if audio_array is not None:
            time_arr = np.arange(len(audio_array)) / sample_rate
            ds = max(1, len(audio_array) // 5000)
            ax1.plot(time_arr[::ds], audio_array[::ds, 0] if audio_array.ndim > 1
                     else audio_array[::ds],
                     linewidth=0.3, alpha=0.85, color=colors['waveform'])

        for i, (s, e) in enumerate(self.silence_regions):
            ax1.axvspan(s, e, alpha=0.25, color=colors['silence'],
                        label='气口 (已剪除)' if i == 0 else '')

        ax1.set_ylabel('振幅', color=colors['text'], fontsize=11)
        ax1.set_title('🎵 音频波形 & 气口检测', color=colors['text'],
                      fontsize=14, fontweight='bold', pad=10)
        ax1.legend(loc='upper right', fontsize=9,
                   facecolor=colors['panel_bg'], edgecolor=colors['grid'],
                   labelcolor=colors['text'])

        # ── 面板 2: 音频增益曲线 ──
        ax2 = axes[1]
        if self.gain_data:
            times = [g['time'] for g in self.gain_data]
            rms_db = [g['rms_db'] for g in self.gain_data]
            peak_db = [g['peak_db'] for g in self.gain_data]

            ax2.fill_between(times, rms_db, -60, alpha=0.3, color=colors['gain_line'])
            ax2.plot(times, rms_db, linewidth=1.0, color=colors['gain_line'],
                     label='RMS 增益 (dB)', alpha=0.9)
            ax2.plot(times, peak_db, linewidth=0.5, color='#ff6348',
                     label='峰值 (dB)', alpha=0.5)

            # 阈值线
            ax2.axhline(y=self.threshold_db, color=colors['threshold'],
                        linestyle='--', linewidth=1.2, alpha=0.8,
                        label=f'阈值 ({self.threshold_db:.0f} dB)')

            # 气口区域标记
            for s, e in self.silence_regions:
                ax2.axvspan(s, e, alpha=0.15, color=colors['silence'])

        ax2.set_ylabel('增益 (dB)', color=colors['text'], fontsize=11)
        ax2.set_ylim(-65, 5)
        ax2.set_title('🔊 音频增益 & 阈值', color=colors['text'],
                      fontsize=14, fontweight='bold', pad=10)
        ax2.legend(loc='upper right', fontsize=9,
                   facecolor=colors['panel_bg'], edgecolor=colors['grid'],
                   labelcolor=colors['text'])

        # ── 面板 3: 片段概览 ──
        ax3 = axes[2]
        # 背景：先画全部为气口
        ax3.barh(0, self.total_duration, left=0, height=0.6,
                 color=colors['silence'], alpha=0.3)
        # 保留片段
        for i, (s, e) in enumerate(self.segments):
            ax3.barh(0, e - s, left=s, height=0.6, color=colors['segment'],
                     alpha=0.8, label='保留片段' if i == 0 else '')
            # 片段标签
            mid = (s + e) / 2
            dur = e - s
            if dur > self.total_duration * 0.02:  # 足够宽才显示标签
                ax3.text(mid, 0, f'{dur:.1f}s', ha='center', va='center',
                         fontsize=7, color='white', fontweight='bold')

        # 气口标记
        for i, (s, e) in enumerate(self.silence_regions):
            ax3.barh(0, e - s, left=s, height=0.6, color=colors['silence'],
                     alpha=0.6, label='气口 (已剪除)' if i == 0 else '')

        ax3.set_yticks([])
        ax3.set_xlabel('时间 (s)', color=colors['text'], fontsize=11)
        ax3.set_title('🎬 剪辑片段总览', color=colors['text'],
                      fontsize=14, fontweight='bold', pad=10)
        ax3.legend(loc='upper right', fontsize=9,
                   facecolor=colors['panel_bg'], edgecolor=colors['grid'],
                   labelcolor=colors['text'])
        ax3.set_xlim(0, self.total_duration)

        # ── 底部统计标注 ──
        total_silence = sum(e - s for s, e in self.silence_regions)
        cut_dur = self.total_duration - total_silence
        ratio = (total_silence / self.total_duration * 100) if self.total_duration > 0 else 0
        stat_text = (f"原始: {format_time(self.total_duration)} │ "
                     f"气口: {len(self.silence_regions)} 个 / {total_silence:.1f}s │ "
                     f"剪辑后: {format_time(cut_dur)} │ "
                     f"节省: {ratio:.1f}%")
        fig.text(0.5, 0.01, stat_text, ha='center', fontsize=11,
                 color=colors['text'], fontstyle='italic',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor=colors['panel_bg'],
                           edgecolor=colors['grid'], alpha=0.9))

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                    edgecolor='none', bbox_inches='tight')
        plt.close(fig)
        print(f"[AutoCut] 📊 剪辑预览时间轴已保存: {output_path}")


# ═══════════════════════════════════════════════════════════════
#  核心剪辑类
# ═══════════════════════════════════════════════════════════════

class AutoCut:
    def __init__(self, video_path: str, threshold: float = -30,
                 min_gap: float = 0.1, merge_gap: float = 0.3):
        self.video_path = video_path
        self.threshold = threshold
        self.min_gap = min_gap
        self.merge_gap = merge_gap
        self.video = None
        self.audio = None
        # 剪辑结果缓存
        self._silence_regions: Optional[List[Tuple[float, float]]] = None
        self._segments_ranges: Optional[List[Tuple[float, float]]] = None
        self._gain_data: Optional[List[Dict]] = None
        self._audio_array = None
        self._sample_rate: int = 44100

    def load_video(self):
        """加载视频文件"""
        if not MOVIEPY_AVAILABLE:
            raise ImportError("需要安装 moviePy: pip install moviePy")
        self.video = VideoFileClip(self.video_path)
        self.audio = self.video.audio
        print(f"[AutoCut] 加载视频: {self.video_path}")
        print(f"  时长: {self.video.duration:.2f}s")
        print(f"  分辨率: {self.video.size}")

    def _ensure_audio_array(self):
        """确保音频数据已加载（缓存）"""
        if self._audio_array is None:
            if self.audio is None:
                raise ValueError("视频没有音频轨道")
            self._audio_array = self.audio.to_soundarray()
            self._sample_rate = self.audio.fps

    def detect_silence(self, audio_clip=None) -> List[Tuple[float, float]]:
        """检测静音区域"""
        if audio_clip is None:
            audio_clip = self.audio

        if audio_clip is None:
            raise ValueError("视频没有音频轨道")

        # 获取音频数据
        self._ensure_audio_array()
        audio_array = self._audio_array
        sample_rate = self._sample_rate

        # 分段分析 (每 0.1s 一段)
        chunk_duration = 0.1
        chunk_samples = int(chunk_duration * sample_rate)

        silence_regions = []
        in_silence = False
        silence_start = 0

        for i in range(0, len(audio_array), chunk_samples):
            chunk = audio_array[i:i+chunk_samples]
            if len(chunk) == 0:
                continue

            # 计算 RMS 音量
            rms = np.sqrt(np.mean(chunk**2))
            db = 20 * np.log10(rms + 1e-10)

            if db < self.threshold:
                if not in_silence:
                    in_silence = True
                    silence_start = i / sample_rate
            else:
                if in_silence:
                    duration = (i / sample_rate) - silence_start
                    if duration >= self.min_gap:
                        silence_regions.append((silence_start, (i / sample_rate)))
                    in_silence = False

        # 处理结尾的静音
        if in_silence:
            duration = (len(audio_array) / sample_rate) - silence_start
            if duration >= self.min_gap:
                silence_regions.append((silence_start, len(audio_array) / sample_rate))

        # 合并相近的气口
        merged = []
        for start, end in silence_regions:
            if merged and start - merged[-1][1] < self.merge_gap:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))

        self._silence_regions = merged
        return merged

    def compute_gain_data(self) -> List[Dict]:
        """
        计算音频增益数据（RMS 和峰值），每 0.1s 一个采样点
        """
        self._ensure_audio_array()
        audio_array = self._audio_array
        sample_rate = self._sample_rate

        chunk_duration = 0.1
        chunk_samples = int(chunk_duration * sample_rate)

        gain_data = []
        for i in range(0, len(audio_array), chunk_samples):
            chunk = audio_array[i:i+chunk_samples]
            if len(chunk) == 0:
                continue

            t = i / sample_rate
            rms = np.sqrt(np.mean(chunk**2))
            peak = np.max(np.abs(chunk))

            rms_db = 20 * np.log10(rms + 1e-10)
            peak_db = 20 * np.log10(peak + 1e-10)

            gain_data.append({
                "time": round(t, 3),
                "rms_db": round(rms_db, 2),
                "peak_db": round(peak_db, 2),
            })

        self._gain_data = gain_data
        return gain_data

    def _compute_segment_ranges(self, silence_regions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """计算保留片段的时间范围"""
        segments = []
        prev_end = 0.0

        for start, end in silence_regions:
            if start - prev_end > 0.3:
                segments.append((prev_end, start))
            prev_end = end

        # 最后一段
        if self.video.duration - prev_end > 0.3:
            segments.append((prev_end, self.video.duration))

        self._segments_ranges = segments
        return segments

    def cut_video(self, silence_regions: List[Tuple[float, float]]) -> List[VideoFileClip]:
        """根据静音区域剪切视频"""
        if self.video is None:
            self.load_video()

        seg_ranges = self._compute_segment_ranges(silence_regions)
        clips = []
        for s, e in seg_ranges:
            clips.append(self.video.subclip(s, e))

        return clips

    def analyze(self) -> dict:
        """分析视频气口"""
        if self.video is None:
            self.load_video()

        silence = self.detect_silence()
        self.compute_gain_data()

        total_silence = sum(end - start for start, end in silence)

        result = {
            "duration": self.video.duration,
            "silence_count": len(silence),
            "total_silence": total_silence,
            "cut_duration": self.video.duration - total_silence,
            "silence_regions": silence
        }

        print(f"\n[AutoCut] 分析结果:")
        print(f"  视频时长: {result['duration']:.2f}s")
        print(f"  气口数量: {result['silence_count']}")
        print(f"  气口总长: {result['total_silence']:.2f}s")
        print(f"  预计剪辑后: {result['cut_duration']:.2f}s")

        if silence:
            print(f"\n  气口位置:")
            for i, (s, e) in enumerate(silence, 1):
                print(f"    {i}. {s:.2f}s - {e:.2f}s ({e-s:.2f}s)")

        return result

    def show_preview(self, silence_regions: List[Tuple[float, float]],
                     save_image: bool = True):
        """
        展示剪辑后预览时间轴
        - 终端 ASCII 预览
        - 可选保存高清时间轴图
        """
        if self._segments_ranges is None:
            self._compute_segment_ranges(silence_regions)
        if self._gain_data is None:
            self.compute_gain_data()

        preview = TimelinePreview(
            total_duration=self.video.duration,
            silence_regions=silence_regions,
            segments=self._segments_ranges,
            gain_data=self._gain_data,
            threshold_db=self.threshold,
        )

        # 终端预览
        preview.print_terminal_preview()

        # 高清图预览
        if save_image:
            self._ensure_audio_array()
            img_name = Path(self.video_path).stem + "_timeline.png"
            img_path = str(Path(self.video_path).parent / img_name)
            preview.save_timeline_image(
                output_path=img_path,
                audio_array=self._audio_array,
                sample_rate=self._sample_rate,
            )

    def export(self, output_path: str, visualize: bool = False,
               preview: bool = True):
        """导出剪辑后的视频"""
        if self.video is None:
            self.load_video()

        silence = self.detect_silence()
        segments = self.cut_video(silence)

        if not segments:
            raise ValueError("没有可导出的片段")

        print(f"[AutoCut] 生成 {len(segments)} 个片段...")

        # 拼接片段
        final = concatenate_videoclips(segments, method="compose")

        # 导出
        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=24,
            preset="medium"
        )

        print(f"[AutoCut] 已导出: {output_path}")

        # 气口波形可视化（旧功能）
        if visualize:
            self.visualize(silence)

        # ★ 新增：剪辑后预览时间轴
        if preview:
            self.show_preview(silence, save_image=True)

        # 清理
        for segment in segments:
            segment.close()
        final.close()
        self.video.close()

    def visualize(self, silence_regions: List[Tuple[float, float]]):
        """可视化音频波形和气口位置（旧版简单图）"""
        if not MOVIEPY_AVAILABLE:
            print("[AutoCut] 可视化需要 matplotlib")
            return

        self._ensure_audio_array()
        audio_array = self._audio_array
        sample_rate = self._sample_rate

        # 绘制波形
        fig, ax = plt.subplots(figsize=(14, 4))

        time = np.arange(len(audio_array)) / sample_rate

        # 降低采样率以加快绘图
        downsample = 100
        time = time[::downsample]
        audio_ds = audio_array[::downsample]

        ax.plot(time, audio_ds, linewidth=0.1, alpha=0.7)

        # 标记气口
        for start, end in silence_regions:
            ax.axvspan(start, end, alpha=0.3, color='red',
                       label='气口' if start == silence_regions[0][0] else '')

        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('音量')
        ax.set_title(f'音频波形与气口检测 (阈值: {self.threshold}dB)')
        ax.legend()

        plt.tight_layout()
        plt.savefig('autocut_waveform.png', dpi=100)
        print("[AutoCut] 波形图已保存: autocut_waveform.png")

    def batch_process(self, input_dir: str, output_dir: str):
        """批量处理目录下的视频"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        video_files = list(input_path.glob("*.mp4")) + \
                      list(input_path.glob("*.mov")) + \
                      list(input_path.glob("*.mkv"))

        print(f"[AutoCut] 找到 {len(video_files)} 个视频文件")

        for video_file in video_files:
            print(f"\n处理: {video_file.name}")
            self.video_path = str(video_file)
            self.load_video()

            output_file = output_path / f"{video_file.stem}_cut{video_file.suffix}"

            try:
                self.analyze()
                self.export(str(output_file))
            except Exception as e:
                print(f"  错误: {e}")


def main():
    parser = argparse.ArgumentParser(description='AutoCut - 自动剪辑气口')
    parser.add_argument('video', nargs='?', help='视频文件路径')
    parser.add_argument('-t', '--threshold', type=float, default=-30,
                       help='静音阈值 dB (默认: -30)')
    parser.add_argument('-g', '--min-gap', type=float, default=0.1,
                       help='最小气口时长 秒 (默认: 0.1)')
    parser.add_argument('-m', '--merge', type=float, default=0.3,
                       help='合并间隔 秒 (默认: 0.3)')
    parser.add_argument('-o', '--output', help='输出文件路径')
    parser.add_argument('-a', '--analyze-only', action='store_true',
                       help='仅分析，不剪辑')
    parser.add_argument('-v', '--visualize', action='store_true',
                       help='生成波形可视化')
    parser.add_argument('--no-preview', action='store_true',
                       help='禁用剪辑后预览时间轴')
    parser.add_argument('--preview-only', action='store_true',
                       help='仅生成预览时间轴（不导出视频）')
    parser.add_argument('--batch', help='批量处理目录')

    args = parser.parse_args()

    if args.batch:
        cutter = AutoCut("", args.threshold, args.min_gap, args.merge)
        cutter.batch_process(args.batch, args.output or args.batch + "_output")
        return

    if not args.video:
        parser.print_help()
        return

    cutter = AutoCut(args.video, args.threshold, args.min_gap, args.merge)

    if args.preview_only:
        # 仅预览模式：分析 + 显示时间轴
        cutter.load_video()
        silence = cutter.detect_silence()
        cutter._compute_segment_ranges(silence)
        cutter.show_preview(silence, save_image=True)
    elif args.analyze_only:
        cutter.load_video()
        result = cutter.analyze()
        # 分析模式也显示预览
        silence = result['silence_regions']
        cutter._compute_segment_ranges(silence)
        cutter.show_preview(silence, save_image=False)
    else:
        output = args.output
        if not output:
            base = Path(args.video).stem
            output = f"{base}_cut.mp4"

        cutter.export(output, visualize=args.visualize,
                      preview=not args.no_preview)


if __name__ == "__main__":
    main()
