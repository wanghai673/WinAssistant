import math
import sys
import os
import time
from array import array
import numpy as np
from faster_whisper import WhisperModel
from pvrecorder import PvRecorder
import opencc


class Reader:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_len: int = 512,
        max_silence_secs: float = 1.8,
        max_session_secs: float = 30,
        calibrate_secs: float = 0.5,
        min_rms_threshold: float = 0.018,
        whisper_model_id: str = "medium",
        whisper_device: str = "",
        whisper_compute_type: str = "",
        device_index: int = None,  # 自动选择麦克风设备
    ):
        """
        初始化 Reader 实例，配置录音参数和 Whisper 模型。

        参数:
            sample_rate (int): 音频采样率（Hz），默认 16000。
            frame_len (int): 每帧采样点数量，默认 512。
            max_silence_secs (float): 最大允许静音时长（秒），超时自动结束录音。
            max_session_secs (float): 单次最大录音时长（秒），超过自动停止。
            calibrate_secs (float): 噪声环境校准时长（秒），用于估计背景噪音。
            min_rms_threshold (float): 声音检测的最小 RMS 阈值，防止误触发。
            whisper_model_id (str): Whisper 模型 ID，如 "tiny"、"base"、"small"、"medium"。
            whisper_device (str): 推理设备，可选 "cpu"、"cuda"，空字符串自动选择。
            whisper_compute_type (str): 推理计算精度类型，可选 "int8"、"float16" 等。
            device_index (int): 麦克风设备索引，None 时自动选择。
        """

        # 录音相关参数
        self.SAMPLE_RATE = sample_rate
        self.FRAME_LEN = frame_len
        self.MAX_SILENCE_SECS = max_silence_secs
        self.MAX_SESSION_SECS = max_session_secs
        self.CALIBRATE_SECS = calibrate_secs
        self.MIN_RMS_THRESHOLD = min_rms_threshold

        # Whisper 模型配置
        self.WHISPER_MODEL_ID = whisper_model_id
        self.WHISPER_DEVICE = whisper_device
        self.WHISPER_COMPUTE_TYPE = whisper_compute_type

        # 自动选择麦克风设备
        self.DEVICE_INDEX = device_index if device_index is not None else self.auto_select_mic_device()

        # 加载 Whisper 模型
        self._WHISPER_MODEL = self._ensure_whisper_model()

        # OpenCC 繁简体转换器
        self.cc = opencc.OpenCC('t2s')  # t2s: 繁体转简体

    def auto_select_mic_device(self):
        """
        优先 '耳机'。
        找不到则返回 -1（默认）。
        """
        devices = PvRecorder.get_available_devices()
        prefer = ("耳机", "headphone")
        best = -1
        for idx, name in enumerate(devices):
            lname = name.lower()
            if any(k in lname for k in prefer):
                return idx
        return best

    def _ensure_whisper_model(self):
        """
        加载 fast-whisper，自动选择 device/compute_type，
        用户通过环境变量指定的优先；不支持时会回退。
        """
        try:
            import ctranslate2
        except Exception as e:
            raise RuntimeError(
                "未找到 ctranslate2。请先安装：pip install ctranslate2\n"
                f"导入失败：{e}"
            )

        # 设备自动探测
        try:
            available = set(ctranslate2.get_available_devices())
        except Exception:
            available = {"cpu"}

        device = self.WHISPER_DEVICE or (
            "cuda" if "cuda" in available else "cpu")

        # 该设备支持的 compute types
        try:
            supported = ctranslate2.get_supported_compute_types(device)
        except Exception:
            supported = ["float32"]

        # 选择 compute_type
        chosen = None
        if self.WHISPER_COMPUTE_TYPE:
            if self.WHISPER_COMPUTE_TYPE not in supported:
                print(f"[Whisper] 警告：compute_type='{self.WHISPER_COMPUTE_TYPE}' "
                      f"在 device='{device}' 不受支持。支持：{supported}。将自动回退。",
                      file=sys.stderr)
            else:
                chosen = self.WHISPER_COMPUTE_TYPE

        if chosen is None:
            # CUDA 优先 int8_float16 / float16，再退 int8；CPU/Metal 优先 int8，再退 float16/float32
            preference = ["int8_float16", "float16", "int8", "int8_float32", "float32"] if device == "cuda" \
                else ["int8", "float16", "int8_float32", "float32"]
            for ct in preference:
                if ct in supported:
                    chosen = ct
                    break
            if chosen is None:
                chosen = supported[0]

        print(
            f"[Whisper] 使用 device={device} | compute_type={chosen} | 支持：{supported}")
        print("正在加载Whisper模型中...")
        try:
            _WHISPER_MODEL = WhisperModel(
                self.WHISPER_MODEL_ID,
                device=device,
                compute_type=chosen,
                cpu_threads=max(1, os.cpu_count() // 2)
            )
        except Exception as e:
            print(f"[加载] 过程中出错：{e}", file=sys.stderr)
            return ""
        print("Whisper模型加载完成。")
        return _WHISPER_MODEL

    def _rms(self, frame):
        """frame: List[int] (int16) -> 0~1"""
        if not frame:
            return 0.0
        s = 0
        for x in frame:
            s += x * x
        return math.sqrt(s / len(frame)) / 32768.0

    def realtime_zh(self, rec_shared=None, beep_guard=0.8):
        """
        唤醒后进入：
        - 复用外部 PvRecorder（如果传入），否则自建
        - 可选吞掉 beep_guard 秒提示音
        - 简易 VAD：检测说话，静音超过 MAX_SILENCE_SECS 结束
        - 用 fast-whisper 识别并返回文本
        """

        use_shared = rec_shared is not None
        rec = rec_shared if use_shared else PvRecorder(
            device_index=self.DEVICE_INDEX, frame_length=self.FRAME_LEN)

        buff = array('h')
        heard_anything = False
        last_voice_t = None
        voice_threshold = self.MIN_RMS_THRESHOLD

        calibrate_frames_needed = int(
            self.CALIBRATE_SECS * self.SAMPLE_RATE / self.FRAME_LEN + 0.5)
        calibrate_frames = 0
        noise_rms_acc = 0.0

        def _read_one():
            return rec.read()

        try:
            if not use_shared:
                rec.start()

            # 吞提示音
            if beep_guard and beep_guard > 0:
                t0 = time.monotonic()
                while time.monotonic() - t0 < beep_guard:
                    try:
                        _ = _read_one()
                    except Exception:
                        break

            print("请开始说话（保持静音约 2 秒自动结束）...")
            start_t = time.monotonic()

            # 噪声校准（不丢句头：缓冲）
            while calibrate_frames < calibrate_frames_needed:
                frame = _read_one()
                r = self._rms(frame)
                noise_rms_acc += r
                calibrate_frames += 1
                buff.extend(frame)

            noise_floor = noise_rms_acc / max(1, calibrate_frames)
            voice_threshold = max(self.MIN_RMS_THRESHOLD, noise_floor * 3.0)

            # 录音 + VAD
            while True:
                frame = _read_one()
                now = time.monotonic()

                if now - start_t > self.MAX_SESSION_SECS:
                    print("[识别] 已达最长录音时长，自动结束。")
                    break

                r = self._rms(frame)
                buff.extend(frame)

                if r >= voice_threshold:
                    heard_anything = True
                    last_voice_t = now
                else:
                    if heard_anything and last_voice_t and (now - last_voice_t) >= self.MAX_SILENCE_SECS:
                        break

            duration_secs = len(buff) / float(self.SAMPLE_RATE)
            if not heard_anything or duration_secs < 0.4:
                print("[识别] 未检测到有效语音或语音过短。")
                return ""

            # int16 PCM -> float32 [-1,1]
            pcm = np.frombuffer(buff.tobytes(), dtype=np.int16).astype(
                np.float32) / 32768.0

            # 识别：中文固定更稳；VAD 再过滤；适度 beam search 提升准确度
            segments, info = self._WHISPER_MODEL.transcribe(
                audio=pcm,
                language="zh",
                task="transcribe",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                beam_size=10,
                best_of=5,
                temperature=0.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=True,
            )

            parts = []
            for seg in segments:
                if seg.text:
                    parts.append(seg.text.strip())

            text = "".join(parts).strip()
            if text:
                text = self.cc.convert(text)
                # print(f"识别结果（fast-whisper zh）：{text}")
            else:
                # print("识别结果为空。")
                pass

            return text

        except KeyboardInterrupt:
            print("\n[识别] 已取消。")
            return ""
        except Exception as e:
            print(f"[识别] 过程中出错：{e}", file=sys.stderr)
            return ""
        finally:
            if not use_shared:
                try:
                    rec.stop()
                except Exception:
                    pass


if __name__ == '__main__':
    r = Reader()
    print(r.realtime_zh())
