import os
import time
import signal
import sys
import json
import math
from array import array
from pathlib import Path

import pvporcupine
from pvrecorder import PvRecorder
import simpleaudio as sa
import numpy as np

from faster_whisper import WhisperModel


from opencc import OpenCC
cc = OpenCC('t2s')  # 繁转简


# === 固定参数（与你原来一致） ===
ACCESS_KEY = "vn5k8kaBoYiHIdzL+PqpXFLuWsVZfr0xZ+rqm31S0idR8orawwQcxQ=="  # 你的 access_key
MODEL_PATH = os.path.abspath("porcupine_params_zh.pv")
KEYWORD_PATHS = [os.path.abspath("小智.ppn")]
SENSITIVITIES = [0.6]
DEVICE_INDEX = -1  # -1 用系统默认麦克风


def list_devices():
    devices = PvRecorder.get_available_devices()
    print("可用音频输入设备列表（索引：名称）：")
    for i, name in enumerate(devices):
        print(f"{i}: {name}")


# === 播放提示音 ===
WAKE_VOICE_PATH = os.path.abspath("在.wav")
_wake_wave = None


def play_wake_voice(block=True):
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.PlaySound(WAKE_VOICE_PATH, winsound.SND_FILENAME)
            return
        else:
            global _wake_wave
            if _wake_wave is None:
                _wake_wave = sa.WaveObject.from_wave_file(WAKE_VOICE_PATH)
            play_obj = _wake_wave.play()
            if block:
                play_obj.wait_done()
    except BaseException as e:
        print(f"[提示音] 播放失败：{type(e).__name__}: {e}",
              file=sys.stderr, flush=True)


def auto_select_mic_device():
    """
    优先 'mic/麦克风/microphone/input'，避开 'speaker/扬声器/耳机/headphone/loopback'。
    找不到则返回 -1（默认）。
    """
    devices = PvRecorder.get_available_devices()
    prefer = ("mic", "麦克风", "microphone", "input")
    avoid = ("speaker", "扬声器", "耳机", "headphone", "loopback")
    best = -1
    for idx, name in enumerate(devices):
        lname = name.lower()
        if any(k in lname for k in avoid):
            continue
        if any(k in lname for k in prefer):
            return idx
        if best == -1:
            best = idx
    return best if best != -1 else -1


def main_loop():
    global DEVICE_INDEX
    list_devices()
    DEVICE_INDEX = auto_select_mic_device()
    print(f"自动选择的音频输入设备索引: {DEVICE_INDEX}")

    porcupine = None
    recorder = None

    try:
        porcupine = pvporcupine.create(
            access_key=ACCESS_KEY,
            model_path=MODEL_PATH,
            keyword_paths=KEYWORD_PATHS,
            sensitivities=SENSITIVITIES,
        )

        recorder = PvRecorder(device_index=DEVICE_INDEX,
                              frame_length=porcupine.frame_length)
        recorder.start()

        print("正在监听唤醒词：『小智』")
        print("按 Ctrl+C 结束。")
        print(
            f"(采样率: {porcupine.sample_rate} Hz, 帧长: {porcupine.frame_length})")

        def handle_sigint(sig, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGINT, handle_sigint)

        while True:
            pcm = recorder.read()
            result = porcupine.process(pcm)
            if result >= 0:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{ts}] 检测到唤醒词：小智", flush=True)
                try:
                    play_wake_voice(block=True)
                    # 复用同一个 recorder，提示音已在上面播放
                    text = realtime_zh(rec_shared=recorder, beep_guard=0.0)
                    if text:
                        print(f"[{ts}] 识别文本：{text}", flush=True)
                    else:
                        print(f"[{ts}] 未识别到有效语音。", flush=True)
                except BaseException as e:
                    print(f"[唤醒处理] 异常：{type(e).__name__}: {e}",
                          file=sys.stderr, flush=True)
                    continue

    except KeyboardInterrupt:
        print("\n已停止。")
    except Exception as e:
        print(f"运行出错：{e}", file=sys.stderr)
    finally:
        if recorder is not None:
            try:
                recorder.stop()
            except Exception:
                pass
        if porcupine is not None:
            porcupine.delete()


if __name__ == "__main__":
    main_loop()
