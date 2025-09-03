from utils.player import Player
from utils.reader import Reader
from utils.rouser import Rouser
from utils.UIoperator import UIOperator
from pvrecorder import PvRecorder
import signal
import time
import sys
import os
from dotenv import load_dotenv
import warnings


def list_devices():
    """获取可选语音设备"""
    devices = PvRecorder.get_available_devices()
    print("可用音频输入设备列表（索引：名称）：")
    for i, name in enumerate(devices):
        print(f"{i}: {name}")


def auto_select_mic_device():
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


def main_loop():
    # Suppress warnings
    warnings.filterwarnings("ignore")
    load_dotenv()
    list_devices()
    DEVICE_INDEX = auto_select_mic_device()
    rouse_word = os.getenv("ROUSE_WORD")
    print(rouse_word)
    print(f"自动选择的音频输入设备索引: {DEVICE_INDEX}")
    rouser = Rouser(access_key=os.getenv("ACCESS_KEY"),
                    device_index=DEVICE_INDEX,
                    keyword=rouse_word)
    player = Player()
    reader = Reader(device_index=DEVICE_INDEX)
    op = UIOperator()

    try:

        recorder = PvRecorder(device_index=DEVICE_INDEX,
                              frame_length=rouser.porcupine.frame_length)
        recorder.start()

        print(f"正在监听唤醒词：『{rouse_word}』")
        print("按 Ctrl+C 结束。")
        # print(
        #     f"(采样率: {rouser.porcupine.sample_rate} Hz, 帧长: {rouser.porcupine.frame_length})")

        def handle_sigint(sig, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGINT, handle_sigint)

        while True:
            pcm = recorder.read()
            result = rouser.porcupine.process(pcm)
            if result >= 0:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{ts}] 检测到唤醒词：{rouse_word}", flush=True)
                try:
                    player.play_voice(block=True)
                    # 复用同一个 recorder，提示音已在上面播放
                    text = reader.realtime_zh(
                        rec_shared=recorder, beep_guard=0.0)
                    if text:
                        print(f"[{ts}] 识别文本：{text}", flush=True)
                        player.play_voice(block=True, type="ok")
                        op.run_and_ask(text)
                    else:
                        print(f"[{ts}] 未识别到有效语音。", flush=True)
                        player.play_voice(block=True, type="sorry")
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


if __name__ == "__main__":
    main_loop()
