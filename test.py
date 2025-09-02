import queue
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel


def realtime_zh_whisper(model_size="small",      # 可选: "small" | "medium" | "large-v3"
                        device="auto",            # "auto" 或 "cuda"/"cpu"
                        silence_seconds=1.0,      # 连续静音判定结束/定稿
                        vad_level=1):             # 0~3 越大越严格（嘈杂环境可用 3）
    sr = 16000
    frame_ms = 20
    blocksize = int(sr * frame_ms / 1000)
    bytes_per_frame = blocksize * 2  # int16(2 bytes)

    # 模型：GPU 用 float16，CPU 用 int8 以提速
    use_cuda = (device == "cuda")
    model = WhisperModel(model_size,
                         device="cuda" if use_cuda else "cpu",
                         compute_type="float16" if use_cuda else "int8")

    vad = webrtcvad.Vad(vad_level)
    q = queue.Queue()

    def _cb(indata, frames, t, status):
        q.put(bytes(indata))  # int16 PCM

    # 用于上下文提示，能让标点/术语更稳
    history = ""

    with sd.RawInputStream(samplerate=sr, channels=1, dtype="int16",
                           blocksize=blocksize, callback=_cb):
        print(f"开始：说话吧（连续 {silence_seconds:.1f}s 静音会自动定稿）。Ctrl+C 退出。")
        buf = bytearray()
        seg = bytearray()
        started, silent = False, 0.0

        while True:
            data = q.get()
            buf.extend(data)

            # 按 20ms 帧做 VAD
            while len(buf) >= bytes_per_frame:
                frame = bytes(buf[:bytes_per_frame])
                del buf[:bytes_per_frame]
                speech = vad.is_speech(frame, sr)

                if speech:
                    started = True
                    silent = 0.0
                    seg.extend(frame)
                elif started:
                    silent += frame_ms / 1000.0
                    seg.extend(frame)  # 留一点尾巴，避免截断

                # 达到静音阈值 -> 定稿并输出
                if started and silent >= silence_seconds:
                    # int16 -> float32 (-1..1)
                    audio = np.frombuffer(seg, dtype=np.int16).astype(
                        np.float32) / 32768.0
                    prompt = "以下是中文口语对话，请用简体中文和恰当标点。" + history[-80:]
                    segments, _ = model.transcribe(
                        audio, language="zh", vad_filter=False,
                        beam_size=5, temperature=0.0, initial_prompt=prompt
                    )
                    text = "".join(s.text for s in segments).strip()
                    if text:
                        print(text)
                        history += text

                    # 重置继续下一句
                    seg.clear()
                    started, silent = False, 0.0


if __name__ == "__main__":
    # GPU 用户改成 device="cuda"，并把 model_size 提到 "large-v3" 追求更高准确率
    realtime_zh_whisper(model_size="small",
                        device="auto", silence_seconds=1)
