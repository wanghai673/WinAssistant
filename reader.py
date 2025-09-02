

# === Whisper(=fast-whisper) 配置 ===
WHISPER_MODEL_ID = os.getenv("WHISPER_MODEL_ID", "medium")
# 留空走自动检测；也可显式设为 "cpu" 或 "cuda"
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "")
# 留空自动选择；常见取值：float16 / int8 / int8_float16 / float32
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "")

_WHISPER_MODEL = None  # 懒加载


def _ensure_whisper_model():
    """
    懒加载 fast-whisper，自动选择 device/compute_type，
    用户通过环境变量指定的优先；不支持时会回退。
    """
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

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

    device = WHISPER_DEVICE or ("cuda" if "cuda" in available else "cpu")

    # 该设备支持的 compute types
    try:
        supported = ctranslate2.get_supported_compute_types(device)
    except Exception:
        supported = ["float32"]

    # 选择 compute_type
    chosen = None
    if WHISPER_COMPUTE_TYPE:
        if WHISPER_COMPUTE_TYPE not in supported:
            print(f"[Whisper] 警告：compute_type='{WHISPER_COMPUTE_TYPE}' "
                  f"在 device='{device}' 不受支持。支持：{supported}。将自动回退。",
                  file=sys.stderr)
        else:
            chosen = WHISPER_COMPUTE_TYPE

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

    _WHISPER_MODEL = WhisperModel(
        WHISPER_MODEL_ID,
        device=device,
        compute_type=chosen,
        cpu_threads=max(1, os.cpu_count() // 2)
    )
    return _WHISPER_MODEL


def _rms(frame):
    """frame: List[int] (int16) -> 0~1"""
    if not frame:
        return 0.0
    s = 0
    for x in frame:
        s += x * x
    return math.sqrt(s / len(frame)) / 32768.0


model = _ensure_whisper_model()


class Reader:

    def __init__(self, DEVICE_INDEX,):
        """_summary_

        Args:
            DEVICE_INDEX (_type_): _description_
        """
        == = 录音 / VAD 配置 == =
        SAMPLE_RATE = 16000
        FRAME_LEN = 512
        MAX_SILENCE_SECS = 1.8
        MAX_SESSION_SECS = 30
        CALIBRATE_SECS = 0.5
        MIN_RMS_THRESHOLD = 0.018

    def realtime_zh(rec_shared=None, beep_guard=0.8):
        """
        唤醒后进入：
        - 复用外部 PvRecorder（如果传入），否则自建
        - 可选吞掉 beep_guard 秒提示音
        - 简易 VAD：检测说话，静音超过 MAX_SILENCE_SECS 结束
        - 用 fast-whisper 识别并返回文本
        """
        try:
            global model
        except Exception as e:
            print(f"[识别] Whisper 模型加载失败：{e}", file=sys.stderr)
            return ""

        use_shared = rec_shared is not None
        rec = rec_shared if use_shared else PvRecorder(
            device_index=DEVICE_INDEX, frame_length=FRAME_LEN)

        buff = array('h')
        heard_anything = False
        last_voice_t = None
        voice_threshold = MIN_RMS_THRESHOLD

        calibrate_frames_needed = int(
            CALIBRATE_SECS * SAMPLE_RATE / FRAME_LEN + 0.5)
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
                r = _rms(frame)
                noise_rms_acc += r
                calibrate_frames += 1
                buff.extend(frame)

            noise_floor = noise_rms_acc / max(1, calibrate_frames)
            voice_threshold = max(MIN_RMS_THRESHOLD, noise_floor * 3.0)

            # 录音 + VAD
            while True:
                frame = _read_one()
                now = time.monotonic()

                if now - start_t > MAX_SESSION_SECS:
                    print("[识别] 已达最长录音时长，自动结束。")
                    break

                r = _rms(frame)
                buff.extend(frame)

                if r >= voice_threshold:
                    heard_anything = True
                    last_voice_t = now
                else:
                    if heard_anything and last_voice_t and (now - last_voice_t) >= MAX_SILENCE_SECS:
                        break

            duration_secs = len(buff) / float(SAMPLE_RATE)
            if not heard_anything or duration_secs < 0.4:
                print("[识别] 未检测到有效语音或语音过短。")
                return ""

            # int16 PCM -> float32 [-1,1]
            pcm = np.frombuffer(buff.tobytes(), dtype=np.int16).astype(
                np.float32) / 32768.0

            # 识别：中文固定更稳；VAD 再过滤；适度 beam search 提升准确度
            segments, info = model.transcribe(
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
                text = cc.convert(text)
                print(f"识别结果（fast-whisper zh）：{text}")
            else:
                print("识别结果为空。")

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
