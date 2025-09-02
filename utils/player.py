import winsound
import sys
import simpleaudio as sa


class Player:
    def __init__(self):
        self.WAKE_VOICE_PATH = 'music/zai.wav'
        self._wake_wave = None  # 初始化为None

    def play_wake_voice(self, block=True):
        try:
            if sys.platform.startswith("win"):
                winsound.PlaySound(self.WAKE_VOICE_PATH,
                                   winsound.SND_FILENAME)
                return
            else:
                if self._wake_wave is None:
                    self._wake_wave = sa.WaveObject.from_wave_file(
                        self.WAKE_VOICE_PATH)
                play_obj = self._wake_wave.play()  # 使用self._wake_wave
                if block:
                    play_obj.wait_done()
        except BaseException as e:
            print(f"[提示音] 播放失败：{type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)


if __name__ == '__main__':
    p = Player()
    p.play_wake_voice(True)
