import winsound
import sys
import simpleaudio as sa


class Player:
    def __init__(self):
        self.WAKE_VOICE_PATH = 'music/zai.wav'
        self.OK_VOICE_PATH = 'music/ok.wav'
        self.SORRY_VOICE_PATH = 'music/sorry.wav'

    def play_voice(self, block=True, type="zai"):
        path = self.WAKE_VOICE_PATH
        if type == "ok":
            path = self.OK_VOICE_PATH
        elif type == "sorry":
            path = self.SORRY_VOICE_PATH
        try:
            if sys.platform.startswith("win"):
                winsound.PlaySound(path,
                                   winsound.SND_FILENAME)
                return
            else:
                _wake_wave = sa.WaveObject.from_wave_file(path)
                play_obj = _wake_wave.play()  # 使用self._wake_wave
                if block:
                    play_obj.wait_done()
        except BaseException as e:
            print(f"[提示音] 播放失败：{type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)


if __name__ == '__main__':
    p = Player()
    p.play_voice(True, "zai")
