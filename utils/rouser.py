import os
import pvporcupine


class Rouser:
    def __init__(
        self,
        access_key=None,
        keyword='porcupine',
        sensitivities=None,
        device_index=-1,
    ):
        """
        初始化唤醒词识别器，可自定义参数
        """
        if access_key is None:
            raise ValueError("access_key 不能为空，请提供有效的 Porcupine access_key")
        self.ACCESS_KEY = access_key
        self.MODEL_PATH = None
        self.KEYWORD_PATHS = None
        if keyword == "小智":
            self.MODEL_PATH = os.path.abspath("model/porcupine_params_zh.pv")
            self.KEYWORD_PATHS = [os.path.abspath("model/小智.ppn")]
            keyword = None
        self.SENSITIVITIES = sensitivities or [0.5]
        self.DEVICE_INDEX = device_index  # -1 用系统默认麦克风
        # 初始化porcupine唤醒词识别器
        self.porcupine = pvporcupine.create(
            access_key=self.ACCESS_KEY,
            model_path=self.MODEL_PATH,
            keyword_paths=self.KEYWORD_PATHS,
            sensitivities=self.SENSITIVITIES,
            keywords=[keyword],
        )

    def process(self, pcm):
        return self.porcupine.process(pcm)


if __name__ == "__main__":
    pass
