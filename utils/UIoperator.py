import pyautogui
import subprocess
import pyperclip
import time


class UIOperator:
    def __init__(self, app_path=None):
        if app_path is None:
            app_path = r"C:\Users\11541\AppData\Local\UiTars\UI-TARS.exe"
        self.app_path = app_path

    def run_and_ask(self, question):
        pyperclip.copy(question)
        subprocess.Popen(self.app_path)
        time.sleep(2)  # 等UI-TARS加载

        # 1. 固定坐标点击“Use Local Computer”按钮
        pyautogui.click(x=715, y=809)
        # 等待该位置像素不是黑色
        while True:
            pixel = pyautogui.pixel(715, 809)
            if sum(pixel) > 200:  # 偏黑色判断，像素值总和小于等于200
                break
            time.sleep(0.2)

        time.sleep(0.5)  # 等待“Use Local Computer”加载完成

        # 2. 输入问题
        pyautogui.hotkey("ctrl", "v")

        time.sleep(0.5)
        # 3. 定位并点击小飞机按钮
        pyautogui.click(x=800, y=852)
        print("T-TARS开始运行")


# 调用示例
if __name__ == "__main__":
    client = UIOperator()
    client.run_and_ask("休眠~")
