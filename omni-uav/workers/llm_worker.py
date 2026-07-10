from typing import List

from PyQt5 import QtCore

from utils import LlmClient


class LlmWorker(QtCore.QThread):
    completed = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, client: LlmClient, prompt: str, image_paths: List[str]):
        super().__init__()
        self.client = client
        self.prompt = prompt
        self.image_paths = image_paths

    def run(self):
        try:
            response = self.client.chat(self.prompt, self.image_paths)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(response)
