from collections import deque


class Data:
    def __init__(self, prompt, response):
        self.prompt = prompt
        self.response = response


class Dataset:
    def __init__(self):
        self._data = deque([])
        self._idx = 0

    def add(self, data):
        self._data.append(data)

    def popleft(self):
        return self._data[self._idx]
