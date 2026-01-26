from dataset import Dataset


class TestDatasetLoader:
    def __init__(self, requests):
        self._requests = requests

    def load(self):
        dataset = Dataset()
        for data in self._requests:
            dataset.add(data)
        return dataset
