import csv
import numpy as np
from dataset import Dataset
from request import Request


class PromptEngineeringDatasetLoader:
    def __init__(self, path, size, request_rate):
        self._path = path
        self._size = size
        inter_arrival_times = np.random.exponential(1/request_rate, size=size)
        self._request_times = np.round(
            np.cumsum(inter_arrival_times), 3).tolist()

    def load(self):
        dataset = Dataset()
        with open(self._path) as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                if i >= self._size:
                    break
                data = Request(
                    len(row['Prompt'].split(' ')),
                    len(row['Response'].split(' ')),
                    self._request_times[i]
                )
                dataset.add(data)
        return dataset
