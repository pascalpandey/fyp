import csv
import numpy as np
from dataset import Dataset
from request import Request


class PromptEngineeringDatasetLoader:
    def __init__(self, path, size, request_rate, sigma):
        self._path = path
        self._size = size
        inter_arrival_times = np.random.exponential(1/request_rate, size=size)
        self._request_times = np.round(
            np.cumsum(inter_arrival_times), 3).tolist()
        self.sigma = sigma
    
    def _get_predicted_length(self, pred_mean):
        noise = np.random.normal(0, self.sigma)
        return int(pred_mean * np.exp(noise))

    def load(self):
        dataset = Dataset()
        with open(self._path) as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                if i >= self._size:
                    break
                prompt_len = len(row['Prompt'].split(' '))
                response_len = len(row['Response'].split(' '))
                data = Request(
                    prompt_len,
                    response_len,
                    self._request_times[i],
                    self._get_predicted_length(response_len)
                )
                dataset.add(data)
        return dataset
