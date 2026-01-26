import csv
import numpy as np
import tiktoken
from dataset import Dataset
from request import Request


class PromptEngineeringDatasetLoader:
    def __init__(self, path, size, request_rate, sigma, fixed_prefill=False):
        self._path = path
        self._size = size

        self._rng = np.random.default_rng(42)
        inter_arrival_times = self._rng.exponential(1/request_rate, size=size)
        self._request_times = np.round(
            np.cumsum(inter_arrival_times), 3).tolist()
 
        self._sigma = sigma
        self._fixed_prefill = fixed_prefill
        self._encoding = tiktoken.encoding_for_model("gpt-4")
    
    def _get_predicted_length(self, actual_response_len):
        noise = self._rng.normal(0, self._sigma)
        return int(actual_response_len * np.exp(noise))

    def load(self):
        dataset = Dataset()
        with open(self._path) as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                if i >= self._size:
                    break
                prompt_len = 1 if self._fixed_prefill else len(self._encoding.encode(row['Prompt']))
                # no need to add +1 for end of text token because this is accounted for by prefill stage taking
                # one time step, which generates the first output token
                response_len = len(self._encoding.encode(row['Response']))
                predicted_response_len = self._get_predicted_length(response_len)
                data = Request(
                    f"Request {i}",
                    prompt_len,
                    response_len,
                    self._request_times[i],
                    predicted_response_len
                )
                dataset.add(data)
        return dataset
