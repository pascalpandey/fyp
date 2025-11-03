import json
import numpy as np
import tiktoken
from dataset import Dataset
from request import Request


class ShareGPTDatasetLoader:
    def __init__(self, path, size, conversation_rate, prompt_rate, sigma, max_context_window):
        self._path = path
        self._size = size
        conversation_inter_arrival_times = np.random.exponential(
            1/conversation_rate, size=size)
        self._conversation_times = np.round(
            np.cumsum(conversation_inter_arrival_times), 3).tolist()
        self._sigma = sigma
        self._prompt_rate = prompt_rate
        self._encoding = tiktoken.encoding_for_model("gpt-4")
        self._max_context_window = max_context_window

    def _get_predicted_length(self, actual_response_len):
        noise = np.random.normal(0, self._sigma)
        return int(actual_response_len * np.exp(noise))

    def load(self):
        dataset = Dataset()
        with open(self._path) as file:
            data = json.load(file)
            for i, row in enumerate(data):
                if i >= self._size:
                    break

                conversation = row['conversations']

                conversation_start = self._conversation_times[i]
                prompt_count = sum(
                    1 for message in conversation if message['from'] == 'gpt')
                prompt_inter_arrival_times = np.random.exponential(
                    1/self._prompt_rate, size=prompt_count)
                prompt_times = np.round(
                    conversation_start + np.cumsum(prompt_inter_arrival_times), 3)

                prompt_idx = 0
                prompt_len = 0
                for j, message in enumerate(conversation):
                    source = message['from']
                    # allow special tokens and count it
                    token_count = len(self._encoding.encode(message['value'], allowed_special={"<|endoftext|>"}))
                    if source == 'human':
                        prompt_len += token_count
                    elif source == 'gpt':
                        # ignore if no response from GPT
                        if token_count == 0:
                            continue
                        # some conversations start with gpt, in that case append the first
                        # gpt message to the first prompt
                        if j != 0:
                            predicted_response_len = self._get_predicted_length(
                                token_count)
                            data = Request(
                                f"Conv {i}, Req {prompt_idx}",
                                min(self._max_context_window, prompt_len),
                                token_count,
                                prompt_times[prompt_idx],
                                predicted_response_len
                            )
                            dataset.add(data)
                            prompt_idx += 1
                        prompt_len += token_count
        return dataset
