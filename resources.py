from dataset import State


class ResourceUnit:
    def __init__(self, name, initial_slots):
        self._name = name
        self._remaining_slots = initial_slots
        self._used_slots = 0
        self._usage_history = []

    def allocate(self, amount, timestamp):
        if amount > self._remaining_slots:
            raise Exception(f"{self._name} slots exceeded")
        self._remaining_slots -= amount
        self._used_slots += amount
        if len(self._usage_history) <= timestamp:
            self._usage_history.append(self._used_slots)
        else:
            self._usage_history[timestamp] += self._used_slots

    def free(self, amount, timestamp):
        self._remaining_slots += amount
        self._used_slots -= amount
        self._usage_history[timestamp] -= amount

    def get_remaining_slots(self):
        return self._remaining_slots

    def get_used_slots(self):
        return self._used_slots


class Resources:
    def __init__(self, gpu_slots, vram_slots):
        self._gpu = ResourceUnit("GPU", gpu_slots)
        self._vram = ResourceUnit("VRAM", vram_slots)
        self._requests = []

    def schedule_requests(self, requests):
        for request in requests:
            self._requests.append(request)

    def start_step(self, timestamp):
        gpu_slots_required = 0
        for request in self._requests:
            if request.state == State.PREFILL:
                gpu_slots_required += int((request.prompt_len - 1)
                                          * request.prompt_len / 2)
            elif request.state == State.DECODE:
                gpu_slots_required += request.state.value + request.prompt_len
            else:
                raise Exception("invalid request state")
        self._gpu.allocate(gpu_slots_required, timestamp)

    def end_previous_step(self, timestamp):
        gpu_slots_freed = 0
        vram_slots_required = 0
        vram_slots_freed = 0
        completed_requests = []
        for i, request in enumerate(self._requests):
            if request.state == State.PREFILL:
                gpu_slots_freed += int((request.prompt_len - 1)
                                       * request.prompt_len / 2)
                vram_slots_required += request.prompt_len
                request.state = State.DECODE
                request.state.value = 0
            elif request.state == State.DECODE:
                gpu_slots_freed += request.state.value + request.prompt_len
                if request.state.value + 1 != len(request.response_len):
                    vram_slots_required += 1
                    request.state.value += 1
                else:
                    vram_slots_freed += request.prompt_len + request.request_len - 1
                    request.state = State.COMPLETED
                    request.response_timestamp = timestamp
                    completed_requests.append(self._requests.pop(i))
            else:
                raise Exception("invalid request state")
        self._gpu.free(gpu_slots_freed, timestamp)
        self._vram.allocate(vram_slots_required, timestamp)
        self._vram.free(vram_slots_freed, timestamp)
        return completed_requests

    def get_remaining_gpu_slots(self):
        return self._gpu.get_slots()

    def get_remaining_vram_slots(self):
        return self._vram.get_slots()

    def get_request_views(self):
        return [request.to_request_view() for request in self._requests]


class ResourcesView:
    def __init__(self, resources):
        self.remining_gpu_slots = resources.get_remaining_gpu_slots()
        self.remaining_vram_slots = resources.get_remaining_vram_slots()
        self.request_views = resources.get_request_views()

    def check_valid_step(self):
        gpu_slots_required = 0
        vram_slots_required = 0
        for request in self.requests_views:
            if request.state == State.PREFILL:
                gpu_slots_required += int((request.prompt_len - 1)
                                          * request.prompt_len / 2)
                vram_slots_required += request.prompt_len
            elif request.state == State.DECODE:
                gpu_slots_required += request.state.value + request.prompt_len
                if request.state.value + 1 == len(request.response_len):
                    vram_slots_required += 1
            else:
                raise Exception("invalid request state")
        if gpu_slots_required > self.remaining_gpu_slots() or \
                vram_slots_required > self.remaining_vram_slots():
            return False
        return True
