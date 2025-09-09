from dataset import RequestState
from enum import Enum
from request import VRAMUpdateType


class GPUPhase(Enum):
    PREFILL = 'prefill'
    DECODE = 'decode'


class VRAM:
    def __init__(self, initial_slots):
        self._remaining_slots = initial_slots
        self._used_slots = 0
        self._usage_history = []

    def allocate(self, amount, timestamp):
        if amount > self._remaining_slots:
            raise Exception(f"VRAM slots exceeded")
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


class GPU:
    def __init__(self, vram_slots):
        self._vram = VRAM(vram_slots)
        self._requests = []

    def schedule_requests(self, requests):
        for request in requests:
            self._requests.append(request)

    # From Alladin paper page 4
    # t_prefill = k1 * num_tokens_in_batch + c1
    # t_decode = k2 * num_tokens_in_batch + c2 * size_of_batch + c3
    # for now assume c1 = c2 = c3 = 0, k1 = k2 = 1
    def start_step(self, phase):
        processing_time = 0
        for request in self._requests:
            if phase == GPUPhase.PREFILL and request.state == RequestState.PREFILL:
                processing_time += request.get_next_step_processing_time()

            if phase == GPUPhase.DECODE and request.state == RequestState.DECODE:
                processing_time += request.get_next_step_processing_time()

        return processing_time

    # From Alladin paper page 4
    # kv_size = h * num_tokens + j
    # for now assume h = 1, j = 0
    def end_previous_step(self, timestamp, phase):
        vram_slots_allocated = 0
        vram_slots_freed = 0
        completed_requests = []
        for i, request in enumerate(self._requests):
            if phase == GPUPhase.PREFILL and request.state == RequestState.PREFILL:
                request.step(timestamp)
                vram_slots_allocated += request.get_end_step_vram_update()[1]

            if phase == GPUPhase.DECODE and request.state == RequestState.DECODE:
                request.step(timestamp)
                update_type, update_slots += request.get_end_step_vram_update()
                if update_type == VRAMUpdateType.ALLOCATE:
                    vram_slots_allocated += update_slots
                else:
                    vram_slots_freed += update_slots
                    completed_requests.append(self._requests.pop(i))

        self._vram.allocate(vram_slots_allocated, timestamp)
        self._vram.free(vram_slots_freed, timestamp)
        return completed_requests

    def get_remaining_vram_slots(self):
        return self._vram.get_slots()

    def get_request_views(self):
        return [request.to_request_view() for request in self._requests]


class GPUView:
    def __init__(self, gpu):
        self.remaining_vram_slots = gpu.get_remaining_vram_slots()
        self.request_views = gpu.get_request_views()

    def is_valid_step(self, phase):
        vram_slots_required = 0
        for request in self.requests_views:
            if phase == GPUPhase.PREFILL and request.state == RequestState.PREFILL:
                vram_slots_required += request.get_end_step_vram_update()[1]

            if phase == GPUPhase.DECODE and request.state == RequestState.DECODE:
                update_type, update_slots += request.get_end_step_vram_update()
                if update_type == VRAMUpdateType.ALLOCATE:
                    vram_slots_required += update_slots

        if vram_slots_required > self.remaining_vram_slots():
            return False
        return True
