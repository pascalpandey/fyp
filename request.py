import uuid
from enum import Enum


class RequestState(Enum):
    PENDING = "pending"
    READY = "ready"
    QUEUED = "queued"
    PREFILL = 0
    DECODE = "decode"
    COMPLETED = "completed"


class VRAMUpdateType(Enum):
    ALLOCATE = "allocate"
    FREE = "free"


def calc_start_step_processing_time(request):
    if request.state == RequestState.PREFILL:
        return request.prompt_len
    elif request.state == RequestState.DECODE:
        return request.prompt_len + request.state.value
    else:
        raise Exception(
            "calc_start_step_processing_time: request is not in PREFILL or DECODE state")


def calc_end_step_vram_update(request):
    if request.state == RequestState.PREFILL:
        return VRAMUpdateType.ALLOCATE, request.prompt_len
    elif request.state == RequestState.DECODE:
        if request.state.value + 1 != len(request.response_len):
            return VRAMUpdateType.ALLOCATE, 1
        return VRAMUpdateType.FREE, request.prompt_len + request.request_len - 1
    else:
        raise Exception(
            "calc_end_step_vram_update: request is not in PREFILL or DECODE state")


class Request:
    def __init__(self, prompt_len, response_len, request_timestamp):
        self.id = uuid.uuid4()
        self.prompt_len = prompt_len
        self.response_len = response_len
        self.request_timestamp = request_timestamp
        self.response_timestamp = None
        self.state = RequestState.PENDING

    def __lt__(self, other):
        return self.request_timestamp < other.request_timestamp

    def to_request_view(self):
        return RequestView(self)

    def get_start_step_processing_time(self):
        return calc_start_step_processing_time(self)

    def get_end_step_vram_update(self):
        return calc_end_step_vram_update(self)

    def step(self, timestamp):
        if self.state == RequestState.PREFILL:
            self.state = RequestState.DECODE
            self.state.value = 0
        elif self.state == RequestState.DECODE:
            if self.state.value + 1 != len(self.response_len):
                self.state.value += 1
            else:
                self.state = RequestState.COMPLETED
                self.response_timestamp = timestamp
        else:
            raise Exception("step: request is not in PREFILL or DECODE state")


class RequestView:
    def __init__(self, request):
        self.id = id
        self.prompt_len = request.prompt_len
        self.request_timestamp = request.request_timestamp
        self.state = request.state

    def get_start_step_processing_time(self):
        return calc_start_step_processing_time(self)

    def get_end_step_vram_update(self):
        return calc_end_step_vram_update(self)
