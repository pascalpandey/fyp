import uuid
from enum import Enum


class RequestState(Enum):
    PENDING = "pending"
    READY = "ready"
    QUEUED = "queued"
    PREFILL = "prefill"
    DECODE = "decode"
    COMPLETED = "completed"


class VRAMUpdateType(Enum):
    ALLOCATE = "allocate"
    FREE = "free"


def calc_start_step_processing_time(request):
    if request.state == RequestState.PREFILL:
        return request.prompt_len
    elif request.state == RequestState.DECODE:
        return request.prompt_len + request.decode_progress
    else:
        raise Exception(
            "calc_start_step_processing_time: request is not in PREFILL or DECODE state")


def calc_end_step_vram_update(request):
    if request.state == RequestState.PREFILL:
        return VRAMUpdateType.ALLOCATE, request.prompt_len
    elif request.state == RequestState.DECODE:
        # because RequestView doesn't have decode_progress field
        if isinstance(request, RequestView) or request.decode_progress + 1 != request.response_len:
            return VRAMUpdateType.ALLOCATE, 1
        return VRAMUpdateType.FREE, request.prompt_len + request.response_len - 1
    else:
        raise Exception(
            "calc_end_step_vram_update: request is not in PREFILL or DECODE state")


def calc_current_vram_usage(request):
    # only requests in DECODE state use VRAM, reuqests on PREFILL state hasn't used VRAM yet
    # as it will only consume VRAM on GPU.end_previous_step()
    if request.state == RequestState.DECODE:
        return request.prompt_len + request.decode_progress
    return 0


class Request:
    def __init__(self, prompt_len, response_len, request_timestamp):
        self.id = uuid.uuid4()
        self.prompt_len = prompt_len
        self.response_len = response_len
        self.request_timestamp = request_timestamp
        self.response_timestamp = None
        self.state = RequestState.PENDING
        self.decode_progress = None

    def __lt__(self, other):
        return self.request_timestamp < other.request_timestamp

    def to_request_view(self):
        return RequestView(self)

    def get_start_step_processing_time(self):
        return calc_start_step_processing_time(self)

    def step(self, timestamp):
        update_type, update_slots = calc_end_step_vram_update(self)
        if self.state == RequestState.PREFILL:
            self.state = RequestState.DECODE
            self.decode_progress = 0
        elif self.state == RequestState.DECODE:
            if self.decode_progress + 1 != self.response_len:
                self.decode_progress += 1
            else:
                self.state = RequestState.COMPLETED
                self.response_timestamp = timestamp
        else:
            raise Exception(
                "Request.step: request is not in PREFILL or DECODE state")
        return update_type, update_slots

    def get_current_vram_usage(self):
        return calc_current_vram_usage(self)


class RequestView:
    def __init__(self, request):
        self.id = request.id
        self.prompt_len = request.prompt_len
        self.request_timestamp = request.request_timestamp
        self.state = request.state
        self.decode_progress = request.decode_progress

    def get_start_step_processing_time(self):
        return calc_start_step_processing_time(self)

    def get_end_step_vram_update(self):
        return calc_end_step_vram_update(self)

    def get_current_vram_usage(self):
        return calc_current_vram_usage(self)
