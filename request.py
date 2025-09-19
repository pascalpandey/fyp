import uuid
from enum import Enum


class RequestState(Enum):
    PENDING = "pending"
    READY = "ready"
    PREFILL = "prefill"
    DECODE = "decode"
    COMPLETED = "completed"


class VRAMUpdateType(Enum):
    ALLOCATE = "allocate"
    FREE = "free"


def _calc_start_step_processing_time(request):
    if request.state == RequestState.PREFILL:
        return request._prompt_len
    elif request.state == RequestState.DECODE:
        return request._prompt_len + request._decode_progress
    else:
        raise Exception(
            "calc_start_step_processing_time: request is not in PREFILL or DECODE state")


def _calc_end_step_vram_update(request):
    if request.state == RequestState.PREFILL:
        return VRAMUpdateType.ALLOCATE, request._prompt_len
    elif request.state == RequestState.DECODE:
        # because RequestView doesn't have response_len field and RequestView will never be in COMPLETE state
        if isinstance(request, RequestView) or request._decode_progress + 1 != request._response_len:
            return VRAMUpdateType.ALLOCATE, 1
        return VRAMUpdateType.FREE, request._prompt_len + request._response_len - 1
    else:
        raise Exception(
            "calc_end_step_vram_update: request is not in PREFILL or DECODE state")


def _calc_current_vram_usage(request):
    # only requests in DECODE state use VRAM, reuqests on PREFILL state hasn't used VRAM yet
    # as it will only consume VRAM on GPU.end_previous_step()
    if request.state == RequestState.DECODE:
        return request._prompt_len + request._decode_progress
    return 0


class Request:
    def __init__(self, prompt_len, response_len, request_timestamp):
        self.id = uuid.uuid4()
        self._prompt_len = prompt_len
        self._response_len = response_len
        self.request_timestamp = request_timestamp
        self.response_timestamp = None
        self.state = RequestState.PENDING
        self._decode_progress = None
        self.history = {self.request_timestamp: RequestState.READY}

    def __lt__(self, other):
        return self.request_timestamp < other.request_timestamp

    def to_request_view(self):
        return RequestView(self)

    def get_start_step_processing_time(self):
        return _calc_start_step_processing_time(self)

    def step(self, timestamp):
        match self.state:
            case RequestState.PENDING:
                # append history handled during initialization because timestamp 
                # param here is not accurate
                self.state = RequestState.READY
            case RequestState.READY:
                self.state = self.history[timestamp] = RequestState.PREFILL
            case RequestState.PREFILL | RequestState.DECODE:
                update_type, update_slots = _calc_end_step_vram_update(self)
                if self.state == RequestState.PREFILL:
                    self.state = self.history[timestamp] = RequestState.DECODE
                    self._decode_progress = 0
                elif self.state == RequestState.DECODE:
                    if self._decode_progress + 1 != self._response_len:
                        self._decode_progress += 1
                    else:
                        self.state = self.history[timestamp] = RequestState.COMPLETED
                        self.response_timestamp = timestamp
                return update_type, update_slots
    
    def reset_to_ready(self, timestamp):
        if self.state not in [RequestState.PREFILL, RequestState.DECODE]:
            raise Exception(f"Request.reset_to_ready: cannot reset state to ready from {self.state}")
        self.state = self.history[timestamp] = RequestState.READY

    def get_current_vram_usage(self):
        return _calc_current_vram_usage(self)


class RequestView:
    def __init__(self, request):
        self.id = request.id
        self._prompt_len = request._prompt_len
        self.request_timestamp = request.request_timestamp
        self.state = request.state
        self._decode_progress = request._decode_progress

    def get_start_step_processing_time(self):
        return _calc_start_step_processing_time(self)

    def get_end_step_vram_update(self):
        return _calc_end_step_vram_update(self)

    def get_current_vram_usage(self):
        return _calc_current_vram_usage(self)

    @property
    def prompt_len(self):
        return self._prompt_len

    @property
    def decode_progress(self):
        return self._decode_progress
    
    @prompt_len.setter
    def prompt_len(self, value):
        self._prompt_len = value

    @decode_progress.setter
    def decode_progress(self, value):
        self.state = value
