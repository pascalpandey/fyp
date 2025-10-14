import uuid
from enum import Enum


class RequestState(Enum):
    PENDING = "pending"  # request not yet sent by client
    READY = "ready"  # request is sent and is currently queued

    # request is scheduled to GPU
    # process_stage tracks whether the request is in prefill/decode stage
    # decode_progress tracks number of tokens decoded so far
    SCHEDULED = "scheduled"

    COMPLETED = "completed"  # request is completed and the full response has been generated


class ProcessStage(Enum):
    DECODE = "decode"
    PREFILL = "prefill"


class VRAMUpdateType(Enum):
    ALLOCATE = "allocate"
    FREE = "free"


class ProcessHistoryState(Enum):
    READY = "ready"
    DECODE = "decode"
    PREFILL = "prefill"
    IDLE = "idle"
    COMPLETED = "completed"


def _calc_end_step_vram_update(request):
    if request.process_stage == ProcessStage.PREFILL:
        return VRAMUpdateType.ALLOCATE, request._prompt_len
    elif request.process_stage == ProcessStage.DECODE:
        # because RequestView doesn't have response_len field and RequestView will never be in COMPLETE state
        if isinstance(request, RequestView) or request._decode_progress + 1 != request._response_len:
            return VRAMUpdateType.ALLOCATE, 1
        return VRAMUpdateType.FREE, request._prompt_len + request._response_len - 1
    else:
        raise Exception(
            "calc_end_step_vram_update: request is not in PREFILL or DECODE state")


def _calc_current_vram_usage(request):
    # only requests in DECODE process_stage use VRAM, requests on PREFILL process_stage hasn't used VRAM yet
    # as it will only consume VRAM on GPU.end_previous_step()
    if request.process_stage == ProcessStage.DECODE:
        return request._prompt_len + request._decode_progress
    return 0


class Request:
    def __init__(self, prompt_len, response_len, request_timestamp, predicted_response_len):
        self.id = uuid.uuid4()
        self._prompt_len = prompt_len
        self._response_len = response_len
        self.request_timestamp = request_timestamp
        self.predicted_response_len = predicted_response_len
        self.response_timestamp = None
        self.state = RequestState.PENDING
        self.process_stage = None
        self._decode_progress = 0
        self.process_history = {
            self.request_timestamp: RequestState.READY}

    def __lt__(self, other):
        return self.request_timestamp < other.request_timestamp

    def to_request_view(self):
        return RequestView(self)

    # def get_start_step_processing_time(self):
    #     if self.process_stage == ProcessStage.PREFILL:
    #         return self._prompt_len
    #     elif self.process_stage == ProcessStage.DECODE:
    #         return self._prompt_len + self._decode_progress
    #     else:
    #         raise Exception(
    #             "get_start_step_processing_time: request process_stage is not in PREFILL or DECODE")

    def step(self, timestamp=None):
        match self.state:
            case RequestState.PENDING:
                # append history handled during initialization, no timestamp param passed
                self.state = RequestState.READY

            case RequestState.READY:
                self.state = RequestState.SCHEDULED
                self.process_stage = ProcessStage.PREFILL

            case RequestState.SCHEDULED:
                update_type, update_slots = _calc_end_step_vram_update(self)
                if self.process_stage == ProcessStage.PREFILL:
                    self.process_stage = ProcessStage.DECODE
                    self._decode_progress = 0

                elif self.process_stage == ProcessStage.DECODE:
                    if self._decode_progress + 1 != self._response_len:
                        self._decode_progress += 1
                    else:
                        self.process_stage = RequestState.COMPLETED
                        self.add_process_history(
                            timestamp, ProcessHistoryState.COMPLETED)
                        self.response_timestamp = timestamp

                return update_type, update_slots

    def preempt(self, timestamp):
        if self.state != RequestState.SCHEDULED:
            raise Exception(
                f"Request.preempt: cannot preempt from {self.state}")
        self.state = RequestState.READY
        self.add_process_history(timestamp, ProcessHistoryState.READY)

    def add_process_history(self, timestamp, process_history_state):
        self.process_history[timestamp] = process_history_state

    def get_current_vram_usage(self):
        return _calc_current_vram_usage(self)


class RequestView:
    def __init__(self, request):
        self.id = request.id
        self._prompt_len = request._prompt_len
        self.request_timestamp = request.request_timestamp
        self.state = request.state
        self.process_stage = request.process_stage
        self._decode_progress = request._decode_progress
        self.predicted_response_len = request.predicted_response_len

    def get_end_step_vram_update(self):
        return _calc_end_step_vram_update(self)

    def get_current_vram_usage(self):
        return _calc_current_vram_usage(self)
    
    def get_total_predicted_vram_usage(self):
        return self._prompt_len + self.predicted_response_len - 1

    # public access aliases, needed because common utility functions with Request uses
    # _prompt_len and _decode_progress
    @property
    def prompt_len(self):
        return self._prompt_len

    @property
    def decode_progress(self):
        return self._decode_progress
