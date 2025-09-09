import heapq
import uuid
from enum import Enum


class State(Enum):
    PENDING = 'pending'
    READY = 'ready'
    QUEUED = 'queued'
    PREFILL = 0
    DECODE = 'decode'
    COMPLETED = 'completed'


class Request:
    def __init__(self, prompt_len, response_len, request_timestamp):
        self.id = uuid.uuid4()
        self.prompt_len = prompt_len
        self.response_len = response_len
        self.request_timestamp = request_timestamp
        self.response_timestamp = None
        self.state = State.PENDING

    def __lt__(self, other):
        return self.request_timestamp < other.request_timestamp

    def to_request_view(self):
        return RequestView(self)


class RequestView:
    def __init__(self, request):
        self.id = request.id
        self.prompt_len = request.prompt_len
        self.request_timestamp = request.request_timestamp
        self.state = request.state


class Dataset:
    def __init__(self):
        self._requests = {}
        self._pending_heap = []
        self._completed_requests_count = 0

    def add(self, request):
        self._requests[request.id] = request
        heapq.heappush(self._pending_heap, request)

    def get_ready_requests_view(self, timestamp):
        ready_requests = []
        while self._pending_heap and self._pending_heap[0].request_timestamp <= timestamp:
            request = heapq.heappop(self._pending_heap)
            request.state = State.READY
            ready_requests.append(request.to_request_view())
        return ready_requests

    def completed_all_requests(self):
        return self._completed_requests_count == len(self._requests)

    def complete_requests(self, completed_requests, timestamp):
        for request in completed_requests:
            self._requests[request.id].state = State.COMPLETED
            self._requests[request.id].response_timestamp = timestamp
        self._completed_requests_count += len(completed_requests)
