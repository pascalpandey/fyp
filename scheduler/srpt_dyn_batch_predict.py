import heapq
from request import RequestState, ProcessStage


class RequestHeapItem:
    def __init__(self, req, key):
        object.__setattr__(self, "req", req)
        object.__setattr__(self, "key", key)
        preemption_cost = max(1, self.predicted_response_len - 2 * self.decode_progress - 2)
        object.__setattr__(self.req, "preemption_cost", preemption_cost)

    def __getattr__(self, name):
        return getattr(self.req, name)

    def __setattr__(self, name, value):
        setattr(self.req, name, value)
    
    def get_value_of_key(self):
        return getattr(self.req, self.key)

    def __lt__(self, other):
        return self.get_value_of_key() < other.get_value_of_key()


class SRPTDynamicBatchPredictScheduler:
    def __init__(self, initial_gpu_view):
        self._queue = []
        self.update_gpu_view(initial_gpu_view)

    def queue(self, request_views):
        for request_view in request_views:
            heapq.heappush(self._queue, RequestHeapItem(request_view, key="predicted_response_len"))

    def decide(self):
        # wait if queue and GPU is empty
        if len(self._queue) == 0 and len(self._gpu_view.request_views) == 0:
            return 1, None, None

        # always try to schedule new requests, even if the GPU is not empty
        scheduled_request_ids = []
        while self._gpu_view.is_valid_step_with_predict() and len(self._queue) > 0:
            request_view = heapq.heappop(self._queue)
            request_view.state = RequestState.SCHEDULED
            request_view.process_stage = ProcessStage.PREFILL
            scheduled_request_ids.append(request_view.id)

            self._gpu_view.request_views.append(request_view)
            heapq.heappush(self._gpu_preemption_cost, RequestHeapItem(request_view, key="preemption_cost"))
        if len(scheduled_request_ids) > 0 and not self._gpu_view.is_valid_step():
            heapq.heappush(self._queue, self._gpu_view.request_views.pop())
            scheduled_request_ids.pop()
        
        # only preempt if:
        # (to_be_scheduled_request.predicted_response_len + 1 + to_be_preempted_request.decode_progress + 1) <
        # (to_be_preempted_request.predicted_response_len - to_be_preempted_request.decode_progress)
        # which simplifies to
        # to_be_scheduled_request.predicted_response_len <
        # to_be_preempted_request.predicted_response_len - 2 * to_be_preempted_request.decode_progress - 2
        preempted_request_ids = []
        while len(self._queue) > 0 and len(self._gpu_preemption_cost) > 0 and \
            self._queue[0].get_value_of_key() < self._gpu_preemption_cost[0].get_value_of_key():
            preempted_request = self._gpu_view.request_views.pop()
            preempted_request.state = RequestState.READY
            preempted_request.process_stage = None
            preempted_request_ids.append(preempted_request.id)
            heapq.heappush(self._queue, RequestHeapItem(preempted_request, key="predicted_response_len"))

            scheduled_request = heapq.heappop(self._queue)
            scheduled_request.state = RequestState.SCHEDULED
            scheduled_request.process_stage = ProcessStage.PREFILL
            scheduled_request_ids.append(scheduled_request.id)

        while not self._gpu_view.is_valid_step():
            request_view = self._gpu_view.request_views.pop()
            request_view.state = RequestState.READY
            request_view.process_stage = None
            preempted_request_ids.append(request_view.id)
            heapq.heappush(self._queue, RequestHeapItem(request_view, key="predicted_response_len"))
            self._gpu_view.remaining_vram_slots += request_view.get_current_vram_usage()
        return 0, scheduled_request_ids, preempted_request_ids

    def update_gpu_view(self, gpu_view):
        self._gpu_view = gpu_view
        self._gpu_preemption_cost = []
        for request_view in self._gpu_view.request_views:
            heapq.heappush(self._gpu_preemption_cost, RequestHeapItem(request_view, key="preemption_cost"))
        return 
