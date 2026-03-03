import heapq
import numpy as np

class RequestHeapItem:
    def __init__(self, req):
        object.__setattr__(self, "req", req)

    def __getattr__(self, name):
        return getattr(self.req, name)

    def __setattr__(self, name, value):
       setattr(self.req, name, value)
    
    def __lt__(self, other):
        return self.predicted_response_len < other.predicted_response_len


class SJFDynamicBatchPredictAdjIQRScheduler:
    def __init__(self, initial_gpu_view):
        self._queue = []
        self._gpu_view = initial_gpu_view
        self._prediction_adjustment = 0
        self._previous_requests = {}
        self._completed_requests = 0
        self._actual = []

    def queue(self, request_views):
        for request_view in request_views:
            heapq.heappush(self._queue, RequestHeapItem(request_view))
    
    def update_gpu_view(self, gpu_view):
        self._gpu_view = gpu_view
        scheduled_request_ids = set([request_view.id for request_view in self._gpu_view.request_views])
        for req_id in self._previous_requests:
            if req_id not in scheduled_request_ids:
                decode_progress, predicted_response_len = self._previous_requests[req_id]
                actual_len = decode_progress + 1
                self._actual.append(actual_len)
        self._previous_requests = {}
        for request_view in self._gpu_view.request_views:
            self._previous_requests[request_view.id] = (request_view.decode_progress, request_view.predicted_response_len)
            if self._actual:
                q1, q3 = np.percentile(self._actual, [25, 75])
                iqr = q3 - q1
                if request_view.predicted_response_len < q1 - iqr or request_view.predicted_response_len > q3 + iqr:
                    request_view.predicted_response_len = request_view.decode_progress + 1
            request_view.predicted_response_len = max(request_view.predicted_response_len, request_view.decode_progress + 1)

    def decide(self):
        # wait if queue and GPU is empty
        if len(self._queue) == 0 and len(self._gpu_view.request_views) == 0:
            return 1, None, None

        # always try to schedule new requests, even if the GPU is not empty
        scheduled_request_ids = []
        while self._gpu_view.is_valid_step_with_predict() and len(self._queue) > 0:
            request_heap_item = heapq.heappop(self._queue)
            self._gpu_view.schedule(request_heap_item.req)
            scheduled_request_ids.append(request_heap_item.id)
        if len(scheduled_request_ids) > 0 and not self._gpu_view.is_valid_step_with_predict():
            req = self._gpu_view.preempt_top()
            heapq.heappush(self._queue, RequestHeapItem(req))
            scheduled_request_ids.pop()
        
        self._gpu_view.request_views.sort(key= lambda x: x.get_remaining_processing_time())

        preempted_request_ids = []
        while not self._gpu_view.is_valid_step_with_predict():
            request_view = self._gpu_view.preempt_top()
            preempted_request_ids.append(request_view.id)
            heapq.heappush(self._queue, RequestHeapItem(request_view))  
        
        for req_id in preempted_request_ids:
            del self._previous_requests[req_id]

        return 0, scheduled_request_ids, preempted_request_ids
