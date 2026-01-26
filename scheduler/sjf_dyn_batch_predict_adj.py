import heapq


class RequestHeapItem:
    def __init__(self, req):
        object.__setattr__(self, "req", req)

    def __getattr__(self, name):
        return getattr(self.req, name)

    def __setattr__(self, name, value):
       setattr(self.req, name, value)
    
    def __lt__(self, other):
        return self.predicted_response_len < other.predicted_response_len


class ViolationCounter:
    def __init__(self, predicted_response_len):
        self.violation_count = 0
        self.initial_predicted_response_len = predicted_response_len
        self.adjusted_predicted_response_len = predicted_response_len
    
    def check_violation(self, current_decode_progress):
        if current_decode_progress == self.adjusted_predicted_response_len:
            self.violation_count += 1
            self.adjusted_predicted_response_len += (0.2 ** self.violation_count) * self.initial_predicted_response_len
        return self.adjusted_predicted_response_len


class SJFDynamicBatchPredictAdjScheduler:
    def __init__(self, initial_gpu_view):
        self._queue = []
        self._gpu_view = initial_gpu_view
        self.pred_adjustment = {}

    def queue(self, request_views):
        for request_view in request_views:
            heapq.heappush(self._queue, RequestHeapItem(request_view))
    
    def update_gpu_view(self, gpu_view):
        self._gpu_view = gpu_view
        for request_view in self._gpu_view.request_views:
            if request_view.id not in self.pred_adjustment:
                self.pred_adjustment[request_view.id] = ViolationCounter(request_view.predicted_response_len)
            else:
                request_view.predicted_response_len = self.pred_adjustment[request_view.id].check_violation(request_view.decode_progress)

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
        return 0, scheduled_request_ids, preempted_request_ids
