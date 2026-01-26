class FCFSDynamicBatchPredictAdjAvgScheduler:
    def __init__(self, initial_gpu_view):
        self._queue = []
        self._gpu_view = initial_gpu_view
        self._prediction_adjustment = 0
        self._previous_requests = {}
        self._completed_requests = 0

    def queue(self, request_views):
        self._queue.extend(request_views)
    
    def update_gpu_view(self, gpu_view):
        self._gpu_view = gpu_view
        scheduled_request_ids = set([request_view.id for request_view in self._gpu_view.request_views])
        for req_id in self._previous_requests:
            if req_id not in scheduled_request_ids:
                decode_progress, predicted_response_len = self._previous_requests[req_id]
                self._prediction_adjustment = (self._prediction_adjustment * self._completed_requests + (decode_progress + 1 - predicted_response_len)) // (self._completed_requests + 1)
                self._completed_requests += 1
        self._previous_requests = {}
        for request_view in self._gpu_view.request_views:
            self._previous_requests[request_view.id] = (request_view.decode_progress, request_view.predicted_response_len)
            request_view.predicted_response_len += self._prediction_adjustment


    def decide(self):
        # wait if queue and GPU is empty
        if len(self._queue) == 0 and len(self._gpu_view.request_views) == 0:
            return 1, None, None

        # try to schedule requests using the total predicted VRAM usage
        scheduled_request_ids = []
        while self._gpu_view.is_valid_step_with_predict() and len(self._queue) > 0:
            request_view = self._queue.pop(0)
            self._gpu_view.schedule(request_view)
            scheduled_request_ids.append(request_view.id)
        if len(scheduled_request_ids) > 0 and not self._gpu_view.is_valid_step_with_predict():
            self._queue.insert(0, self._gpu_view.preempt_top())
            scheduled_request_ids.pop()

        preempted_request_ids = []
        while not self._gpu_view.is_valid_step_with_predict():
            request_view = self._gpu_view.preempt_top()
            preempted_request_ids.append(request_view.id)
            self._queue.insert(0, request_view)
        
        for req_id in preempted_request_ids:
            del self._previous_requests[req_id]

        return 0, scheduled_request_ids, preempted_request_ids
