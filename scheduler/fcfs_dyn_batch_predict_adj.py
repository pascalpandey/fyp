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


class FCFSDynamicBatchPredictAdjScheduler:
    def __init__(self, initial_gpu_view):
        self._queue = []
        self._gpu_view = initial_gpu_view
        self.pred_adjustment = {}

    def queue(self, request_views):
        self._queue.extend(request_views)
    
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
        return 0, scheduled_request_ids, preempted_request_ids
