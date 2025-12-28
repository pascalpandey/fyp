class FCFSDynamicBatchScheduler:
    def __init__(self, initial_gpu_view):
        self._queue = []
        self._gpu_view = initial_gpu_view

    def queue(self, request_views):
        self._queue.extend(request_views)
    
    def update_gpu_view(self, gpu_view):
        self._gpu_view = gpu_view

    def decide(self):
        # wait if queue and GPU is empty
        if len(self._queue) == 0 and len(self._gpu_view.request_views) == 0:
            return 1, None, None

        # always try to schedule new requests, even if the GPU is not empty
        scheduled_request_ids = []
        while self._gpu_view.is_valid_step() and len(self._queue) > 0:
            request_view = self._queue.pop(0)
            self._gpu_view.schedule(request_view)
            scheduled_request_ids.append(request_view.id)
        if len(scheduled_request_ids) > 0 and not self._gpu_view.is_valid_step():
            self._queue.insert(0, self._gpu_view.preempt_top())
            scheduled_request_ids.pop()
        if scheduled_request_ids:
            return 0, scheduled_request_ids, []

        preempted_request_ids = []
        while not self._gpu_view.is_valid_step():
            request_view = self._gpu_view.preempt_top()
            preempted_request_ids.append(request_view.id)
            self._queue.insert(0, request_view)
        return 0, [], preempted_request_ids
