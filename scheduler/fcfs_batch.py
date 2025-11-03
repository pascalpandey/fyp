from request import RequestState, ProcessStage


class FCFSBatchScheduler:
    def __init__(self, initial_gpu_view):
        self._queue = []
        self._gpu_view = initial_gpu_view

    def queue(self, request_views):
        self._queue.extend(request_views)

    def decide(self):
        # wait if queue and GPU is empty
        if len(self._queue) == 0 and len(self._gpu_view.request_views) == 0:
            return 1, None, None

        # if GPU is empty, pack as many requests as possible to the GPU, if it overflows insert back to the queue
        if len(self._gpu_view.request_views) == 0:
            while self._gpu_view.is_valid_step() and len(self._queue) > 0:
                request_view = self._queue.pop(0)
                self._gpu_view.schedule(request_view)
            if not self._gpu_view.is_valid_step():
                self._queue.insert(0, self._gpu_view.preempt_top())
            return 0, [request_view.id for request_view in self._gpu_view.request_views], []

        preempted_request_ids = []
        while not self._gpu_view.is_valid_step():
            request_view = self._gpu_view.preempt_top()
            preempted_request_ids.append(request_view.id)
            self._queue.insert(0, request_view)
        return 0, [], preempted_request_ids

    def update_gpu_view(self, gpu_view):
        self._gpu_view = gpu_view
