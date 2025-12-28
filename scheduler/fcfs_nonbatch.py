class FCFSNonBatchScheduler:
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

        if len(self._gpu_view.request_views) == 0:
            return 0, [self._queue.pop(0).id], []

        return 0, [], []
