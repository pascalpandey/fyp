from gpu import GPUPhase
from request import RequestState


class FCFSScheduler:
    def __init__(self, initial_gpu_view):
        self._queue = []
        self._gpu_view = initial_gpu_view

    def queue(self, request_views):
        self._queue.extend(request_views)

    def decide(self):
        if len(self._queue) == 0 and len(self._gpu_view.request_views) == 0:
            return 1, None, None, None

        # if GPU is empty, pack as many requests as possible to the GPU, if it overflows insert back to the queue
        if len(self._gpu_view.request_views) == 0:
            while self._gpu_view.is_valid_step(GPUPhase.PREFILL) and len(self._queue) > 0:
                request_view = self._queue.pop(0)
                request_view.state = RequestState.PREFILL
                self._gpu_view.request_views.append(request_view)
            if not self._gpu_view.is_valid_step(GPUPhase.PREFILL):
                self._queue.insert(0, self._gpu_view.request_views.pop())
            return 0, GPUPhase.PREFILL, [request_view.id for request_view in self._gpu_view.request_views], []

        preempted_requests_id = []
        while not self._gpu_view.is_valid_step(GPUPhase.DECODE):
            request_view = self._gpu_view.request_views.pop(0)
            self._gpu_view.remaining_vram_slots += request_view.get_current_vram_usage()
            request_view.state = RequestState.READY
            preempted_requests_id.append(request_view.id)
            self._queue.insert(0, request_view)
        return 0, GPUPhase.DECODE, [], preempted_requests_id

    def update_gpu_view(self, gpu_view):
        self._gpu_view = gpu_view
