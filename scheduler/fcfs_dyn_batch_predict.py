from gpu import GPUPhase
from request import RequestState, ProcessStage


class FCFSDynamicBatchPredictScheduler:
    def __init__(self, initial_gpu_view):
        self._queue = []
        self._gpu_view = initial_gpu_view

    def queue(self, request_views):
        self._queue.extend(request_views)

    def decide(self):
        pass

    def update_gpu_view(self, gpu_view):
        self._gpu_view = gpu_view
