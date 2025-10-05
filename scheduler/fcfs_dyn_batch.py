from gpu import GPUPhase
from request import RequestState, ProcessStage


class FCFSDynamicBatchScheduler:
    def __init__(self, initial_gpu_view):
        self._queue = []
        self._gpu_view = initial_gpu_view

    def queue(self, request_views):
        self._queue.extend(request_views)

    def decide(self):
         # wait if queue and GPU is empty
        if len(self._queue) == 0 and len(self._gpu_view.request_views) == 0:
            return 1, None, None, None

        # always try to schedule new requests, even if the GPU is not empty
        scheduled_request_ids = []
        while self._gpu_view.is_valid_step(GPUPhase.PREFILL) and len(self._queue) > 0:
            request_view = self._queue.pop(0)
            request_view.state = RequestState.SCHEDULED
            request_view.process_stage = ProcessStage.PREFILL
            self._gpu_view.request_views.append(request_view)
            scheduled_request_ids.append(request_view.id)
        if not self._gpu_view.is_valid_step(GPUPhase.PREFILL):
            self._queue.insert(0, self._gpu_view.request_views.pop())
            scheduled_request_ids.pop()
        if scheduled_request_ids:
            return 0, GPUPhase.PREFILL, scheduled_request_ids, []

        preempted_requests_id = []
        while not self._gpu_view.is_valid_step(GPUPhase.DECODE):
            request_view = self._gpu_view.request_views.pop()
            self._gpu_view.remaining_vram_slots += request_view.get_current_vram_usage()
            request_view.state = RequestState.READY
            request_view.process_stage = None
            preempted_requests_id.append(request_view.id)
            self._queue.insert(0, request_view)
        return 0, GPUPhase.DECODE, [], preempted_requests_id

    def update_gpu_view(self, gpu_view):
        self._gpu_view = gpu_view
