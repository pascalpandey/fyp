from gpu import GPUPhase
from request import RequestState

class FCFSScheduler:
    def __init__(self, initial_gpu_view):
        self._queue = []
        self._gpu_view = initial_gpu_view
    
    def queue(self, request_views):
        self.queue.extend(request_views)
    
    def decide(self):
        if len(self._queue) == 0:
            return 1, None, None, None
        
        if len(self._gpu_view.request_views) == 0:
            scheduled_requests_id = []
            remaining_vram = self._gpu_view.get_remaining_vram_slots()
            while len(self._queue) > 0 and remaining_vram > 0:
                request = self._queue.pop(0)
                scheduled_requests_id.append(request.id)
                remaining_vram -= request.get_end_step_vram_update()[1]
            return 0, GPUPhase.PREFILL, scheduled_requests_id, []

        preempted_requests_id = []
        while not self._gpu_view.is_valid_step(GPUPhase.DECODE):
            request = self._gpu_view.request_views.pop(0)
            request.state = RequestState.READY
            preempted_requests_id.append(request.id)
            self._queue.insert(0, )
            return 0, GPUPhase.DECODE, [], preempted_requests_id
        return 0, GPUPhase.DECODE, [], []
        

    def update_gpu_view(self, gpu_view):
        self._gpu_view = gpu_view