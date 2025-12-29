from request import RequestState, ProcessStage


class SJFDynamicBatchScheduler:
    """
    Shortest Job First (SJF) Dynamic Batch Scheduler
    
    Key characteristics:
    - Processes multiple requests simultaneously (batching)
    - Uses length predictor to estimate job lengths
    - Prioritizes shortest jobs first when packing the batch
    - Continuously adds new requests even while GPU is processing (dynamic)
    - Should significantly outperform static SJF batch
    """
    
    def __init__(self, initial_gpu_view):
        self._queue = []
        self._gpu_view = initial_gpu_view
    
    def _predict_job_length(self, request_view):
        """
        Predict the total job length (processing time) for a request.
        Uses the existing predicted_response_len from the dataset loader.
        """
        # Use the predicted response length from the dataset loader
        prompt_len = getattr(request_view, '_prompt_len', 0)
        predicted_response_len = getattr(request_view, 'predicted_response_len', prompt_len)
        
        # Total job length = response generation
        return predicted_response_len
    
    def queue(self, request_views):
        """Add new requests to the queue and sort by predicted job length"""
        self._queue.extend(request_views)
        # Keep queue sorted by predicted job length (shortest first)
        self._queue.sort(key=lambda req: self._predict_job_length(req))
    
    def decide(self):
        """
        Decide what to do next:
        1. If queue and GPU are empty, wait
        2. Always try to schedule shortest jobs, even if GPU is not empty (dynamic)
        3. If GPU is overloaded, preempt jobs until valid
        """
        # wait if queue and GPU is empty
        if len(self._queue) == 0 and len(self._gpu_view.request_views) == 0:
            return 1, None, None

        # always try to schedule new requests, even if the GPU is not empty (DYNAMIC)
        scheduled_request_ids = []
        while self._gpu_view.is_valid_step() and len(self._queue) > 0:
            # Get the shortest job (first in sorted queue)
            request_view = self._queue.pop(0)
            request_view.state = RequestState.SCHEDULED
            request_view.process_stage = ProcessStage.PREFILL
            self._gpu_view.request_views.append(request_view)
            scheduled_request_ids.append(request_view.id)
        if len(scheduled_request_ids) > 0 and not self._gpu_view.is_valid_step():
            # If we overpacked, remove the last added request
            self._queue.insert(0, self._gpu_view.request_views.pop())
            scheduled_request_ids.pop()
        if scheduled_request_ids:
            return 0, scheduled_request_ids, []

        # Handle preemption if GPU is overloaded
        preempted_requests_id = []
        while not self._gpu_view.is_valid_step():
            request_view = self._gpu_view.request_views.pop()
            self._gpu_view.remaining_vram_slots += request_view.get_current_vram_usage()
            request_view.state = RequestState.READY
            request_view.process_stage = None
            preempted_requests_id.append(request_view.id)
            self._queue.insert(0, request_view)
        
        # Re-sort queue after preemption to maintain SJF order
        if len(self._queue) > 1:
            self._queue.sort(key=lambda req: self._predict_job_length(req))
        
        return 0, [], preempted_requests_id

    def update_gpu_view(self, gpu_view):
        """Update the scheduler's view of the GPU state"""
        self._gpu_view = gpu_view
        
        # Re-sort queue in case new requests arrived or prediction changes
        if len(self._queue) > 1:
            self._queue.sort(key=lambda req: self._predict_job_length(req))
    
    def get_queue_info(self):
        """
        Debug method to inspect current queue state
        Returns list of (request_id, predicted_length) tuples
        """
        return [(req.id, self._predict_job_length(req)) for req in self._queue]

