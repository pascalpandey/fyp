class SJFNonBatchScheduler:
    """
    Shortest Job First (SJF) Non-Batch Scheduler
    
    Similar to FCFSNonBatchScheduler but prioritizes requests with 
    shortest predicted response length instead of arrival time.
    
    Key characteristics:
    - Processes one request at a time (no batching)
    - Uses length predictor to estimate job lengths
    - Always selects the request with shortest predicted completion time
    - Can reduce average waiting time compared to FCFS
    """
    
    def __init__(self, initial_gpu_view):
        self._queue = []
        self._gpu_view = initial_gpu_view
    
    def _predict_job_length(self, request_view):
        """
        Predict the total job length (processing time) for a request.
        Uses the existing predicted_response_len from the dataset loader.
        Total job length = prompt processing + predicted response generation
        """
        # Use the predicted response length from the dataset loader
        prompt_len = getattr(request_view, '_prompt_len', 0)
        predicted_response_len = getattr(request_view, 'predicted_response_len', prompt_len)
        
        # Total job length = response generation
        return predicted_response_len
    
    def queue(self, request_views):
        """Add new requests to the queue"""
        self._queue.extend(request_views)
        # Keep queue sorted by predicted job length (shortest first)
        self._queue.sort(key=lambda req: self._predict_job_length(req))
    
    def decide(self):
        """
        Decide what to do next:
        1. If queue and GPU are empty, wait
        2. If GPU is empty and queue has requests, start shortest job
        3. If GPU is busy, continue current job
        """
        # Wait if both queue and GPU are empty
        if len(self._queue) == 0 and len(self._gpu_view.request_views) == 0:
            return 1, None, None
        
        # If GPU is empty, start the shortest job from queue
        if len(self._gpu_view.request_views) == 0:
            if len(self._queue) > 0:
                # Get the shortest job (first in sorted queue)
                shortest_job = self._queue.pop(0)
                return 0, [shortest_job.id], []
        
        # Continue processing current job
        return 0, [], []
    
    def update_gpu_view(self, gpu_view):
        """Update the scheduler's view of the GPU state"""
        self._gpu_view = gpu_view
        
        # Re-sort queue in case prediction accuracy improves over time
        # (This is optional optimization - could be removed for efficiency)
        if len(self._queue) > 1:
            self._queue.sort(key=lambda req: self._predict_job_length(req))
    
    def get_queue_info(self):
        """
        Debug method to inspect current queue state
        Returns list of (request_id, predicted_length) tuples
        """
        return [(req.id, self._predict_job_length(req)) for req in self._queue]
