from request import RequestState, ProcessStage


class SRTFNonBatchScheduler:
    """
    Shortest Remaining Time First (SRTF) Non-Batch Scheduler
    
    The preemptive version of Shortest Job First (SJF) scheduling.
    
    Key characteristics:
    - Processes one request at a time (no batching)
    - Uses length predictor to estimate remaining processing time
    - Preempts current job if a shorter job arrives in the queue
    - Always selects the request with shortest remaining time
    - Guarantees optimal average waiting time (with perfect predictions)
    
    Preemption behavior:
    - Preempted jobs PRESERVE their decode progress
    - They resume from DECODE stage (skip PREFILL)
    - Only remaining tokens need to be generated
    - Preemption cost is minimal (just context switch overhead)
    
    Algorithm Steps:
    1. At each time unit, check all ready requests (queue + currently running)
    2. Calculate remaining time for each request
    3. Select the request with shortest remaining time
    4. If different from current, preempt current and schedule new one
    5. Continue until all requests complete
    """
    
    def __init__(self, initial_gpu_view):
        self._queue = []
        self._gpu_view = initial_gpu_view
        self._current_running_id = None  # Track which request is currently running
        self.preserve_progress_on_preempt = True  # SRTF preserves decode progress
    
    def _calculate_remaining_time(self, request_view):
        """
        Calculate the remaining processing time for a request.
        
        Remaining time depends on the current stage:
        - Fresh request (READY, no progress): 1 (prefill) + predicted_response_len
        - Preempted request (READY, has progress): predicted_response_len - decode_progress
        - PREFILL stage: 1 (remaining prefill) + predicted_response_len
        - DECODE stage: predicted_response_len - decode_progress
        
        Note: PREFILL takes 1 time unit (see gpu.py line 88-98)
        
        Returns:
            float: Remaining processing time in time units
        """
        predicted_response_len = getattr(request_view, 'predicted_response_len', 0)
        decode_progress = getattr(request_view, '_decode_progress', 0)
        was_preempted = getattr(request_view, '_was_preempted_in_decode', False)
        
        # If request is in DECODE stage, remaining = tokens left to decode
        if request_view.process_stage == ProcessStage.DECODE:
            remaining = predicted_response_len - decode_progress
            return max(remaining, 0)
        
        # If request was preempted mid-decode, it will resume from decode
        # (skip prefill, continue from where it left off)
        if was_preempted:
            remaining = predicted_response_len - decode_progress
            return max(remaining, 0)
        
        # For fresh PREFILL or READY state, add 1 time unit for prefill stage
        return 1 + predicted_response_len
    
    def _calculate_preemption_cost(self, request_view):
        """
        Calculate the overhead cost of preempting a running request.
        
        Since preempted jobs now PRESERVE decode progress and resume from DECODE:
        - No progress is lost
        - Only context switch overhead applies
        
        We use a small threshold to prevent excessive thrashing.
        """
        # Minimal overhead - just context switching cost
        # This prevents preemption when savings are marginal
        CONTEXT_SWITCH_OVERHEAD = 1
        return CONTEXT_SWITCH_OVERHEAD
    
    def queue(self, request_views):
        """Add new requests to the queue"""
        self._queue.extend(request_views)
        # Keep queue sorted by remaining time (shortest first)
        self._queue.sort(key=lambda req: self._calculate_remaining_time(req))
    
    def decide(self):
        """
        Decide what to do next using SRTF algorithm:
        1. If queue and GPU are empty, wait
        2. If GPU is empty and queue has requests, schedule shortest from queue
        3. If GPU is running and queue has shorter job, preempt
        4. Otherwise continue current job
        
        SIMPLIFIED VERSION: Only preempt when a queued job is shorter than running job.
        This avoids infinite preemption loops.
        """
        # Wait if both queue and GPU are empty
        if len(self._queue) == 0 and len(self._gpu_view.request_views) == 0:
            self._current_running_id = None
            return 1, None, None
        
        # Case 1: GPU is empty and we have queued requests
        if len(self._gpu_view.request_views) == 0 and len(self._queue) > 0:
            # Schedule the shortest job from queue
            shortest_job = self._queue.pop(0)  # Queue is already sorted
            self._current_running_id = shortest_job.id
            return 0, [shortest_job.id], []
        
        # Case 2: GPU is running, check if we should preempt
        if len(self._gpu_view.request_views) > 0:
            current_running = self._gpu_view.request_views[0]
            current_remaining = self._calculate_remaining_time(current_running)
            
            # Check if any queued job has shorter remaining time
            if len(self._queue) > 0:
                shortest_queued = self._queue[0]  # First in sorted queue
                queued_remaining = self._calculate_remaining_time(shortest_queued)
                
                # Preemption decision with progress preservation:
                # Since preempted jobs preserve their progress and resume from DECODE:
                #   - current_remaining stays the same after preemption
                #   - Only add context switch overhead to prevent thrashing
                preemption_overhead = self._calculate_preemption_cost(current_running)
                
                # Preempt if the queued job's remaining time + overhead is less than current
                # This means the new job will complete faster, improving avg wait time
                if queued_remaining + preemption_overhead < current_remaining:
                    # Remove shortest from queue
                    shortest_queued = self._queue.pop(0)
                    
                    # Add the preempted request back to the queue
                    # Mark it as preempted so remaining time calculation knows it will resume
                    current_running.state = RequestState.READY
                    current_running._was_preempted_in_decode = True
                    # Keep process_stage and decode_progress intact for accurate remaining time
                    self._queue.append(current_running)
                    self._queue.sort(key=lambda req: self._calculate_remaining_time(req))
                    
                    return 0, [shortest_queued.id], [current_running.id]
            
            # Continue current job
            return 0, [], []
        
        # Default: wait
        return 1, None, None
    
    def update_gpu_view(self, gpu_view):
        """Update the scheduler's view of the GPU state"""
        self._gpu_view = gpu_view
        
        # Re-sort queue to maintain SRTF order
        # Remaining times may change as jobs progress
        if len(self._queue) > 1:
            self._queue.sort(key=lambda req: self._calculate_remaining_time(req))
    
    def get_queue_info(self):
        """
        Debug method to inspect current queue state
        Returns list of (request_id, remaining_time) tuples
        """
        queue_info = [(req.id, self._calculate_remaining_time(req)) for req in self._queue]
        
        # Also show currently running job
        running_info = []
        for req in self._gpu_view.request_views:
            running_info.append((req.id, self._calculate_remaining_time(req), "RUNNING"))
        
        return {
            'queue': queue_info,
            'running': running_info
        }

