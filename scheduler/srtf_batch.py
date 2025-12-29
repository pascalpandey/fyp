from request import RequestState, ProcessStage


class SRTFBatchScheduler:
    """
    Shortest Remaining Time First (SRTF) Batch Scheduler
    
    The preemptive version of SJF Batch scheduling.
    
    Key characteristics:
    - Processes multiple requests simultaneously (batching)
    - Uses remaining time (not total job length) for scheduling decisions
    - Preempts running jobs when shorter jobs arrive in queue
    - Prioritizes jobs with shortest remaining time when packing the batch
    - Can reduce average waiting time compared to SJF batch
    
    Preemption behavior:
    - Preempted jobs PRESERVE their decode progress
    - They resume from DECODE stage (skip PREFILL)
    - Only remaining tokens need to be generated
    - Preemption cost is minimal (just context switch overhead)
    """
    
    def __init__(self, initial_gpu_view):
        self._queue = []
        self._gpu_view = initial_gpu_view
        self.preserve_progress_on_preempt = True  # SRTF preserves decode progress
    
    def _calculate_remaining_time(self, request_view):
        """
        Calculate the remaining processing time for a request.
        
        Remaining time depends on the current stage:
        - Fresh request (READY, no progress): 1 (prefill) + predicted_response_len
        - Preempted request (READY, has progress): predicted_response_len - decode_progress
        - PREFILL stage: 1 (remaining prefill) + predicted_response_len
        - DECODE stage: predicted_response_len - decode_progress
        
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
        if was_preempted:
            remaining = predicted_response_len - decode_progress
            return max(remaining, 0)
        
        # For fresh PREFILL or READY state, add 1 time unit for prefill stage
        return 1 + predicted_response_len
    
    def _calculate_preemption_cost(self):
        """
        Calculate the overhead cost of preempting a running request.
        
        Since preempted jobs PRESERVE decode progress and resume from DECODE:
        - No progress is lost
        - Only context switch overhead applies
        """
        CONTEXT_SWITCH_OVERHEAD = 1
        return CONTEXT_SWITCH_OVERHEAD
    
    def queue(self, request_views):
        """Add new requests to the queue and sort by remaining time"""
        self._queue.extend(request_views)
        # Keep queue sorted by remaining time (shortest first)
        self._queue.sort(key=lambda req: self._calculate_remaining_time(req))
    
    def decide(self):
        """
        Decide what to do next using SRTF algorithm:
        1. If queue and GPU are empty, wait
        2. If GPU is empty, pack as many shortest-remaining-time jobs as possible
        3. Check if any queued job should preempt a running job (SRTF logic)
        4. If GPU is overloaded, preempt longest-remaining-time jobs until valid
        """
        # Wait if queue and GPU are empty
        if len(self._queue) == 0 and len(self._gpu_view.request_views) == 0:
            return 1, None, None

        # If GPU is empty, pack as many requests as possible
        if len(self._gpu_view.request_views) == 0:
            while self._gpu_view.is_valid_step_with_predict() and len(self._queue) > 0:
                # Get the job with shortest remaining time (first in sorted queue)
                request_view = self._queue[0]
                
                # Check if this is a previously preempted request
                # If so, it needs extra VRAM for resume (prompt_len + decode_progress)
                was_preempted = getattr(request_view, '_was_preempted_in_decode', False)
                if was_preempted:
                    resume_vram_needed = request_view._prompt_len + request_view._decode_progress
                    if resume_vram_needed > self._gpu_view.remaining_vram_slots:
                        # Not enough VRAM to resume this request, skip for now
                        break
                    # Account for the resume VRAM in our view
                    self._gpu_view.remaining_vram_slots -= resume_vram_needed
                
                request_view = self._queue.pop(0)
                request_view.state = RequestState.SCHEDULED
                # If previously preempted, it will resume from DECODE
                if was_preempted:
                    request_view.process_stage = ProcessStage.DECODE
                else:
                    request_view.process_stage = ProcessStage.PREFILL
                self._gpu_view.request_views.append(request_view)
                
            # If predictive check failed after last append, remove the last added request
            if not self._gpu_view.is_valid_step_with_predict():
                self._queue.insert(0, self._gpu_view.request_views.pop())
            return 0, [request_view.id for request_view in self._gpu_view.request_views], []

        # SRTF Logic: Check if any queued job should preempt a running job
        # Note: We only preempt in this turn, and schedule the replacement in next turn
        # This avoids VRAM accounting issues (simulator processes schedules before preemptions)
        if len(self._queue) > 0 and len(self._gpu_view.request_views) > 0:
            shortest_queued = self._queue[0]
            queued_remaining = self._calculate_remaining_time(shortest_queued)
            preemption_overhead = self._calculate_preemption_cost()
            
            # Find the running job with longest remaining time
            longest_running = max(
                self._gpu_view.request_views,
                key=lambda req: self._calculate_remaining_time(req)
            )
            longest_remaining = self._calculate_remaining_time(longest_running)
            
            # Preempt if queued job has significantly shorter remaining time
            if queued_remaining + preemption_overhead < longest_remaining:
                # Only preempt in this turn - the shorter job will be scheduled next turn
                longest_running.state = RequestState.READY
                longest_running._was_preempted_in_decode = True
                
                self._gpu_view.request_views.remove(longest_running)
                self._gpu_view.remaining_vram_slots += longest_running.get_current_vram_usage()
                
                # Add preempted job back to queue (it will be re-sorted)
                self._queue.append(longest_running)
                self._queue.sort(key=lambda req: self._calculate_remaining_time(req))
                
                return 0, [], [longest_running.id]

        # Handle preemption if GPU is overloaded (VRAM constraint)
        preempted_requests_id = []
        while not self._gpu_view.is_valid_step():
            # Preempt the job with longest remaining time
            longest_job = max(
                self._gpu_view.request_views,
                key=lambda req: self._calculate_remaining_time(req)
            )
            self._gpu_view.request_views.remove(longest_job)
            self._gpu_view.remaining_vram_slots += longest_job.get_current_vram_usage()
            longest_job.state = RequestState.READY
            longest_job._was_preempted_in_decode = True
            preempted_requests_id.append(longest_job.id)
            self._queue.insert(0, longest_job)
        
        # Re-sort queue after preemption to maintain SRTF order
        if len(self._queue) > 1:
            self._queue.sort(key=lambda req: self._calculate_remaining_time(req))
        
        return 0, [], preempted_requests_id

    def update_gpu_view(self, gpu_view):
        """Update the scheduler's view of the GPU state"""
        self._gpu_view = gpu_view
        
        # Re-sort queue to maintain SRTF order
        if len(self._queue) > 1:
            self._queue.sort(key=lambda req: self._calculate_remaining_time(req))
    
    def get_queue_info(self):
        """
        Debug method to inspect current queue state
        Returns list of (request_id, remaining_time) tuples
        """
        queue_info = [(req.id, self._calculate_remaining_time(req)) for req in self._queue]
        running_info = [(req.id, self._calculate_remaining_time(req), "RUNNING") 
                       for req in self._gpu_view.request_views]
        return {
            'queue': queue_info,
            'running': running_info
        }

