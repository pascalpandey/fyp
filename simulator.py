class Simulator:
    def __init__(self, dataset, gpu, scheduler):
        self._dataset = dataset
        self._gpu = gpu
        self._scheduler = scheduler
        self._t = 0
        self._current_phase = None
    
    def run(self):
        while True:
            if self._current_phase is not None:
                completed_requests = self._gpu.end_previous_step(self._t, self._current_phase)
                self._dataset.increment_completed_requests(len(completed_requests))
                if self._dataset.completed_all_requests():
                    break
        
            self._scheduler.update_gpu_view(self._gpu.get_gpu_view())

            ready_data = self._dataset.get_ready_requests_view(self._t)
            self._scheduler.queue(ready_data)

            # From Alladin paper page 4
            # In default settings like vLLM [12] or split-phase inference, one batch can only contain prefill or decode.
            wait_time, phase, scheduled_request_ids, preempted_request_ids = self._scheduler.decide()
            if wait_time != 0:
                self._t += wait_time
                continue
            
            self._current_phase = phase
            self._gpu.schedule_requests([self._dataset._requests[request_id] for request_id in scheduled_request_ids], self._t)
            self._gpu.preempt_requests([self._dataset._requests[request_id] for request_id in preempted_request_ids], self._t)

            processing_time = self._gpu.start_step(self._t, self._current_phase)

            self._t += processing_time