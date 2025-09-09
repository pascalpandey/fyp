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
                self._dataset.complete_requests(completed_requests, self._t)
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
            for id in scheduled_request_ids:
                request = self._dataset._requests[id]
                self._gpu.add_request(request)
            for id in preempted_request_ids:
                request = self._dataset._requests[id]
                self._gpu.remove_request(request)

            processing_time = self._gpu.start_step(self._current_phase)

            self._t += processing_time