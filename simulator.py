class Simulator:
    def __init__(self, dataset, gpu, scheduler):
        self._dataset = dataset
        self._gpu = gpu
        self._scheduler = scheduler
        self._t = 0
    
    def run(self):
        while True:
            if self._t > 0:
                completed_requests = self._gpu.end_previous_step(self._t)
                self._dataset.increment_completed_requests(len(completed_requests))
                # if self._dataset._completed_requests_count > 0 and self._dataset._completed_requests_count % 10000 == 0:
                # if completed_requests:
                #     if self._dataset._completed_requests_count % 5000:
                #         print(self._dataset._completed_requests_count)
                if self._dataset.completed_all_requests():
                    break
        
            self._scheduler.update_gpu_view(self._gpu.get_gpu_view())

            ready_data = self._dataset.get_ready_requests_view(self._t)
            self._scheduler.queue(ready_data)

            # From Alladin paper page 4
            # In default settings like vLLM [12] or split-phase inference, one batch can only contain prefill or decode.
            wait_time, scheduled_request_ids, preempted_request_ids = self._scheduler.decide()
            if wait_time != 0:
                self._t += wait_time
                continue
            
            # TODO:
            # maybe preempt first then schedule, so that if a request is in both the preempt and schedule list, it will be 
            # as if the request is restarted, maybe this is faster in certain cases
            self._gpu.preempt_requests([self._dataset._requests[request_id] for request_id in preempted_request_ids], self._t)
            self._gpu.schedule_requests([self._dataset._requests[request_id] for request_id in scheduled_request_ids], self._t)

            processing_time = self._gpu.start_step(self._t)

            self._t += processing_time