class Simulator:
    def __init__(self, dataset, scheduler, resources):
        self._dataset = dataset
        self._scheduler = scheduler
        self._resources = resources
        self._t = 0
    
    def run(self):
        while True:
            completed_requests = self._resources.end_previous_step(self._t)
            self._dataset.complete_requests(completed_requests, self._t)
            if self._dataset.completed_all_requests():
                break
        
            self._scheduler.update_resources_view(self._resources.get_resources_view())

            ready_data = self._dataset.get_ready_requests_view(self._t)
            self._scheduler.queue(ready_data)

            scheduled_request_ids, preempted_request_ids = self._scheduler.decide()
            for id in scheduled_request_ids:
                request = self._dataset._requests[id]
                self._resources.add_request(request)
            for id in preempted_request_ids:
                self._resources.remove_request(id)

            self._resources.start_step(self._t)

            self._t += 1