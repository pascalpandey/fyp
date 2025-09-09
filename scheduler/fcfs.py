class FCFSScheduler:
    def __init__(self):
        self._queue = []
        self._resources_view = None
    
    def queue(self, request_views):
        self.queue.extend(request_views)

    def schedule_requests(self):
        scheduled = self.queue
        self.queue = []
        return scheduled
    
    def preempt_requests(self):
        preempted = []
        return preempted
    
    def complete_requests(self, completed_requests):
        for request in completed_requests:
            request.state = 'completed'