import heapq
import copy
from sortedcontainers import SortedList
from request import ProcessStage


class RequestPriority:
    def __init__(self, req):
        object.__setattr__(self, "req", req)
        priority = self.prompt_len + self.predicted_response_len - 2 * self.decode_progress - (2 if self.process_stage == ProcessStage.DECODE else 0)
        object.__setattr__(self.req, "priority", priority)

    def __getattr__(self, name):
        return getattr(self.req, name)

    def __setattr__(self, name, value):
       setattr(self.req, name, value)
    
    def __lt__(self, other):
        return self.priority < other.priority


class ViolationCounter:
    def __init__(self, predicted_response_len):
        self.violation_count = 0
        self.initial_predicted_response_len = predicted_response_len
        self.adjusted_predicted_response_len = predicted_response_len
    
    def check_violation(self, current_decode_progress):
        if current_decode_progress == self.adjusted_predicted_response_len:
            self.violation_count += 1
            self.adjusted_predicted_response_len += (0.25 ** self.violation_count) * self.initial_predicted_response_len
        return self.adjusted_predicted_response_len


class BicriteriaDynamicBatchPredictPreemptiveAdjOverScheduler:
    def __init__(self, initial_gpu_view):
        self._queue = []
        self.update_gpu_view(initial_gpu_view)
        self.pred_adjustment = {}

    def queue(self, request_views):
        for request_view in request_views:
            heapq.heappush(self._queue, RequestPriority(request_view))
    
    def update_gpu_view(self, gpu_view):
        self._gpu_view = gpu_view
        self._gpu_request_priority = SortedList([])
        for request_view in self._gpu_view.request_views:
            if request_view.id not in self.pred_adjustment:
                self.pred_adjustment[request_view.id] = ViolationCounter(request_view.predicted_response_len)
            else:
                request_view.predicted_response_len = self.pred_adjustment[request_view.id].check_violation(request_view.decode_progress)
        for request_view in self._gpu_view.request_views:
            self._gpu_request_priority.add(RequestPriority(request_view))

    def decide(self):
        # wait if queue and GPU is empty
        if len(self._queue) == 0 and len(self._gpu_view.request_views) == 0:
            return 1, None, None

        # always try to schedule new requests, even if the GPU is not empty
        scheduled_request_ids = []
        scheduled_requests = []
        while self._gpu_view.is_valid_step_with_predict() and len(self._queue) > 0:
            request_heap_item = heapq.heappop(self._queue)
            self._gpu_view.schedule(request_heap_item.req)
            scheduled_request_ids.append(request_heap_item.id)
            scheduled_requests.append(request_heap_item.req)
        if len(scheduled_request_ids) > 0 and not self._gpu_view.is_valid_step_with_predict():
            heapq.heappush(self._queue, RequestPriority(self._gpu_view.preempt_top()))
            scheduled_request_ids.pop()
            scheduled_requests.pop()

        for scheduled_request_view in scheduled_requests:
           self._gpu_request_priority.add(RequestPriority(scheduled_request_view))
        
        # only preempt if to_be_scheduled.request_priority < to_be_preempted.request_priority
        preempted_request_ids = []
        while len(self._queue) > 0 and len(self._gpu_request_priority) > 0:
            schedule_candidate_heap_item = self._queue[0]
            if schedule_candidate_heap_item.priority >= self._gpu_request_priority[-1].priority:
                break
            
            preempt_candidate = self._gpu_request_priority[-1].req
            if not self._gpu_view.try_swap_with_predict(preempt_candidate.id, copy.deepcopy(schedule_candidate_heap_item.req)):
                break

            preempted_request_sorted_list_item = self._gpu_request_priority.pop()
            self._gpu_view.preempt_request(preempted_request_sorted_list_item.id)
            preempted_request_ids.append(preempted_request_sorted_list_item.id)
            heapq.heappush(self._queue, RequestPriority(preempted_request_sorted_list_item.req))
            
            scheduled_request_heap_item = heapq.heappop(self._queue)
            self._gpu_view.schedule(scheduled_request_heap_item.req)
            scheduled_request_ids.append(scheduled_request_heap_item.id)
            self._gpu_request_priority.add(RequestPriority(scheduled_request_heap_item.req))
        
        scheduled_requests = []
        while self._gpu_view.is_valid_step_with_predict() and len(self._queue) > 0:
            request_heap_item = heapq.heappop(self._queue)
            self._gpu_view.schedule(request_heap_item.req)
            scheduled_request_ids.append(request_heap_item.id)
            scheduled_requests.append(request_heap_item.req)
        if len(scheduled_request_ids) > 0 and not self._gpu_view.is_valid_step_with_predict():
            heapq.heappush(self._queue, RequestPriority(self._gpu_view.preempt_top()))
            scheduled_request_ids.pop()
            scheduled_requests.pop()

        for scheduled_request_view in scheduled_requests:
           self._gpu_request_priority.add(RequestPriority(scheduled_request_view))

        # will still need to preempt if actual step is invalid
        while not self._gpu_view.is_valid_step_with_predict():
            preempted_request_sorted_list_item = self._gpu_request_priority.pop()
            self._gpu_view.preempt_request(preempted_request_sorted_list_item.id)
            preempted_request_ids.append(preempted_request_sorted_list_item.id)
            heapq.heappush(self._queue, RequestPriority(preempted_request_sorted_list_item.req))
        
        return 0, scheduled_request_ids, preempted_request_ids
