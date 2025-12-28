import heapq
from sortedcontainers import SortedList


class RequestHeapItem:
    def __init__(self, req):
        object.__setattr__(self, "req", req)

    def __getattr__(self, name):
        return getattr(self.req, name)

    def __setattr__(self, name, value):
       setattr(self.req, name, value)
    
    def __lt__(self, other):
        return self.predicted_response_len < other.predicted_response_len
    
class SortedListItem:
    def __init__(self, req):
        object.__setattr__(self, "req", req)
        preemption_benefit = max(1, self.predicted_response_len - 2 * self.decode_progress - 2)
        object.__setattr__(self.req, "preemption_benefit", preemption_benefit)

    def __getattr__(self, name):
        return getattr(self.req, name)

    def __setattr__(self, name, value):
       setattr(self.req, name, value)
    
    def __lt__(self, other):
        return self.preemption_benefit < other.preemption_benefit

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



class SRPTDynamicBatchPredictAdjustmentScheduler:
    def __init__(self, initial_gpu_view):
        self._queue = []
        self.update_gpu_view(initial_gpu_view)
        self.pred_adjustment = {}

    def queue(self, request_views):
        for request_view in request_views:
            heapq.heappush(self._queue, RequestHeapItem(request_view))
    
    def update_gpu_view(self, gpu_view):
        self._gpu_view = gpu_view
        self._gpu_preemption_benefit = SortedList([])
        for request_view in self._gpu_view.request_views:
            if request_view.id not in self.pred_adjustment:
                self.pred_adjustment[request_view.id] = ViolationCounter(request_view.predicted_response_len)
            else:
                request_view.predicted_response_len = self.pred_adjustment[request_view.id].check_violation(request_view.decode_progress)
        for request_view in self._gpu_view.request_views:
            self._gpu_preemption_benefit.add(SortedListItem(request_view))

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
            heapq.heappush(self._queue, RequestHeapItem(self._gpu_view.preempt_top()))
            scheduled_request_ids.pop()
            scheduled_requests.pop()

        for scheduled_request_view in scheduled_requests:
           self._gpu_preemption_benefit.add(SortedListItem(scheduled_request_view))
        
        # only preempt if:
        # to_be_scheduled_request.predicted_response_len <
        # to_be_preempted_request.predicted_response_len - 2 * to_be_preempted_request.decode_progress

        # try to schedule all requests in the queue with predicted_response_len less than the largest preemption_benefit
        while len(self._queue) > 0 and len(self._gpu_preemption_benefit) > 0:
            schedule_candidate_heap_item = self._queue[0]
            swap_idx = self._gpu_preemption_benefit.bisect_right(SortedListItem(schedule_candidate_heap_item))
            if swap_idx == len(self._gpu_preemption_benefit):
                break
            
            scheduled_request_heap_item = heapq.heappop(self._queue)
            self._gpu_view.schedule(scheduled_request_heap_item.req)
            scheduled_request_ids.append(scheduled_request_heap_item.id)
            self._gpu_preemption_benefit.add(SortedListItem(scheduled_request_heap_item.req))

        # preempt from the request with the largest preemption benefit until valid
        preempted_request_ids = []
        while not self._gpu_view.is_valid_step_with_predict():
            preempted_request_sorted_list_item = self._gpu_preemption_benefit.pop()
            self._gpu_view.preempt_request(preempted_request_sorted_list_item.id)
            preempted_request_ids.append(preempted_request_sorted_list_item.id)
            heapq.heappush(self._queue, RequestHeapItem(preempted_request_sorted_list_item.req))

        scheduled_set = set(scheduled_request_ids)
        preempted_set = set(preempted_request_ids)
        conflict = scheduled_set & preempted_set
        scheduled_request_ids = [i for i in scheduled_request_ids if i not in conflict]
        preempted_request_ids = [i for i in preempted_request_ids if i not in conflict]

        # will still need to preempt if actual step is invalid
        while not self._gpu_view.is_valid_step():
            request_view = self._gpu_view.preempt_top()
            preempted_request_ids.append(request_view.id)
            heapq.heappush(self._queue, RequestHeapItem(request_view))

        return 0, scheduled_request_ids, preempted_request_ids
