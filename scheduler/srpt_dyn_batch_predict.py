import heapq
import copy
from sortedcontainers import SortedList
from request import ProcessStage


class RequestHeapItem:
    def __init__(self, req):
        object.__setattr__(self, "req", req)

    def __getattr__(self, name):
        return getattr(self.req, name)

    def __setattr__(self, name, value):
       setattr(self.req, name, value)
    
    def __lt__(self, other):
        return self.get_remaining_processing_time() < other.get_remaining_processing_time()
 

class SortedListItem:
    def __init__(self, req):
        object.__setattr__(self, "req", req)
        preemption_benefit = self.predicted_response_len - 2 * self.decode_progress - (2 if self.process_stage == ProcessStage.DECODE else 0)
        object.__setattr__(self.req, "preemption_benefit", preemption_benefit)

    def __getattr__(self, name):
        return getattr(self.req, name)

    def __setattr__(self, name, value):
       setattr(self.req, name, value)
    
    def __lt__(self, other):
        return self.get_remaining_processing_time() < other.get_remaining_processing_time()


class SRPTDynamicBatchPredictScheduler:
    def __init__(self, initial_gpu_view):
        self._queue = []
        self.update_gpu_view(initial_gpu_view)

    def queue(self, request_views):
        for request_view in request_views:
            heapq.heappush(self._queue, RequestHeapItem(request_view))
    
    def update_gpu_view(self, gpu_view):
        self._gpu_view = gpu_view
        self._gpu_remaining_processing_time = SortedList([])
        for request_view in self._gpu_view.request_views:
            self._gpu_remaining_processing_time.add(SortedListItem(request_view))

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
           self._gpu_remaining_processing_time.add(SortedListItem(scheduled_request_view))
        
        # only preempt if:
        # to_be_scheduled_request.predicted_response_len <
        # to_be_preempted_request.predicted_response_len - 2 * to_be_preempted_request.decode_progress - 2

        # try to schedule requests in the queue with remaining processing time less than the largest remaining processing time
        # but also less than the preemption benefit of the preempted request
        preempted_request_ids = []
        while len(self._queue) > 0 and len(self._gpu_remaining_processing_time) > 0:
            schedule_candidate_heap_item = self._queue[0]
            swap_idx = None
            for i in range(len(self._gpu_remaining_processing_time) - 1, -1, -1):
                if schedule_candidate_heap_item.get_remaining_processing_time() >= self._gpu_remaining_processing_time[i].get_remaining_processing_time():
                    break
                if schedule_candidate_heap_item.get_remaining_processing_time() < self._gpu_remaining_processing_time[i].preemption_benefit:
                    swap_idx = i
                    break
            
            if swap_idx is None:
                break

            preempt_candidate = self._gpu_remaining_processing_time[swap_idx].req
            if not self._gpu_view.try_swap_with_predict(preempt_candidate.id, copy.deepcopy(schedule_candidate_heap_item.req)):
                break

            preempted_request_sorted_list_item = self._gpu_remaining_processing_time.pop(swap_idx)
            self._gpu_view.preempt_request(preempted_request_sorted_list_item.id)
            preempted_request_ids.append(preempted_request_sorted_list_item.id)
            heapq.heappush(self._queue, RequestHeapItem(preempted_request_sorted_list_item.req))

            scheduled_request_heap_item = heapq.heappop(self._queue)
            self._gpu_view.schedule(scheduled_request_heap_item.req)
            scheduled_request_ids.append(scheduled_request_heap_item.id)
            self._gpu_remaining_processing_time.add(SortedListItem(scheduled_request_heap_item.req))

        # will still need to preempt if actual step is invalid
        while not self._gpu_view.is_valid_step():
            preempted_request_sorted_list_item = self._gpu_remaining_processing_time.pop()
            self._gpu_view.preempt_request(preempted_request_sorted_list_item.id)
            preempted_request_ids.append(preempted_request_sorted_list_item.id)
            heapq.heappush(self._queue, RequestHeapItem(preempted_request_sorted_list_item.req))

        return 0, scheduled_request_ids, preempted_request_ids
