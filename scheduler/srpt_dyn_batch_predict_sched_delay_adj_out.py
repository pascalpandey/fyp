import heapq
import copy
import numpy as np
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

    def __getattr__(self, name):
        return getattr(self.req, name)

    def __setattr__(self, name, value):
       setattr(self.req, name, value)
    
    def __lt__(self, other):
        return self.get_remaining_processing_time() < other.get_remaining_processing_time()


class SRPTDynamicBatchPredictScheduleDelayAdjOutScheduler:
    def __init__(self, initial_gpu_view):
        self._queue = []
        self._previous_requests = {}
        self._actual = []
        self.update_gpu_view(initial_gpu_view)

    def queue(self, request_views):
        for request_view in request_views:
            heapq.heappush(self._queue, RequestHeapItem(request_view))
    
    def update_gpu_view(self, gpu_view):
        self._gpu_view = gpu_view
        scheduled_request_ids = set([request_view.id for request_view in self._gpu_view.request_views])
        for req_id in self._previous_requests:
            if req_id not in scheduled_request_ids:
                decode_progress, predicted_response_len = self._previous_requests[req_id]
                actual_len = decode_progress + 1
                self._actual.append(actual_len)
        self._previous_requests = {}
        self._gpu_remaining_processing_time = SortedList([])
        for request_view in self._gpu_view.request_views:
            self._previous_requests[request_view.id] = (request_view.decode_progress, request_view.predicted_response_len)
            if self._actual:
                q1, q3 = np.percentile(self._actual, [25, 75])
                iqr = q3 - q1
                if request_view.predicted_response_len < q1 - iqr or request_view.predicted_response_len > q3 + iqr:
                    request_view.predicted_response_len = request_view.decode_progress + 1
            request_view.predicted_response_len = max(request_view.predicted_response_len, request_view.decode_progress + 1)
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
        
        preempted_request_ids = []
        while len(self._queue) > 0 and len(self._gpu_remaining_processing_time) > 0:
            if not self._gpu_view.is_valid_step_with_predict():
                break

            schedule_candidate_heap_item = self._queue[0]

            schedule_delay = self._gpu_view.get_schedule_delay(schedule_candidate_heap_item)
            if schedule_delay is None:
                break
            
            swap_idx = None
            for i in range(len(self._gpu_remaining_processing_time) - 1, -1, -1):
                preempt_candidate = self._gpu_remaining_processing_time[i]
                if preempt_candidate.get_remaining_processing_time() <= schedule_candidate_heap_item.get_remaining_processing_time():
                    break
                if not self._gpu_view.try_swap_with_predict(preempt_candidate.id, copy.deepcopy(schedule_candidate_heap_item.req)):
                    continue
                if len(self._queue) > 1 and preempt_candidate.get_total_processing_time() > self._queue[1].get_remaining_processing_time():
                    incoming_schedule_delay = self._gpu_view.try_swap_get_next_in_queue_schedule_delay(preempt_candidate.id, copy.deepcopy(schedule_candidate_heap_item.req), copy.deepcopy(self._queue[1].req))
                else:
                    incoming_schedule_delay = self._gpu_view.try_swap_get_preempted_schedule_delay(preempt_candidate.id, copy.deepcopy(schedule_candidate_heap_item.req))
                if schedule_delay - incoming_schedule_delay > preempt_candidate.get_current_scheduled_age():
                    swap_idx = i
                    break
            
            if swap_idx is None:
                break

            preempted_request_sorted_list_item = self._gpu_remaining_processing_time.pop(swap_idx)
            self._gpu_view.preempt_request(preempted_request_sorted_list_item.id)
            preempted_request_ids.append(preempted_request_sorted_list_item.id)
            heapq.heappush(self._queue, RequestHeapItem(preempted_request_sorted_list_item.req))

            scheduled_request_heap_item = heapq.heappop(self._queue)
            self._gpu_view.schedule(scheduled_request_heap_item.req)
            scheduled_request_ids.append(scheduled_request_heap_item.id)
            self._gpu_remaining_processing_time.add(SortedListItem(scheduled_request_heap_item.req))
        
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

        while not self._gpu_view.is_valid_step_with_predict():
            preempted_request_sorted_list_item = self._gpu_remaining_processing_time.pop()
            self._gpu_view.preempt_request(preempted_request_sorted_list_item.id)
            preempted_request_ids.append(preempted_request_sorted_list_item.id)
            heapq.heappush(self._queue, RequestHeapItem(preempted_request_sorted_list_item.req))
        
        for req_id in preempted_request_ids:
            del self._previous_requests[req_id]

        return 0, scheduled_request_ids, preempted_request_ids
