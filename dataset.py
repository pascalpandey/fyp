import heapq

class Dataset:
    def __init__(self):
        self._requests = {}
        self._pending_heap = []
        self._completed_requests_count = 0

    def add(self, request):
        self._requests[request.id] = request
        heapq.heappush(self._pending_heap, request)

    def get_ready_requests_view(self, timestamp):
        ready_requests = []
        while self._pending_heap and self._pending_heap[0].request_timestamp <= timestamp:
            request = heapq.heappop(self._pending_heap)
            request.step(None)
            ready_requests.append(request.to_request_view())
        return ready_requests

    def completed_all_requests(self):
        return self._completed_requests_count == len(self._requests)

    def increment_completed_requests(self, completed_requests_count):
        self._completed_requests_count += completed_requests_count
    
    def show_results(self):
        average_latency = sum(
            request.response_timestamp - request.request_timestamp
            for request in self._requests.values()
        ) / len(self._requests)
        print(f"Average Latency: {average_latency:.3f} time units")
