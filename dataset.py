import os
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
    
    def _show_average_latency(self):
        average_latency = sum(
            request.response_timestamp - request.request_timestamp
            for request in self._requests.values()
        ) / len(self._requests)
        print(f"Average Latency: {average_latency:.3f} time units")
    
    def _visualize_request_history(self, results_path):
        os.makedirs(results_path, exist_ok=True)
        html_path = os.path.join(results_path, "request_timeline.html")

        rows_html = ""
        row_height = 25
        spacing = 3
        width_scale = 0.25

        for i, req_key in enumerate(self._requests):
            request = self._requests[req_key]
            history_timestamps = sorted(request.history.keys())
            y = i * (row_height + spacing)

            bars = ""
            for j in range(len(history_timestamps) - 1):
                start = history_timestamps[j]
                end = history_timestamps[j + 1]
                state = request.history[start].value.lower()
                color = {
                    "ready": "lightgray",
                    "prefill": "skyblue",
                    "decode": "orange"
                }.get(state, "black")

                left = start * width_scale
                width = max((end - start) * width_scale, 1)
                bars += f'<div class="state-bar" style="left:{left}px; width:{width}px; background-color:{color};"></div>\n'

            rows_html += f'<div class="request-row" style="top:{y}px;">{bars}</div>\n'

        total_height = len(self._requests) * (row_height + spacing)

        html_content = f'''
        <html>
        <head>
            <style>
                body {{ font-family: sans-serif; }}
                .timeline {{
                    position: relative;
                    width: max-content;
                    min-width: 100%;
                    height: {total_height}px;
                    overflow-x: auto;
                    overflow-y: auto;
                    border: 1px solid #ccc;
                    padding: 10px;
                }}
                .request-row {{
                    position: absolute;
                    height: {row_height}px;
                    width: max-content;
                }}
                .state-bar {{
                    position: absolute;
                    height: {row_height}px;
                }}
            </style>
        </head>
        <body>
            {rows_html}
        </body>
        </html>
        '''

        with open(html_path, "w") as f:
            f.write(html_content)

        print(f"Saved timeline visualization to {html_path}")

    
    def show_results(self, results_path):
        self._show_average_latency()
        self._visualize_request_history(results_path)
    

