import os
import heapq
import pandas as pd
import plotly.figure_factory as ff


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
            request.step() # PENDING to READY
            ready_requests.append(request.to_request_view())
        return ready_requests

    def completed_all_requests(self):
        return self._completed_requests_count == len(self._requests)

    def increment_completed_requests(self, completed_requests_count):
        self._completed_requests_count += completed_requests_count

    def _show_average_latency(self, scheduler_name):
        average_latency = sum(
            request.response_timestamp - request.request_timestamp
            for request in self._requests.values()
        ) / len(self._requests)
        print(f"Average Latency {scheduler_name}: {average_latency:.3f} time units")

    def _visualize_request_history(self, results_path, scheduler_name):
        os.makedirs(results_path, exist_ok=True)
        html_path = os.path.join(results_path, f"{scheduler_name}_request_timeline.html")

        rows = []
        for i, req_key in enumerate(self._requests):
            request = self._requests[req_key]
            history_timestamps = sorted(request.process_history.keys())

            for j in range(len(history_timestamps) - 1):
                start = history_timestamps[j]
                end = history_timestamps[j + 1]
                state = request.process_history[start].value

                rows.append({
                    "Task": f"Request {i}",
                    "Start": start,
                    "Finish": end,
                    "State": state,
                })

        df = pd.DataFrame(rows)

        fig = ff.create_gantt(
            df,
            colors={
                "ready": "rgb(211,211,211)",
                "prefill": "rgb(135,206,235)",
                "decode": "rgb(255,165,0)",
                "idle": "rgb(169,169,169)"
            },
            index_col='State',
            show_colorbar=True,
            bar_width=0.4,
            showgrid_x=True,
            showgrid_y=True,
            group_tasks=True,
            title="Request Timeline"
        )

        fig.update_layout(
            xaxis_type="linear",
            height=15*df["Task"].nunique()
        )

        fig.show()
        fig.write_html(html_path)

        print(f"Saved {scheduler_name} timeline visualization to {html_path}")

    def show_results(self, results_path, scheduler_name):
        self._show_average_latency(scheduler_name)
        self._visualize_request_history(results_path, scheduler_name)
