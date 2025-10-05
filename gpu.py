import os
import pandas as pd
import plotly.express as px
from enum import Enum
from request import RequestState, VRAMUpdateType


class GPUPhase(Enum):
    PREFILL = 'prefill'
    DECODE = 'decode'


class VRAM:
    def __init__(self, initial_slots):
        self._remaining_slots = initial_slots
        self._used_slots = 0
        self._usage_history = {}

    def allocate(self, amount, timestamp):
        if amount > self._remaining_slots:
            raise Exception(
                f"VRAM.allocate: VRAM slots exceeded, trying to allocate {amount} slots but only {self._remaining_slots} slots remain")
        self._remaining_slots -= amount
        if self._remaining_slots < 0:
            raise Exception(
                "VRAM.allocate: VRAM remaining slots went negative")
        self._used_slots += amount
        self._usage_history[timestamp] = self._used_slots

    def free(self, amount, timestamp):
        self._remaining_slots += amount
        self._used_slots -= amount
        if self._used_slots < 0:
            raise Exception("VRAM.free: VRAM used slots went negative")
        self._usage_history[timestamp] = self._used_slots

    def get_remaining_slots(self):
        return self._remaining_slots

    def get_used_slots(self):
        return self._used_slots

    def get_usage_history(self):
        return self._usage_history


class GPU:
    def __init__(self, vram_slots):
        self._vram = VRAM(vram_slots)
        self._requests = []

    def schedule_requests(self, scheduled_requests, timestamp):
        for scheduled_request in scheduled_requests:
            scheduled_request.step(timestamp)
            self._requests.append(scheduled_request)

    def preempt_requests(self, preempted_requests, timestamp):
        reclaimed_slots = 0
        for preempted_request in preempted_requests:
            found = False
            for i, scheduled_request in enumerate(self._requests):
                if scheduled_request.id == preempted_request.id:
                    reclaimed_slots += scheduled_request.get_current_vram_usage()
                    preempted_request = self._requests.pop(i)
                    preempted_request.reset_to_ready(timestamp)
                    found = True
                    break
            if not found:
                raise Exception(
                    "GPU.preempt_requests: request {request} not found in scheduled requests")
        self._vram.free(reclaimed_slots, timestamp)

    # From Alladin paper page 4
    # t_prefill = k1 * num_tokens_in_batch + c1
    # t_decode = k2 * num_tokens_in_batch + c2 * size_of_batch + c3
    # for now assume c1 = c2 = c3 = 0, k1 = k2 = 1
    # def start_step(self, phase):
    #     processing_time = 0
    #     for request in self._requests:
    #         if phase == GPUPhase.PREFILL and request.state == RequestState.PREFILL:
    #             processing_time += request.get_start_step_processing_time()

    #         if phase == GPUPhase.DECODE and request.state == RequestState.DECODE:
    #             processing_time += request.get_start_step_processing_time()

    #     return processing_time
    
    # Assume one time unit for both prefill and decode
    def start_step(self, _):
        return 1

    # From Alladin paper page 4
    # kv_size = h * num_tokens + j
    # for now assume h = 1, j = 0
    def end_previous_step(self, timestamp, phase):
        vram_slots_allocated = 0
        vram_slots_freed = 0
        completed_requests = []
        for i, request in enumerate(self._requests):
            if phase == GPUPhase.PREFILL and request.state == RequestState.PREFILL:
                vram_slots_allocated += request.step(timestamp)[1]

            if phase == GPUPhase.DECODE and request.state == RequestState.DECODE:
                update_type, update_slots = request.step(timestamp)
                if update_type == VRAMUpdateType.ALLOCATE:
                    vram_slots_allocated += update_slots
                else:
                    vram_slots_freed += update_slots
                    completed_requests.append(self._requests.pop(i))

        self._vram.allocate(vram_slots_allocated, timestamp)
        self._vram.free(vram_slots_freed, timestamp)
        return completed_requests

    def get_used_vram_slots(self):
        return self._vram.get_used_slots()

    def get_remaining_vram_slots(self):
        return self._vram.get_remaining_slots()

    def get_request_views(self):
        return [request.to_request_view() for request in self._requests]

    def get_gpu_view(self):
        return GPUView(self)

    def visualize_history(self, results_path, scheduler_name):
        filename = f"{scheduler_name}_vram_usage.html"
        os.makedirs(results_path, exist_ok=True)

        df = pd.DataFrame({
            "Time": list(self._vram.get_usage_history().keys()),
            "VRAM": list(self._vram.get_usage_history().values())
        })

        fig = px.line(df, x="Time", y="VRAM", markers=True,
                      title="VRAM Usage Over Time")
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="VRAM Usage",
            hovermode="x unified"
        )

        html_path = os.path.join(results_path, filename)
        fig.write_html(html_path)
        print(f"Saved {scheduler_name} VRAM usage plot to {html_path}")


class GPUView:
    def __init__(self, gpu):
        self.used_vram_slots = gpu.get_used_vram_slots()
        self.remaining_vram_slots = gpu.get_remaining_vram_slots()
        self.request_views = gpu.get_request_views()

    def is_valid_step(self, phase):
        vram_slots_required = 0
        for request in self.request_views:
            if phase == GPUPhase.PREFILL and request.state == RequestState.PREFILL:
                vram_slots_required += request.get_end_step_vram_update()[1]

            if phase == GPUPhase.DECODE and request.state == RequestState.DECODE:
                update_type, update_slots = request.get_end_step_vram_update()
                if update_type == VRAMUpdateType.ALLOCATE:
                    vram_slots_required += update_slots

        if vram_slots_required > self.remaining_vram_slots:
            return False
        return True
