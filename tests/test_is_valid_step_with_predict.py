import pytest

from gpu import GPU
from request import (
    Request,
    RequestState,
    ProcessStage,
)


def make_scheduled_request(prompt_len, predicted_response_len, process_stage, decode_progress):
    r = Request(
        visualization_name="test",
        prompt_len=prompt_len,
        response_len=predicted_response_len,
        request_timestamp=0,
        predicted_response_len=predicted_response_len,
    )
    r.state = RequestState.SCHEDULED
    r.process_stage = process_stage
    r._decode_progress = decode_progress
    return r.to_request_view()


@pytest.mark.parametrize(
    "total_vram, used, requests, expected",
    [
        (15, 7, [(1, 9, ProcessStage.DECODE, 6), (1, 9, ProcessStage.PREFILL, 0)], True),
        (15, 6, [(1, 9, ProcessStage.DECODE, 5), (1, 9, ProcessStage.PREFILL, 0)], True),
        (15, 5, [(1, 9, ProcessStage.DECODE, 4), (1, 9, ProcessStage.PREFILL, 0)], True),
        (15, 4, [(1, 9, ProcessStage.DECODE, 3), (1, 9, ProcessStage.PREFILL, 0)], False),
        (15, 3, [(1, 9, ProcessStage.DECODE, 2), (1, 9, ProcessStage.PREFILL, 0)], False),
        (15, 2, [(1, 9, ProcessStage.DECODE, 1), (1, 9, ProcessStage.PREFILL, 0)], False),
    ]
)
def test_is_valid_step_with_predict(total_vram, used, requests, expected):
    gpu = GPU(vram_slots=total_vram)
    gpu_view = gpu.get_gpu_view()

    gpu_view.used_vram_slots = used
    gpu_view.remaining_vram_slots = total_vram - used

    gpu_view.request_views = [
        make_scheduled_request(p, r, s, d)
        for (p, r, s, d) in requests
    ]

    assert gpu_view.is_valid_step_with_predict() is expected
