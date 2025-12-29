# SRTF (Shortest Remaining Time First) Implementation

## Overview

This document explains the implementation of the **Shortest Remaining Time First (SRTF)** scheduling algorithm, which is the **preemptive version** of Shortest Job First (SJF).

## What is SRTF?

**SRTF** is a CPU scheduling algorithm that:
- Selects the process with the **shortest remaining time** to complete
- **Preempts** the currently running process if a new process arrives with less remaining time
- Guarantees **optimal average waiting time** (with perfect time predictions)
- Is a **preemptive** algorithm (unlike SJF which can be non-preemptive)

## Algorithm Steps

### Step 1: Input
- Processes arrive with:
  - **Arrival time**: When the request enters the system
  - **Burst time** (predicted): Estimated processing time from `sjf_length_predictor.py`

### Step 2: Initialize
- **Remaining time** = predicted response length for each request
- **Current time** = 0
- **Queue** = empty list of ready requests

### Step 3: At Each Time Unit
- Add all newly arrived requests to the ready queue
- Update remaining times for all requests (queue + running)

### Step 4: Select Process
- Calculate remaining time for ALL requests (queued + currently running)
- Select the request with the **shortest remaining time**
- If this is different from the currently running request → **PREEMPT**

### Step 5: Execute
- Execute the selected process for 1 time unit
- Reduce its remaining time by 1
- Increment current time

### Step 6: Repeat
- Repeat Steps 3-5 until all processes complete

## Implementation Details

### File: `scheduler/srtf_nonbatch.py`

#### Key Methods

##### 1. `_calculate_remaining_time(request_view)`
Calculates how much time remains for a request:

```python
def _calculate_remaining_time(self, request_view):
    predicted_response_len = getattr(request_view, 'predicted_response_len', 0)
    
    # If in DECODE stage, subtract progress
    if request_view.process_stage == ProcessStage.DECODE:
        decode_progress = getattr(request_view, '_decode_progress', 0)
        remaining = predicted_response_len - decode_progress
        return max(remaining, 0)
    
    # For PREFILL or READY, full job length remains
    return predicted_response_len
```

**Remaining Time Calculation:**
- **READY (queued)**: `predicted_response_len` (full job remains)
- **PREFILL stage**: `predicted_response_len` (about to start)
- **DECODE stage**: `predicted_response_len - decode_progress` (partially done)

##### 2. `queue(request_views)`
Adds new requests and sorts by remaining time:

```python
def queue(self, request_views):
    self._queue.extend(request_views)
    # Keep sorted by remaining time (shortest first)
    self._queue.sort(key=lambda req: self._calculate_remaining_time(req))
```

##### 3. `decide()`
The core SRTF decision logic:

```python
def decide(self):
    # 1. Collect all candidates (queue + running)
    candidates = []
    for req in self._queue:
        candidates.append({
            'request': req,
            'remaining_time': self._calculate_remaining_time(req),
            'location': 'queue'
        })
    for req in self._gpu_view.request_views:
        candidates.append({
            'request': req,
            'remaining_time': self._calculate_remaining_time(req),
            'location': 'running'
        })
    
    # 2. Select shortest remaining time
    shortest = min(candidates, key=lambda x: x['remaining_time'])
    
    # 3. Check if preemption needed
    current_running = self._gpu_view.request_views[0] if self._gpu_view.request_views else None
    
    if current_running and shortest['request'].id != current_running.id:
        # PREEMPT: shorter job found
        return 0, [shortest['request'].id], [current_running.id]
    
    # 4. Continue current or schedule new
    # ...
```

## How to Run

### Option 1: Run Full Benchmark (All Schedulers)

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run all schedulers including SRTF
python main.py
```

This runs:
- FCFS (non-batch, batch, dynamic)
- SJF (non-batch, batch, dynamic)
- **SRTF (non-batch)** ← New!

### Option 2: Test SRTF Only

```bash
# Run the SRTF test script
python test_srtf.py
```

This script:
- Loads 20 requests from the dataset
- Runs SRTF scheduler
- Runs SJF scheduler for comparison
- Shows latency comparison
- Generates visualization files

### Option 3: Run Only SRTF in Main

Edit `main.py` line 18-27:

```python
SCHEDULER_NAMES = [
    'srtf_nonbatch',  # Only SRTF
]
```

Then run:
```bash
python main.py
```

## Latency Calculation

Latency is automatically calculated by the `Dataset` class:

```python
def show_average_latency(self, scheduler_name):
    average_latency = sum(
        request.response_timestamp - request.request_timestamp
        for request in self._requests.values()
    ) / len(self._requests)
    print(f"Average Latency {scheduler_name}: {average_latency:.3f} time units")
```

**Latency** = `response_timestamp - request_timestamp`

For each request:
- `request_timestamp`: When the request arrived in the system
- `response_timestamp`: When the request completed
- Latency measures total time from arrival to completion

## Expected Results

SRTF should achieve:
- ✅ **Lower average latency** than SJF (non-preemptive)
- ✅ **Optimal average waiting time** (with perfect predictions)
- ✅ **More preemptions** (visible in timeline visualization)
- ✅ **Better response for short jobs** that arrive late

### Example Output

```
Average Latency SRTF:  45.234 time units
Average Latency SJF:   52.167 time units
Improvement:           13.29%
```

## Visualizations

After running, check the `./results/` folder for:

1. **Request Timeline** (`*_request_timeline.html`)
   - Shows when each request is: ready (gray), prefill (blue), decode (orange)
   - SRTF will show more state transitions (preemptions)

2. **VRAM Usage** (`*_vram_usage.html`)
   - Shows GPU memory usage over time
   - For non-batch, should be similar to SJF

## Key Differences: SRTF vs SJF

| Feature | SJF (Non-Batch) | SRTF (Non-Batch) |
|---------|-----------------|------------------|
| **Preemption** | ❌ No | ✅ Yes |
| **Selection** | Shortest total time | Shortest *remaining* time |
| **When decides** | Only when GPU idle | Every time unit |
| **Optimality** | Good | Optimal* |
| **Context switches** | Low | Higher |
| **Starvation risk** | Medium | Higher (long jobs) |

*With perfect time predictions

## Integration with Length Predictor

The SRTF scheduler uses predictions from `sjf_length_predictor.py`:

```python
# In RequestView (from request.py)
self.predicted_response_len = request.predicted_response_len

# In SRTF scheduler
predicted_response_len = getattr(request_view, 'predicted_response_len', 0)
```

The predictor estimates response length based on:
- Prompt length
- Word count
- Question indicators
- Code/explanation keywords
- Complexity score

## Limitations

1. **Single Request Processing**: Non-batch version (batched SRTF not implemented)
2. **Prediction Accuracy**: Performance depends on `sjf_length_predictor` accuracy
3. **Overhead**: More preemptions = more context switches
4. **Starvation**: Very long jobs might wait indefinitely if short jobs keep arriving

## Future Enhancements

Potential improvements:
- **SRTF Batch**: Process multiple requests simultaneously
- **SRTF Dynamic Batch**: Continuously add requests while processing
- **Aging**: Prevent starvation by increasing priority of waiting jobs
- **Prediction Updates**: Refine remaining time as job progresses

## Files Modified/Created

- ✅ `scheduler/srtf_nonbatch.py` - SRTF scheduler implementation
- ✅ `main.py` - Added SRTF to scheduler list
- ✅ `test_srtf.py` - Test and demo script
- ✅ `SRTF_IMPLEMENTATION.md` - This documentation

## References

- Original request description: "implement a shortest remaining time first algorithm"
- SJF implementation: `scheduler/sjf_nonbatch.py`
- Length predictor: `length_predictor/sjf_length_predictor.py`
- Simulator: `simulator.py`

