# SRTF Quick Start Guide

## What is SRTF?

**Shortest Remaining Time First (SRTF)** is the **preemptive version of SJF**. It always runs the job with the shortest remaining time, preempting longer jobs when shorter ones arrive.

## Quick Run

### 1. Test SRTF Only (Recommended for First Time)

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run test (20 requests, shows SRTF vs SJF comparison)
python test_srtf.py
```

**Output:**
- Average latency comparison
- Visualizations in `./results/` folder
- Shows preemptive behavior

### 2. Run Full Benchmark (All Schedulers)

```bash
python main.py
```

Runs all schedulers including:
- FCFS (3 variants)
- SJF (3 variants)
- **SRTF** (new!)

### 3. Run SRTF Only in Main

Edit `main.py` line 18-27:
```python
SCHEDULER_NAMES = [
    'srtf_nonbatch',
]
```

Then:
```bash
python main.py
```

## View Results

After running, open these HTML files in your browser:

- `results/test_srtf_nonbatch_request_timeline.html` - Timeline showing preemptions
- `results/test_srtf_nonbatch_vram_usage.html` - Memory usage over time

Look for:
- ✅ Lower average latency than SJF
- ✅ More state changes (preemptions) in timeline
- ✅ Short jobs complete faster even if they arrive late

## How SRTF Works

```
Time 0: Job A arrives (remaining: 10)
        → Run A

Time 2: Job B arrives (remaining: 3)
        → A has 8 remaining, B has 3
        → PREEMPT A, run B (shorter!)

Time 5: B completes
        → Run A again (8 remaining)

Time 13: A completes
```

**Key Point:** SRTF checks remaining time at each decision point and switches to the shortest job, even if it means interrupting the current job.

## Algorithm Summary

**Step 1:** At each time unit, check all requests (queue + running)

**Step 2:** Calculate remaining time for each:
- Queued request: `predicted_response_len`
- Running request: `predicted_response_len - decode_progress`

**Step 3:** Select the request with shortest remaining time

**Step 4:** If different from current → preempt current, schedule new

**Step 5:** Execute for 1 time unit, update remaining time

**Step 6:** Repeat until all complete

## Expected Results

SRTF typically achieves **10-15% better latency** than non-preemptive SJF because:
- Short jobs don't wait behind long jobs
- Adapts dynamically as jobs progress
- Optimal average waiting time (with perfect predictions)

## Troubleshooting

**Error: "No module named 'scheduler.srtf_nonbatch'"**
→ Make sure you're in the project root directory (`C:\Users\boxua\OneDrive\Desktop\fyp\`)

**Error: "FileNotFoundError: data/prompt_engineering_dataset.csv"**
→ Download the dataset from the Kaggle link in README.md

**No visualizations generated**
→ Set `SAVE_VISUALIZATIONS = True` in main.py (should be default)

## Files Created

- `scheduler/srtf_nonbatch.py` - SRTF implementation
- `test_srtf.py` - Test script
- `SRTF_IMPLEMENTATION.md` - Detailed documentation
- `SRTF_QUICKSTART.md` - This file

## Next Steps

1. Run `python test_srtf.py` to see SRTF in action
2. Check visualizations in `./results/` folder
3. Read `SRTF_IMPLEMENTATION.md` for technical details
4. Try different parameters in `test_srtf.py` or `main.py`

