"""
Test and demonstrate the SRTF (Shortest Remaining Time First) scheduler.

This script runs ONLY the SRTF scheduler to show its behavior and
compare it with SJF non-batch scheduler.
"""

import numpy as np
import copy
from loader.prompt_engineering_dataset import PromptEngineeringDatasetLoader
from scheduler.srtf_nonbatch import SRTFNonBatchScheduler
from scheduler.sjf_nonbatch import SJFNonBatchScheduler
from simulator import Simulator
from gpu import GPU, GPUView

np.random.seed(42)

# Configuration
DATA_PATH = './data/prompt_engineering_dataset.csv'
MAX_DATA_SIZE = 20  # Small dataset for testing
REQUEST_RATE = 1/3  # Request every 3 time units
PREDICTED_LEN_STDEV = 0.1
VRAM_SLOTS = 100
RESULTS_PATH = './results'

def main():
    print("="*70)
    print("SRTF (Shortest Remaining Time First) Scheduler Test")
    print("="*70)
    print()
    
    # Load dataset
    print("Loading dataset...")
    loader = PromptEngineeringDatasetLoader(
        DATA_PATH, 
        MAX_DATA_SIZE, 
        REQUEST_RATE, 
        PREDICTED_LEN_STDEV
    )
    dataset = loader.load()
    print(f"Loaded {MAX_DATA_SIZE} requests")
    print()
    
    # Test SRTF
    print("Running SRTF scheduler...")
    print("-" * 70)
    dataset_srtf = copy.deepcopy(dataset)
    gpu_srtf = GPU(VRAM_SLOTS)
    scheduler_srtf = SRTFNonBatchScheduler(GPUView(gpu_srtf))
    
    simulator_srtf = Simulator(dataset_srtf, gpu_srtf, scheduler_srtf)
    simulator_srtf.run()
    
    print("\n✓ SRTF completed")
    dataset_srtf.show_average_latency("SRTF")
    dataset_srtf.visualize_request_history(RESULTS_PATH, "srtf_nonbatch", "test")
    gpu_srtf.visualize_history(RESULTS_PATH, "srtf_nonbatch", "test")
    print()
    
    # Compare with SJF (non-preemptive)
    print("Running SJF scheduler for comparison...")
    print("-" * 70)
    dataset_sjf = copy.deepcopy(dataset)
    gpu_sjf = GPU(VRAM_SLOTS)
    scheduler_sjf = SJFNonBatchScheduler(GPUView(gpu_sjf))
    
    simulator_sjf = Simulator(dataset_sjf, gpu_sjf, scheduler_sjf)
    simulator_sjf.run()
    
    print("\n✓ SJF completed")
    dataset_sjf.show_average_latency("SJF")
    dataset_sjf.visualize_request_history(RESULTS_PATH, "sjf_nonbatch", "test")
    gpu_sjf.visualize_history(RESULTS_PATH, "sjf_nonbatch", "test")
    print()
    
    # Show comparison
    print("="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    srtf_latency = sum(
        req.response_timestamp - req.request_timestamp
        for req in dataset_srtf._requests.values()
    ) / len(dataset_srtf._requests)
    
    sjf_latency = sum(
        req.response_timestamp - req.request_timestamp
        for req in dataset_sjf._requests.values()
    ) / len(dataset_sjf._requests)
    
    print(f"SRTF Average Latency:  {srtf_latency:.3f} time units")
    print(f"SJF Average Latency:   {sjf_latency:.3f} time units")
    print(f"Improvement:           {((sjf_latency - srtf_latency) / sjf_latency * 100):.2f}%")
    print()
    
    print("SRTF Characteristics:")
    print("  ✓ Preemptive - can interrupt running jobs")
    print("  ✓ Dynamic - always runs shortest remaining time job")
    print("  ✓ Optimal - minimizes average waiting time (with perfect predictions)")
    print()
    
    print("Visualizations saved to ./results/ folder:")
    print("  - test_srtf_nonbatch_request_timeline.html")
    print("  - test_srtf_nonbatch_vram_usage.html")
    print("  - test_sjf_nonbatch_request_timeline.html")
    print("  - test_sjf_nonbatch_vram_usage.html")
    print()

if __name__ == "__main__":
    main()

