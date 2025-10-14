import numpy as np
import copy
from loader.prompt_engineering_dataset import PromptEngineeringDatasetLoader
from scheduler.fcfs_batch import FCFSBatchScheduler
from scheduler.fcfs_nonbatch import FCFSNonBatchScheduler
from scheduler.fcfs_dyn_batch import FCFSDynamicBatchScheduler
from scheduler.fcfs_dyn_batch_predict import FCFSDynamicBatchPredictScheduler
from simulator import Simulator
from gpu import GPU, GPUView

np.random.seed(42)

DATA_PATH = './data/prompt_engineering_dataset.csv'
DATA_SIZE = 100
REQUEST_RATE = 1/5
PREDICTED_LEN_STDEV = 0.1
VRAM_SLOTS = 100
RESULTS_PATH = './results'
SCHEDULER_NAMES = [
    'fcfs_nonbatch',
    'fcfs_batch',
    'fcfs_dynamic_batch',
    'fcfs_dynamic_batch_predict',
]
SCHEDULER_DICT = {
    'fcfs_nonbatch': FCFSNonBatchScheduler,
    'fcfs_batch': FCFSBatchScheduler,
    'fcfs_dynamic_batch': FCFSDynamicBatchScheduler,
    'fcfs_dynamic_batch_predict': FCFSDynamicBatchPredictScheduler
}
SAVE_VISUALIZATIONS = True

def main():
    loader = PromptEngineeringDatasetLoader(
        DATA_PATH, DATA_SIZE, REQUEST_RATE, PREDICTED_LEN_STDEV)
    dataset = loader.load()

    for scheduler_name in SCHEDULER_NAMES:
        dataset_copy = copy.deepcopy(dataset)

        gpu = GPU(VRAM_SLOTS)

        scheduler = SCHEDULER_DICT[scheduler_name](GPUView(gpu))

        simulator = Simulator(dataset_copy, gpu, scheduler)
        simulator.run()

        dataset_copy.show_average_latency(scheduler_name)
        if SAVE_VISUALIZATIONS:
            dataset_copy.visualize_request_history(RESULTS_PATH, scheduler_name)
            gpu.visualize_history(RESULTS_PATH, scheduler_name)

        print() # newline

if __name__ == "__main__":
    main()
