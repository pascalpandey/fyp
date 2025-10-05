import numpy as np
import copy
from loader.prompt_engineering_dataset import PromptEngineeringDatasetLoader
from scheduler.fcfs_batch import FCFSBatchScheduler
from scheduler.fcfs_nonbatch import FCFSNonBatchScheduler
from simulator import Simulator
from gpu import GPU, GPUView

np.random.seed(42)

DATA_PATH = './data/prompt_engineering_dataset.csv'
DATA_SIZE = 100
REQUEST_RATE = 1/5
VRAM_SLOTS = 100
RESULTS_PATH = './results'
SCHEDULER_NAMES = 'fcfs_batch,fcfs_nonbatch'
SCHEDULER_DICT = {
    'fcfs_batch': FCFSBatchScheduler,
    'fcfs_nonbatch': FCFSNonBatchScheduler
}


def main():
    loader = PromptEngineeringDatasetLoader(DATA_PATH, DATA_SIZE, REQUEST_RATE)
    dataset = loader.load()

    for scheduler_name in SCHEDULER_NAMES.split(','):
        dataset_copy = copy.deepcopy(dataset)

        gpu = GPU(VRAM_SLOTS)

        scheduler = SCHEDULER_DICT[scheduler_name](GPUView(gpu))

        simulator = Simulator(dataset_copy, gpu, scheduler)
        simulator.run()

        dataset_copy.show_results(RESULTS_PATH, scheduler_name)
        gpu.visualize_history(RESULTS_PATH, scheduler_name)


if __name__ == "__main__":
    main()
