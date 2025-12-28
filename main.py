import numpy as np
import copy
from loader.prompt_engineering_dataset import PromptEngineeringDatasetLoader
from loader.sharegpt_dataset import ShareGPTDatasetLoader
from scheduler.fcfs_batch import FCFSBatchScheduler
from scheduler.fcfs_nonbatch import FCFSNonBatchScheduler
from scheduler.fcfs_dyn_batch import FCFSDynamicBatchScheduler
from scheduler.fcfs_dyn_batch_predict import FCFSDynamicBatchPredictScheduler
from scheduler.sjf_nonbatch import SJFNonBatchScheduler
from scheduler.sjf_dyn_batch_predict import SJFDynamicBatchPredictScheduler
from scheduler.srpt_dyn_batch_predict import SRPTDynamicBatchPredictScheduler
from scheduler.srpt_dyn_batch_predict_adj import SRPTDynamicBatchPredictAdjustmentScheduler
from simulator import Simulator
from gpu import GPU, GPUView

np.random.seed(42)

RESULTS_PATH = './results'
SCHEDULER_NAMES = [
    'fcfs_nonbatch',
    'fcfs_batch',
    'fcfs_dynamic_batch',
    'fcfs_dynamic_batch_predict',
    'sjf_nonbatch',
    'sjf_dynamic_batch_predict',
    'srpt_dynamic_batch_predict',
    'srpt_dynamic_batch_predict_adjustment'
]
SCHEDULER_DICT = {
    'fcfs_nonbatch': FCFSNonBatchScheduler,
    'fcfs_batch': FCFSBatchScheduler,
    'fcfs_dynamic_batch': FCFSDynamicBatchScheduler,
    'fcfs_dynamic_batch_predict': FCFSDynamicBatchPredictScheduler,
    'sjf_nonbatch': SJFNonBatchScheduler,
    'sjf_dynamic_batch_predict': SJFDynamicBatchPredictScheduler,
    'srpt_dynamic_batch_predict': SRPTDynamicBatchPredictScheduler,
    'srpt_dynamic_batch_predict_adjustment': SRPTDynamicBatchPredictAdjustmentScheduler
}
SAVE_VISUALIZATIONS = False

PROMPT_ENGINEERING_DATA_PATH = './data/prompt_engineering_dataset.csv'
PROMPT_ENGINEERING_MAX_DATA_SIZE = 5010 # max is 5010
PROMPT_ENGINEERING_REQUEST_RATE = 1/3
PROMPT_ENGINEERING_PREDICTED_LEN_STDEV = 0
PROMPT_ENGINEERING_VRAM_SLOTS = 150

SHAREGPT_DATA_PATH = './data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json'
SHAREGPT_MAX_CONVERSATION_COUNT = 500 # max is 94145
SHAREGPT_CONVERSATION_RATE = 1/100
SHAREGPT_PROMPT_RATE = 1/20
SHAREGPT_PREDICTED_LEN_STDEV = 0
SHAREGPT_VRAM_SLOTS = 15000
SHAREGPT_MAX_CONTEXT_WINDOW = 8192

EXPERIMENT_PARAMETERS = {
    "prompt_engineering": (
        PromptEngineeringDatasetLoader(
            PROMPT_ENGINEERING_DATA_PATH, 
            PROMPT_ENGINEERING_MAX_DATA_SIZE, 
            PROMPT_ENGINEERING_REQUEST_RATE, 
            PROMPT_ENGINEERING_PREDICTED_LEN_STDEV
        ),
        PROMPT_ENGINEERING_VRAM_SLOTS 
    ),
    "sharegpt": (
        ShareGPTDatasetLoader(
            SHAREGPT_DATA_PATH, 
            SHAREGPT_MAX_CONVERSATION_COUNT, 
            SHAREGPT_CONVERSATION_RATE, 
            SHAREGPT_PROMPT_RATE, 
            SHAREGPT_PREDICTED_LEN_STDEV,
            SHAREGPT_MAX_CONTEXT_WINDOW
        ),
        SHAREGPT_VRAM_SLOTS
    )
}

def main():
    for dataset_name, (loader, vram_slots) in EXPERIMENT_PARAMETERS.items():
        print(f"{dataset_name} dataset")
        dataset = loader.load()

        for scheduler_name in SCHEDULER_NAMES:
            dataset_copy = copy.deepcopy(dataset)

            gpu = GPU(vram_slots)

            scheduler = SCHEDULER_DICT[scheduler_name](GPUView(gpu))

            simulator = Simulator(dataset_copy, gpu, scheduler)
            simulator.run()

            dataset_copy.show_average_latency(scheduler_name)
            if SAVE_VISUALIZATIONS:
                dataset_copy.visualize_request_history(RESULTS_PATH, scheduler_name, dataset_name)
                gpu.visualize_history(RESULTS_PATH, scheduler_name, dataset_name)

        print() # newline

if __name__ == "__main__":
    main()
