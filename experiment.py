import os
import json
import matplotlib.pyplot as plt
from loader.prompt_engineering_dataset import PromptEngineeringDatasetLoader
from loader.sharegpt_dataset import ShareGPTDatasetLoader
from loader.test_dataset import TestDatasetLoader
from scheduler.fcfs_static_batch import FCFSStaticBatchScheduler
from scheduler.fcfs_nonbatch import FCFSNonBatchScheduler
from scheduler.fcfs_dyn_batch import FCFSDynamicBatchScheduler
from scheduler.fcfs_dyn_batch_predict import FCFSDynamicBatchPredictScheduler
from scheduler.fcfs_dyn_batch_predict_adj_over import FCFSDynamicBatchPredictAdjOverScheduler
from scheduler.fcfs_dyn_batch_predict_adj_avg import FCFSDynamicBatchPredictAdjAvgScheduler
from scheduler.fcfs_dyn_batch_predict_adj_iqr import FCFSDynamicBatchPredictAdjIQRScheduler
from scheduler.sjf_nonbatch import SJFNonBatchScheduler
from scheduler.sjf_dyn_batch import SJFDynamicBatchScheduler
from scheduler.sjf_dyn_batch_predict import SJFDynamicBatchPredictScheduler
from scheduler.sjf_dyn_batch_predict_adj_over import SJFDynamicBatchPredictAdjOverScheduler
from scheduler.sjf_dyn_batch_predict_adj_iqr import SJFDynamicBatchPredictAdjIQRScheduler
from scheduler.srpt_dyn_batch_predict import SRPTDynamicBatchPredictScheduler
from scheduler.srpt_dyn_batch_predict_adj_over import SRPTDynamicBatchPredictAdjOverScheduler
from scheduler.srpt_dyn_batch_predict_adj_iqr import SRPTDynamicBatchPredictAdjIQRScheduler
from scheduler.srpt_dyn_batch_predict_sched_delay_adj_iqr import SRPTDynamicBatchPredictScheduleDelayAdjIQRScheduler
from scheduler.bi_dyn_batch_predict_preemptive import BicriteriaDynamicBatchPredictPreemptiveScheduler
from scheduler.bi_dyn_batch_predict_non_preemptive import BicriteriaDynamicBatchPredictNonPreemptiveScheduler
from scheduler.bi_dyn_batch_predict_preemptive_adj_over import BicriteriaDynamicBatchPredictPreemptiveAdjOverScheduler
from scheduler.bi_dyn_batch_predict_preemptive_adj_iqr import BicriteriaDynamicBatchPredictPreemptiveAdjIQRScheduler
from scheduler.bi_dyn_batch_predict_sched_delay_preemptive_adj_iqr import BicriteriaDynamicBatchPredictScheduleDelayPreemptiveAdjIQRScheduler
from request import Request


RESULTS_PATH = './results'

SCHEDULER_DICT = {
    'FCFS Non-Batch': FCFSNonBatchScheduler,
    'FCFS Static Batch': FCFSStaticBatchScheduler,
    'FCFS Dynamic Batch': FCFSDynamicBatchScheduler,
    'FCFS Dynamic Batch Predict': FCFSDynamicBatchPredictScheduler,
    'FCFS Dynamic Batch Predict Overestimation Adjustor': FCFSDynamicBatchPredictAdjOverScheduler,
    'FCFS Dynamic Batch Predict Average Adjustor': FCFSDynamicBatchPredictAdjAvgScheduler,
    'FCFS Dynamic Batch Predict IQR Adjustor': FCFSDynamicBatchPredictAdjIQRScheduler,
    'SJF Non-Batch': SJFNonBatchScheduler,
    'SJF Dynamic Batch': SJFDynamicBatchScheduler,
    'SJF Dynamic Batch Predict': SJFDynamicBatchPredictScheduler,
    'SJF Dynamic Batch Predict Overestimation Adjustor': SJFDynamicBatchPredictAdjOverScheduler,
    'SJF Dynamic Batch Predict IQR Adjustor': SJFDynamicBatchPredictAdjIQRScheduler,
    'SRPT Dynamic Batch Predict': SRPTDynamicBatchPredictScheduler,
    'SRPT Dynamic Batch Predict Overestimation Adjustor': SRPTDynamicBatchPredictAdjOverScheduler,
    'SRPT Dynamic Batch Predict IQR Adjustor': SRPTDynamicBatchPredictAdjIQRScheduler,
    'SRPT Dynamic Batch Predict Schedule Delay IQR Adjustor': SRPTDynamicBatchPredictScheduleDelayAdjIQRScheduler,
    'Bicriteria Dynamic Batch Predict Preemptive': BicriteriaDynamicBatchPredictPreemptiveScheduler,
    'Bicriteria Dynamic Batch Predict Non-Preemptive': BicriteriaDynamicBatchPredictNonPreemptiveScheduler,
    'Bicriteria Dynamic Batch Predict Preemptive Overestimation Adjustor': BicriteriaDynamicBatchPredictPreemptiveAdjOverScheduler,
    'Bicriteria Dynamic Batch Predict Preemptive IQR Adjustor': BicriteriaDynamicBatchPredictPreemptiveAdjIQRScheduler,
    'Bicriteria Dynamic Batch Predict Schedule Delay Preemptive IQR Adjustor': BicriteriaDynamicBatchPredictScheduleDelayPreemptiveAdjIQRScheduler
}

PROMPT_ENGINEERING_DEFAULT_DATA_PATH = './data/prompt_engineering_dataset.csv'
PROMPT_ENGINEERING_DEFAULT_MAX_DATA_SIZE = 5010  # max is 5010
PROMPT_ENGINEERING_DEFAULT_REQUEST_RATE = 1/3
PROMPT_ENGINEERING_DEFAULT_PREDICTED_LEN_STDEV = 0.1
PROMPT_ENGINEERING_DEFAULT_VRAM_SLOTS = 100

SHAREGPT_DEFAULT_DATA_PATH = './data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json'
SHAREGPT_DEFAULT_MAX_CONVERSATION_COUNT = 94145  # max is 94145, 500 conversations is 1616 requests, 94145 conversations is 162105 requests
SHAREGPT_DEFAULT_CONVERSATION_RATE = 1/100
SHAREGPT_DEFAULT_PROMPT_RATE = 1/20
SHAREGPT_DEFAULT_PREDICTED_LEN_STDEV = 0.1
SHAREGPT_DEFAULT_VRAM_SLOTS = 2000
SHAREGPT_DEFAULT_MAX_CONVERSATION_TOKEN_COUNT = 1000

DATASET_DEFAULT_PARAMETERS_LIST = [
    (
        "prompt_engineering",
        PromptEngineeringDatasetLoader(
            PROMPT_ENGINEERING_DEFAULT_DATA_PATH,
            PROMPT_ENGINEERING_DEFAULT_MAX_DATA_SIZE,
            PROMPT_ENGINEERING_DEFAULT_REQUEST_RATE,
            SHAREGPT_DEFAULT_PREDICTED_LEN_STDEV
        ),
        PROMPT_ENGINEERING_DEFAULT_VRAM_SLOTS
    ),
    (
        "sharegpt",
        ShareGPTDatasetLoader(
            SHAREGPT_DEFAULT_DATA_PATH,
            SHAREGPT_DEFAULT_MAX_CONVERSATION_COUNT,
            SHAREGPT_DEFAULT_CONVERSATION_RATE,
            SHAREGPT_DEFAULT_PROMPT_RATE,
            SHAREGPT_DEFAULT_PREDICTED_LEN_STDEV,
            SHAREGPT_DEFAULT_MAX_CONVERSATION_TOKEN_COUNT
        ),
        SHAREGPT_DEFAULT_VRAM_SLOTS
    )
]


class Experiment:
    def __init__(self, experiment_name, schedulers, datasets, visualization_x_axis, save_experiment_results=False, save_timeline_visualizations=False):
        self.experiment_name = experiment_name
        self.save_experiment_results = save_experiment_results
        self.save_timeline_visualizations = save_timeline_visualizations
        self.schedulers = [(scheduler_name, SCHEDULER_DICT[scheduler_name]) for scheduler_name in schedulers]
        self.datasets = datasets
        self.visualization_x_axis = visualization_x_axis
        self.visualization_y_axis = {scheduler_name: [] for scheduler_name in schedulers}
    
    def add_result(self, scheduler_name, average_latency):
        self.visualization_y_axis[scheduler_name].append(average_latency)
    
    def visualize_results(self):
        if not self.save_experiment_results:
            return

        results_path = os.path.join(RESULTS_PATH, self.experiment_name)
        os.makedirs(results_path, exist_ok=True)
        png_path = os.path.join(results_path, "result.png")
        json_path = os.path.join(results_path, "result_data.json")

        plt.figure(figsize=(10, 6), dpi=200)

        x_name, x = next(iter(self.visualization_x_axis.items()))

        snapshot = {
            "experiment_name": self.experiment_name,
            "x_name": x_name,
            "x_values": x,
            "schedulers": {
                scheduler_name: self.visualization_y_axis[scheduler_name]
                for scheduler_name, _ in self.schedulers
            }
        }

        with open(json_path, "w") as f:
            json.dump(snapshot, f, indent=4)

        print("average")
        for scheduler_name, _ in self.schedulers:
            y = self.visualization_y_axis[scheduler_name]

            (line,) = plt.plot(
                x,
                y,
                marker="o",
                markersize=3,
                linewidth=1,
                label=scheduler_name
            )

            color = line.get_color()
            avg = sum(y) / len(y)
            plt.axhline(
                y=avg,
                linestyle="--",
                linewidth=0.8,
                color=color,
                alpha=0.7,
                label=f"{scheduler_name} (avg={avg:.3f})"
            )

            print(f"{scheduler_name}: {avg:.3f} time units")

        plt.xlabel(x_name)
        plt.ylabel("Average Latency")
        plt.title(self.experiment_name)
        plt.legend(fontsize=7)
        plt.grid(True, linewidth=0.5, alpha=0.6)

        plt.savefig(png_path, bbox_inches="tight")
        plt.close()

        print() # newline


# ============================================================================
# PROMPT ENGINEERING VARIED SLOT STDEV=0.1 EXPERIMENT
# ============================================================================
prompt_engineering_varied_slots = Experiment(
    "Prompt Engineering Varied VRAM Capacity Slots",
    save_experiment_results=True,
    schedulers=[
        # "FCFS Non-Batch",
        # "FCFS Static Batch",
        # "FCFS Dynamic Batch",
        # "FCFS Dynamic Batch Predict",
        # "FCFS Dynamic Batch Predict Overestimation Adjustor",
        # "FCFS Dynamic Batch Predict Average Adjustor",
        "FCFS Dynamic Batch Predict IQR Adjustor",
        # "SJF Non-Batch",
        # "SJF Dynamic Batch",
        # "SJF Dynamic Batch Predict",
        # "SJF Dynamic Batch Predict Overestimation Adjustor",
        "SJF Dynamic Batch Predict IQR Adjustor",
        # "SRPT Dynamic Batch Predict",
        # "SRPT Dynamic Batch Predict Overestimation Adjustor",
        "SRPT Dynamic Batch Predict IQR Adjustor",
        "SRPT Dynamic Batch Predict Schedule Delay IQR Adjustor",
        # "Bicriteria Dynamic Batch Predict Preemptive",
        # "Bicriteria Dynamic Batch Predict Non-Preemptive",
        # "Bicriteria Dynamic Batch Predict Preemptive Overestimation Adjustor",
        # "Bicriteria Dynamic Batch Predict Preemptive IQR Adjustor",
        "Bicriteria Dynamic Batch Predict Schedule Delay Preemptive IQR Adjustor"
    ],
    datasets=[],
    visualization_x_axis={"vram_slots": []}
)
for i in range(8):
    vram_slots = 90 + i*5
    prompt_engineering_varied_slots.datasets.append(
        (
            f"prompt_engineering_{vram_slots}_slots",
            PromptEngineeringDatasetLoader(
                PROMPT_ENGINEERING_DEFAULT_DATA_PATH,
                PROMPT_ENGINEERING_DEFAULT_MAX_DATA_SIZE,
                PROMPT_ENGINEERING_DEFAULT_REQUEST_RATE,
                PROMPT_ENGINEERING_DEFAULT_PREDICTED_LEN_STDEV
            ),
            vram_slots
        )
    )
    prompt_engineering_varied_slots.visualization_x_axis["vram_slots"].append(vram_slots)


# ============================================================================
# SHAREGPT VARIED SLOTS STDEV=0.1 EXPERIMENT
# ============================================================================
sharegpt_varied_slots = Experiment(
    "ShareGPT Varied VRAM Capacity Slots",
    save_experiment_results=True,
    schedulers=[
        # "FCFS Non-Batch",
        # "FCFS Static Batch",
        # "FCFS Dynamic Batch",
        # "FCFS Dynamic Batch Predict",
        # "FCFS Dynamic Batch Predict Overestimation Adjustor",
        # "FCFS Dynamic Batch Predict Average Adjustor",
        "FCFS Dynamic Batch Predict IQR Adjustor",
        # "SJF Non-Batch",
        # "SJF Dynamic Batch",
        # "SJF Dynamic Batch Predict",
        # "SJF Dynamic Batch Predict Overestimation Adjustor",
        "SJF Dynamic Batch Predict IQR Adjustor",
        # "SRPT Dynamic Batch Predict",
        # "SRPT Dynamic Batch Predict Overestimation Adjustor",
        "SRPT Dynamic Batch Predict IQR Adjustor",
        "SRPT Dynamic Batch Predict Schedule Delay IQR Adjustor",
        # "Bicriteria Dynamic Batch Predict Preemptive",
        # "Bicriteria Dynamic Batch Predict Non-Preemptive",
        # "Bicriteria Dynamic Batch Predict Preemptive Overestimation Adjustor",
        # "Bicriteria Dynamic Batch Predict Preemptive IQR Adjustor",
        "Bicriteria Dynamic Batch Predict Schedule Delay Preemptive IQR Adjustor"
    ],
    datasets=[],
    visualization_x_axis={"vram_slots": []}
)
for i in range(8):
    vram_slots = 9000 + i * 250
    sharegpt_varied_slots.datasets.append(
        (
            f"sharegpt_{vram_slots}_slots",
            ShareGPTDatasetLoader(
                SHAREGPT_DEFAULT_DATA_PATH,
                SHAREGPT_DEFAULT_MAX_CONVERSATION_COUNT,
                SHAREGPT_DEFAULT_CONVERSATION_RATE,
                SHAREGPT_DEFAULT_PROMPT_RATE,
                0.1,
                SHAREGPT_DEFAULT_MAX_CONVERSATION_TOKEN_COUNT
            ),
            vram_slots
        )
    )
    sharegpt_varied_slots.visualization_x_axis["vram_slots"].append(vram_slots)


# ============================================================================
# PROMPT ENGINEERING VARIED SLOTS PERFECT PREDICTION EXPERIMENT
# ============================================================================
prompt_engineering_varied_slots_perfect = Experiment(
    "Prompt Engineering Varied VRAM Capacity Slots, Perfect Predictions",
    save_experiment_results=True,
    schedulers=[
        # "FCFS Non-Batch",
        # "FCFS Static Batch",
        # "FCFS Dynamic Batch",
        # "FCFS Dynamic Batch Predict",
        # "FCFS Dynamic Batch Predict Overestimation Adjustor",
        # "FCFS Dynamic Batch Predict Average Adjustor",
        "FCFS Dynamic Batch Predict IQR Adjustor",
        # "SJF Non-Batch",
        # "SJF Dynamic Batch",
        # "SJF Dynamic Batch Predict",
        # "SJF Dynamic Batch Predict Overestimation Adjustor",
        "SJF Dynamic Batch Predict IQR Adjustor",
        # "SRPT Dynamic Batch Predict",
        # "SRPT Dynamic Batch Predict Overestimation Adjustor",
        "SRPT Dynamic Batch Predict IQR Adjustor",
        "SRPT Dynamic Batch Predict Schedule Delay IQR Adjustor",
        # "Bicriteria Dynamic Batch Predict Preemptive",
        # "Bicriteria Dynamic Batch Predict Non-Preemptive",
        # "Bicriteria Dynamic Batch Predict Preemptive Overestimation Adjustor",
        # "Bicriteria Dynamic Batch Predict Preemptive IQR Adjustor",
        "Bicriteria Dynamic Batch Predict Schedule Delay Preemptive IQR Adjustor"
    ],
    datasets=[],
    visualization_x_axis={"vram_slots": []}
)
for i in range(10):
    vram_slots = 80 + i*5
    prompt_engineering_varied_slots_perfect.datasets.append(
        (
            f"prompt_engineering_{vram_slots}_slots",
            PromptEngineeringDatasetLoader(
                PROMPT_ENGINEERING_DEFAULT_DATA_PATH,
                PROMPT_ENGINEERING_DEFAULT_MAX_DATA_SIZE,
                PROMPT_ENGINEERING_DEFAULT_REQUEST_RATE,
                SHAREGPT_DEFAULT_PREDICTED_LEN_STDEV
            ),
            vram_slots
        )
    )
    prompt_engineering_varied_slots_perfect.visualization_x_axis["vram_slots"].append(vram_slots)


# ============================================================================
# SHAREGPT VARIED SLOTS PERFECT PREDICTION EXPERIMENT
# ============================================================================
sharegpt_varied_slots_perfect = Experiment(
    "ShareGPT Varied VRAM Capacity Slots, Perfect Predictions",
    save_experiment_results=True,
    schedulers=[
        # "FCFS Non-Batch",
        # "FCFS Static Batch",
        # "FCFS Dynamic Batch",
        # "FCFS Dynamic Batch Predict",
        # "FCFS Dynamic Batch Predict Overestimation Adjustor",
        # "FCFS Dynamic Batch Predict Average Adjustor",
        "FCFS Dynamic Batch Predict IQR Adjustor",
        # "SJF Non-Batch",
        # "SJF Dynamic Batch",
        # "SJF Dynamic Batch Predict",
        # "SJF Dynamic Batch Predict Overestimation Adjustor",
        "SJF Dynamic Batch Predict IQR Adjustor",
        # "SRPT Dynamic Batch Predict",
        # "SRPT Dynamic Batch Predict Overestimation Adjustor",
        "SRPT Dynamic Batch Predict IQR Adjustor",
        "SRPT Dynamic Batch Predict Schedule Delay IQR Adjustor",
        # "Bicriteria Dynamic Batch Predict Preemptive",
        # "Bicriteria Dynamic Batch Predict Non-Preemptive",
        # "Bicriteria Dynamic Batch Predict Preemptive Overestimation Adjustor",
        # "Bicriteria Dynamic Batch Predict Preemptive IQR Adjustor",
        "Bicriteria Dynamic Batch Predict Schedule Delay Preemptive IQR Adjustor"
    ],
    datasets=[],
    visualization_x_axis={"vram_slots": []}
)
for i in range(13):
    vram_slots = 9000 + i * 500
    sharegpt_varied_slots_perfect.datasets.append(
        (
            f"sharegpt_{vram_slots}_slots",
            ShareGPTDatasetLoader(
                SHAREGPT_DEFAULT_DATA_PATH,
                SHAREGPT_DEFAULT_MAX_CONVERSATION_COUNT,
                SHAREGPT_DEFAULT_CONVERSATION_RATE,
                SHAREGPT_DEFAULT_PROMPT_RATE,
                SHAREGPT_DEFAULT_PREDICTED_LEN_STDEV,
                SHAREGPT_DEFAULT_MAX_CONVERSATION_TOKEN_COUNT
            ),
            vram_slots
        )
    )
    sharegpt_varied_slots_perfect.visualization_x_axis["vram_slots"].append(vram_slots)


# ============================================================================
# PROMPT ENGINEERING VARIED SLOTS PERFECT PREDICTION FIXED PREFILL EXPERIMENT
# ============================================================================
prompt_engineering_varied_slots_perfect_fixed = Experiment(
    "Prompt Engineering Varied VRAM Capacity Slots, Perfect Predictions, Fixed Prefill",
    save_experiment_results=True,
    schedulers=[
        # "FCFS Non-Batch",
        # "FCFS Static Batch",
        # "FCFS Dynamic Batch",
        # "FCFS Dynamic Batch Predict",
        # "FCFS Dynamic Batch Predict Overestimation Adjustor",
        # "FCFS Dynamic Batch Predict Average Adjustor",
        "FCFS Dynamic Batch Predict IQR Adjustor",
        # "SJF Non-Batch",
        # "SJF Dynamic Batch",
        # "SJF Dynamic Batch Predict",
        # "SJF Dynamic Batch Predict Overestimation Adjustor",
        "SJF Dynamic Batch Predict IQR Adjustor",
        # "SRPT Dynamic Batch Predict",
        # "SRPT Dynamic Batch Predict Overestimation Adjustor",
        "SRPT Dynamic Batch Predict IQR Adjustor",
        "SRPT Dynamic Batch Predict Schedule Delay IQR Adjustor",
        # "Bicriteria Dynamic Batch Predict Preemptive",
        # "Bicriteria Dynamic Batch Predict Non-Preemptive",
        # "Bicriteria Dynamic Batch Predict Preemptive Overestimation Adjustor",
        # "Bicriteria Dynamic Batch Predict Preemptive IQR Adjustor",
        "Bicriteria Dynamic Batch Predict Schedule Delay Preemptive IQR Adjustor"
    ],
    datasets=[],
    visualization_x_axis={"vram_slots": []}
)
for i in range(10):
    vram_slots = 80 + i*5
    prompt_engineering_varied_slots_perfect_fixed.datasets.append(
        (
            f"prompt_engineering_{vram_slots}_slots",
            PromptEngineeringDatasetLoader(
                PROMPT_ENGINEERING_DEFAULT_DATA_PATH,
                PROMPT_ENGINEERING_DEFAULT_MAX_DATA_SIZE,
                PROMPT_ENGINEERING_DEFAULT_REQUEST_RATE,
                SHAREGPT_DEFAULT_PREDICTED_LEN_STDEV,
                True
            ),
            vram_slots
        )
    )
    prompt_engineering_varied_slots_perfect_fixed.visualization_x_axis["vram_slots"].append(vram_slots)


# ============================================================================
# TEST DATASET EXPERIMENT
# ============================================================================
test_dataset = Experiment(
    "Test Dataset",
    schedulers=[
        # "FCFS Non-Batch",
        # "FCFS Static Batch",
        # "FCFS Dynamic Batch",
        # "FCFS Dynamic Batch Predict",
        # "FCFS Dynamic Batch Predict Overestimation Adjustor",
        # "FCFS Dynamic Batch Predict Average Adjustor",
        # "FCFS Dynamic Batch Predict IQR Adjustor",
        # "SJF Non-Batch",
        # "SJF Dynamic Batch",
        # "SJF Dynamic Batch Predict",
        # "SJF Dynamic Batch Predict Overestimation Adjustor",
        "SJF Dynamic Batch Predict IQR Adjustor",
        # "SRPT Dynamic Batch Predict",
        # "SRPT Dynamic Batch Predict Overestimation Adjustor",
        # "SRPT Dynamic Batch Predict IQR Adjustor",
        "SRPT Dynamic Batch Predict Schedule Delay IQR Adjustor",
        # "Bicriteria Dynamic Batch Predict Preemptive",
        # "Bicriteria Dynamic Batch Predict Non-Preemptive",
        # "Bicriteria Dynamic Batch Predict Preemptive Overestimation Adjustor",
        # "Bicriteria Dynamic Batch Predict Preemptive IQR Adjustor",
        # "Bicriteria Dynamic Batch Predict Schedule Delay Preemptive IQR Adjustor"
    ],
    datasets=[
        (
            "test_dataset",
            TestDatasetLoader(
                [
                    Request("Request 0", 1, 9, 0, 9),
                    Request("Request 1", 1, 4, 1, 4),
                ]
            ),
            10
        ),
    ],
    visualization_x_axis=None,
    save_timeline_visualizations=True
)


# ============================================================================
# PROMPT ENGINEERING VARIED STDEV EXPERIMENT
# ============================================================================
prompt_engineering_varied_stdev = Experiment(
    "Prompt Engineering Varied STDEV",
    save_experiment_results=True,
    schedulers=[
        # "FCFS Non-Batch",
        # "FCFS Static Batch",
        # "FCFS Dynamic Batch",
        # "FCFS Dynamic Batch Predict",
        # "FCFS Dynamic Batch Predict Overestimation Adjustor",
        "FCFS Dynamic Batch Predict Average Adjustor",
        # "FCFS Dynamic Batch Predict IQR Adjustor",
        # "SJF Non-Batch",
        # "SJF Dynamic Batch",
        # "SJF Dynamic Batch Predict",
        # "SJF Dynamic Batch Predict Overestimation Adjustor",
        # "SJF Dynamic Batch Predict IQR Adjustor",
        # "SRPT Dynamic Batch Predict",
        # "SRPT Dynamic Batch Predict Overestimation Adjustor",
        # "SRPT Dynamic Batch Predict IQR Adjustor",
        # "SRPT Dynamic Batch Predict Schedule Delay IQR Adjustor",
        # "Bicriteria Dynamic Batch Predict Preemptive",
        # "Bicriteria Dynamic Batch Predict Non-Preemptive",
        # "Bicriteria Dynamic Batch Predict Preemptive Overestimation Adjustor",
        # "Bicriteria Dynamic Batch Predict Preemptive IQR Adjustor",
        # "Bicriteria Dynamic Batch Predict Schedule Delay Preemptive IQR Adjustor"
    ],
    datasets=[],
    visualization_x_axis={"stdev": []}
)
for i in range(16):
    stdev = i * 0.0125
    prompt_engineering_varied_stdev.datasets.append(
        (
            f"prompt_engineering_{stdev:.4f}_stdev",
            PromptEngineeringDatasetLoader(
                PROMPT_ENGINEERING_DEFAULT_DATA_PATH,
                PROMPT_ENGINEERING_DEFAULT_MAX_DATA_SIZE,
                PROMPT_ENGINEERING_DEFAULT_REQUEST_RATE,
                stdev
            ),
            100
        )
    )
    prompt_engineering_varied_stdev.visualization_x_axis["stdev"].append(stdev)


# ============================================================================
# SHAREGPT VARIED STDEV EXPERIMENT
# ============================================================================
sharegpt_varied_stdev = Experiment(
    "ShareGPT Varied STDEV",
    save_experiment_results=True,
    schedulers=[
        "FCFS Non-Batch",
        "FCFS Static Batch",
        "FCFS Dynamic Batch",
        "FCFS Dynamic Batch Predict",
        "FCFS Dynamic Batch Predict Overestimation Adjustor",
        "FCFS Dynamic Batch Predict Average Adjustor",
        "FCFS Dynamic Batch Predict IQR Adjustor",
        # "SJF Non-Batch",
        # "SJF Dynamic Batch",
        # "SJF Dynamic Batch Predict",
        # "SJF Dynamic Batch Predict Overestimation Adjustor",
        # "SJF Dynamic Batch Predict IQR Adjustor",
        # "SRPT Dynamic Batch Predict",
        # "SRPT Dynamic Batch Predict Overestimation Adjustor",
        # "SRPT Dynamic Batch Predict IQR Adjustor",
        # "SRPT Dynamic Batch Predict Schedule Delay IQR Adjustor",
        # "Bicriteria Dynamic Batch Predict Preemptive",
        # "Bicriteria Dynamic Batch Predict Non-Preemptive",
        # "Bicriteria Dynamic Batch Predict Preemptive Overestimation Adjustor",
        # "Bicriteria Dynamic Batch Predict Preemptive IQR Adjustor",
        # "Bicriteria Dynamic Batch Predict Schedule Delay Preemptive IQR Adjustor"
    ],
    datasets=[],
    visualization_x_axis={"stdev": []}
)
for i in range(16):
    stdev = i * 0.0125
    sharegpt_varied_stdev.datasets.append(
        (
            f"sharegpt_{stdev:.4f}_stdev",
            ShareGPTDatasetLoader(
                SHAREGPT_DEFAULT_DATA_PATH,
                SHAREGPT_DEFAULT_MAX_CONVERSATION_COUNT,
                SHAREGPT_DEFAULT_CONVERSATION_RATE,
                SHAREGPT_DEFAULT_PROMPT_RATE,
                stdev,
                SHAREGPT_DEFAULT_MAX_CONVERSATION_TOKEN_COUNT
            ),
            10000
        )
    )
    sharegpt_varied_stdev.visualization_x_axis["stdev"].append(stdev)


# ============================================================================
# DEFAULT EXPERIMENT
# ============================================================================
default = Experiment(
    "default",
    schedulers=[
        # "FCFS Non-Batch",
        # "FCFS Static Batch",
        # "FCFS Dynamic Batch",
        # "FCFS Dynamic Batch Predict",
        # "FCFS Dynamic Batch Predict Overestimation Adjustor",
        # "FCFS Dynamic Batch Predict Average Adjustor",
        # "FCFS Dynamic Batch Predict IQR Adjustor",
        # "SJF Non-Batch",
        # "SJF Dynamic Batch",
        # "SJF Dynamic Batch Predict",
        # "SJF Dynamic Batch Predict Overestimation Adjustor",
        # "SJF Dynamic Batch Predict IQR Adjustor",
        # "SRPT Dynamic Batch Predict",
        # "SRPT Dynamic Batch Predict Overestimation Adjustor",
        # "SRPT Dynamic Batch Predict IQR Adjustor",
        # "SRPT Dynamic Batch Predict Schedule Delay IQR Adjustor",
        # "Bicriteria Dynamic Batch Predict Preemptive",
        # "Bicriteria Dynamic Batch Predict Non-Preemptive",
        # "Bicriteria Dynamic Batch Predict Preemptive Overestimation Adjustor",
        # "Bicriteria Dynamic Batch Predict Preemptive IQR Adjustor",
        # "Bicriteria Dynamic Batch Predict Schedule Delay Preemptive IQR Adjustor"
    ],
    datasets=[
        (
            "prompt_engineering",
            PromptEngineeringDatasetLoader(
                PROMPT_ENGINEERING_DEFAULT_DATA_PATH,
                PROMPT_ENGINEERING_DEFAULT_MAX_DATA_SIZE,
                PROMPT_ENGINEERING_DEFAULT_REQUEST_RATE,
                SHAREGPT_DEFAULT_PREDICTED_LEN_STDEV
            ),
            PROMPT_ENGINEERING_DEFAULT_VRAM_SLOTS
        ),
        # (
        #     "sharegpt",
        #     ShareGPTDatasetLoader(
        #         SHAREGPT_DEFAULT_DATA_PATH,
        #         SHAREGPT_DEFAULT_MAX_CONVERSATION_COUNT,
        #         SHAREGPT_DEFAULT_CONVERSATION_RATE,
        #         SHAREGPT_DEFAULT_PROMPT_RATE,
        #         SHAREGPT_DEFAULT_PREDICTED_LEN_STDEV,
        #         SHAREGPT_DEFAULT_MAX_CONVERSATION_TOKEN_COUNT
        #     ),
        #     SHAREGPT_DEFAULT_VRAM_SLOTS
        # )
    ],
    visualization_x_axis=None
)

EXPERIMENTS = [
    # default,
    # prompt_engineering_varied_slots_perfect_fixed,
    # prompt_engineering_varied_slots_perfect,
    # sharegpt_varied_slots_perfect,
    # prompt_engineering_varied_slots,
    # sharegpt_varied_slots,
    prompt_engineering_varied_stdev,
    # sharegpt_varied_stdev,
    # test_dataset,
]
