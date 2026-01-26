import os
import matplotlib.pyplot as plt
from loader.prompt_engineering_dataset import PromptEngineeringDatasetLoader
from loader.sharegpt_dataset import ShareGPTDatasetLoader
from loader.test_dataset import TestDatasetLoader
from scheduler.fcfs_static_batch import FCFSStaticBatchScheduler
from scheduler.fcfs_nonbatch import FCFSNonBatchScheduler
from scheduler.fcfs_dyn_batch import FCFSDynamicBatchScheduler
from scheduler.fcfs_dyn_batch_predict import FCFSDynamicBatchPredictScheduler
from scheduler.fcfs_dyn_batch_predict_adj import FCFSDynamicBatchPredictAdjScheduler
from scheduler.fcfs_dyn_batch_predict_adj_avg import FCFSDynamicBatchPredictAdjAvgScheduler
from scheduler.sjf_nonbatch import SJFNonBatchScheduler
from scheduler.sjf_dyn_batch import SJFDynamicBatchScheduler
from scheduler.sjf_dyn_batch_predict import SJFDynamicBatchPredictScheduler
from scheduler.sjf_dyn_batch_predict_adj import SJFDynamicBatchPredictAdjScheduler
from scheduler.sjf_dyn_batch_predict_adj_avg import SJFDynamicBatchPredictAdjAvgScheduler
from scheduler.srpt_dyn_batch_predict import SRPTDynamicBatchPredictScheduler
from scheduler.srpt_dyn_batch_predict_adj import SRPTDynamicBatchPredictAdjScheduler
from scheduler.srpt_dyn_batch_predict_adj_avg import SRPTDynamicBatchPredictAdjAvgScheduler
from scheduler.bi_dyn_batch_predict_preemptive import BicriteriaDynamicBatchPredictPreemptiveScheduler
from scheduler.bi_dyn_batch_predict_non_preemptive import BicriteriaDynamicBatchPredictNonPreemptiveScheduler
from scheduler.bi_dyn_batch_predict_preemptive_adj import BicriteriaDynamicBatchPredictPreemptiveAdjScheduler
from scheduler.bi_dyn_batch_predict_preemptive_adj_avg import BicriteriaDynamicBatchPredictPreemptiveAdjAvgScheduler
from request import Request

RESULTS_PATH = './results'

SCHEDULER_DICT = {
    'fcfs_nonbatch': FCFSNonBatchScheduler,
    'fcfs_static_batch': FCFSStaticBatchScheduler,
    'fcfs_dynamic_batch': FCFSDynamicBatchScheduler,
    'fcfs_dynamic_batch_predict': FCFSDynamicBatchPredictScheduler,
    'fcfs_dynamic_batch_predict_adjustment': FCFSDynamicBatchPredictAdjScheduler,
    'fcfs_dynamic_batch_predict_adjustment_average': FCFSDynamicBatchPredictAdjAvgScheduler,
    'sjf_nonbatch': SJFNonBatchScheduler,
    'sjf_dynamic_batch': SJFDynamicBatchScheduler,
    'sjf_dynamic_batch_predict': SJFDynamicBatchPredictScheduler,
    'sjf_dynamic_batch_predict_adjustment': SJFDynamicBatchPredictAdjScheduler,
    'sjf_dynamic_batch_predict_adjustment_average': SJFDynamicBatchPredictAdjAvgScheduler,
    'srpt_dynamic_batch_predict': SRPTDynamicBatchPredictScheduler,
    'srpt_dynamic_batch_predict_adjustment': SRPTDynamicBatchPredictAdjScheduler,
    'srpt_dynamic_batch_predict_adjustment_average': SRPTDynamicBatchPredictAdjAvgScheduler,
    'bicriteria_dynamic_batch_predict_preemptive': BicriteriaDynamicBatchPredictPreemptiveScheduler,
    'bicriteria_dynamic_batch_predict_non_preemptive': BicriteriaDynamicBatchPredictNonPreemptiveScheduler,
    'bicriteria_dynamic_batch_predict_preemptive_adjustment': BicriteriaDynamicBatchPredictPreemptiveAdjScheduler,
    'bicriteria_dynamic_batch_predict_preemptive_adjustment_average': BicriteriaDynamicBatchPredictPreemptiveAdjAvgScheduler,
}

PROMPT_ENGINEERING_DEFAULT_DATA_PATH = './data/prompt_engineering_dataset.csv'
PROMPT_ENGINEERING_DEFAULT_MAX_DATA_SIZE = 5010  # max is 5010
PROMPT_ENGINEERING_DEFAULT_REQUEST_RATE = 1/3
PROMPT_ENGINEERING_DEFAULT_PREDICTED_LEN_STDEV = 0
PROMPT_ENGINEERING_DEFAULT_VRAM_SLOTS = 100

SHAREGPT_DEFAULT_DATA_PATH = './data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json'
SHAREGPT_DEFAULT_MAX_CONVERSATION_COUNT = 500  # max is 94145
SHAREGPT_DEFAULT_CONVERSATION_RATE = 1/100
SHAREGPT_DEFAULT_PROMPT_RATE = 1/20
SHAREGPT_DEFAULT_PREDICTED_LEN_STDEV = 0
SHAREGPT_DEFAULT_VRAM_SLOTS = 10000
SHAREGPT_DEFAULT_MAX_CONTEXT_WINDOW = 8192

DATASET_DEFAULT_PARAMETERS_LIST = [
    (
        "prompt_engineering",
        PromptEngineeringDatasetLoader(
            PROMPT_ENGINEERING_DEFAULT_DATA_PATH,
            PROMPT_ENGINEERING_DEFAULT_MAX_DATA_SIZE,
            PROMPT_ENGINEERING_DEFAULT_REQUEST_RATE,
            PROMPT_ENGINEERING_DEFAULT_PREDICTED_LEN_STDEV
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
            SHAREGPT_DEFAULT_MAX_CONTEXT_WINDOW
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

        plt.figure(figsize=(10, 6), dpi=200)

        x_name, x = next(iter(self.visualization_x_axis.items()))

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
        plt.legend()
        plt.grid(True, linewidth=0.5, alpha=0.6)

        plt.savefig(png_path, bbox_inches="tight")
        plt.close()

        print() # newline


# ============================================================================
# PROMPT ENGINEERING VARIED SLOT STDEV=0.3 EXPERIMENT
# ============================================================================
prompt_engineering_varied_slots = Experiment(
    "prompt_engineering_varied_slots",
    save_experiment_results=True,
    schedulers=[
        # 'fcfs_nonbatch',
        # 'fcfs_batch',
        # 'fcfs_dynamic_batch',
        # 'fcfs_dynamic_batch_predict',
        # 'fcfs_dynamic_batch_predict_adjustment',
        'fcfs_dynamic_batch_predict_adjustment_average',
        # # 'sjf_nonbatch',
        # 'sjf_dynamic_batch_predict',
        # 'sjf_dynamic_batch_predict_adjustment',
        'sjf_dynamic_batch_predict_adjustment_average',
        # 'srpt_dynamic_batch_predict',
        # 'srpt_dynamic_batch_predict_adjustment',
        'srpt_dynamic_batch_predict_adjustment_average',
        # 'bicriteria_dynamic_batch_predict_preemptive',
        # 'bicriteria_dynamic_batch_predict_non_preemptive',
        # 'bicriteria_dynamic_batch_predict_preemptive_adjustment',
        'bicriteria_dynamic_batch_predict_preemptive_adjustment_average',
    ],
    datasets=[],
    visualization_x_axis={"vram_slots": []}
)
for i in range(8):
    vram_slots = 100 + i*5
    prompt_engineering_varied_slots.datasets.append(
        (
            f"prompt_engineering_{vram_slots}_slots",
            PromptEngineeringDatasetLoader(
                PROMPT_ENGINEERING_DEFAULT_DATA_PATH,
                PROMPT_ENGINEERING_DEFAULT_MAX_DATA_SIZE,
                PROMPT_ENGINEERING_DEFAULT_REQUEST_RATE,
                0.3
            ),
            vram_slots
        )
    )
    prompt_engineering_varied_slots.visualization_x_axis["vram_slots"].append(vram_slots)


# ============================================================================
# SHAREGPT VARIED SLOTS STDEV=0.3 EXPERIMENT
# ============================================================================
sharegpt_varied_slots = Experiment(
    "sharegpt_varied_slots",
    save_experiment_results=True,
    schedulers=[
        # 'fcfs_nonbatch',
        # 'fcfs_batch',
        # 'fcfs_dynamic_batch',
        # 'fcfs_dynamic_batch_predict',
        # 'fcfs_dynamic_batch_predict_adjustment',
        'fcfs_dynamic_batch_predict_adjustment_average',
        # # 'sjf_nonbatch',
        # 'sjf_dynamic_batch_predict',
        # 'sjf_dynamic_batch_predict_adjustment',
        'sjf_dynamic_batch_predict_adjustment_average',
        # 'srpt_dynamic_batch_predict',
        # 'srpt_dynamic_batch_predict_adjustment',
        'srpt_dynamic_batch_predict_adjustment_average',
        # 'bicriteria_dynamic_batch_predict_preemptive',
        # 'bicriteria_dynamic_batch_predict_non_preemptive',
        # 'bicriteria_dynamic_batch_predict_preemptive_adjustment',
        'bicriteria_dynamic_batch_predict_preemptive_adjustment_average',
    ],
    datasets=[],
    visualization_x_axis={"vram_slots": []}
)
for i in range(8):
    vram_slots = 6000 + i * 250
    sharegpt_varied_slots.datasets.append(
        (
            f"sharegpt_{vram_slots}_slots",
            ShareGPTDatasetLoader(
                SHAREGPT_DEFAULT_DATA_PATH,
                SHAREGPT_DEFAULT_MAX_CONVERSATION_COUNT,
                SHAREGPT_DEFAULT_CONVERSATION_RATE,
                SHAREGPT_DEFAULT_PROMPT_RATE,
                0.3,
                SHAREGPT_DEFAULT_MAX_CONTEXT_WINDOW
            ),
            vram_slots
        )
    )
    sharegpt_varied_slots.visualization_x_axis["vram_slots"].append(vram_slots)


# ============================================================================
# PROMPT ENGINEERING VARIED SLOTS PERFECT PREDICTION EXPERIMENT
# ============================================================================
prompt_engineering_varied_slots_perfect = Experiment(
    "prompt_engineering_varied_slots_perfect",
    save_experiment_results=True,
    schedulers=[
        'sjf_dynamic_batch_predict',
        'srpt_dynamic_batch_predict',
        'bicriteria_dynamic_batch_predict_preemptive',
        'bicriteria_dynamic_batch_predict_non_preemptive',
    ],
    datasets=[],
    visualization_x_axis={"vram_slots": []}
)
for i in range(12):
    vram_slots = 50 + i*5
    prompt_engineering_varied_slots_perfect.datasets.append(
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
    prompt_engineering_varied_slots_perfect.visualization_x_axis["vram_slots"].append(vram_slots)


# ============================================================================
# SHAREGPT VARIED SLOTS PERFECT PREDICTION EXPERIMENT
# ============================================================================
sharegpt_varied_slots_perfect = Experiment(
    "sharegpt_varied_slots_perfect",
    save_experiment_results=True,
    schedulers=[
        'sjf_dynamic_batch_predict',
        'srpt_dynamic_batch_predict',
        'bicriteria_dynamic_batch_predict_preemptive',
        'bicriteria_dynamic_batch_predict_non_preemptive',
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
                SHAREGPT_DEFAULT_MAX_CONTEXT_WINDOW
            ),
            vram_slots
        )
    )
    sharegpt_varied_slots_perfect.visualization_x_axis["vram_slots"].append(vram_slots)


# ============================================================================
# PROMPT ENGINEERING VARIED SLOTS PERFECT PREDICTION FIXED PREFILL EXPERIMENT
# ============================================================================
prompt_engineering_varied_slots_perfect_fixed = Experiment(
    "prompt_engineering_varied_slots_perfect_fixed",
    save_experiment_results=True,
    schedulers=[
        'sjf_dynamic_batch_predict',
        'srpt_dynamic_batch_predict',
    ],
    datasets=[],
    visualization_x_axis={"vram_slots": []}
)
for i in range(12):
    vram_slots = 100 + i*5
    prompt_engineering_varied_slots_perfect_fixed.datasets.append(
        (
            f"prompt_engineering_{vram_slots}_slots",
            PromptEngineeringDatasetLoader(
                PROMPT_ENGINEERING_DEFAULT_DATA_PATH,
                PROMPT_ENGINEERING_DEFAULT_MAX_DATA_SIZE,
                PROMPT_ENGINEERING_DEFAULT_REQUEST_RATE,
                PROMPT_ENGINEERING_DEFAULT_PREDICTED_LEN_STDEV,
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
    "test_dataset",
    schedulers=[
        'sjf_dynamic_batch_predict',
        'srpt_dynamic_batch_predict',
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
    "prompt_engineering_varied_stdev",
    save_experiment_results=True,
    schedulers=[
        # 'fcfs_nonbatch',
        # 'fcfs_static_batch',
        # 'fcfs_dynamic_batch',
        # 'fcfs_dynamic_batch_predict',
        # 'fcfs_dynamic_batch_predict_adjustment',
        # 'fcfs_dynamic_batch_predict_adjustment_average',
        'sjf_dynamic_batch',
        'sjf_dynamic_batch_predict',
        'sjf_dynamic_batch_predict_adjustment',
        # 'srpt_dynamic_batch_predict',
        # 'srpt_dynamic_batch_predict_adjustment',
        # 'bicriteria_dynamic_batch_predict_preemptive',
        # 'bicriteria_dynamic_batch_predict_non_preemptive',
        # 'bicriteria_dynamic_batch_predict_preemptive_adjustment'
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
# DEFAULT EXPERIMENT
# ============================================================================
default = Experiment(
    "default",
    schedulers=[
        # 'fcfs_nonbatch',
        # 'fcfs_batch',
        # 'fcfs_dynamic_batch',
        # 'fcfs_dynamic_batch_predict',
        # 'fcfs_dynamic_batch_predict_adjustment',
        # 'fcfs_dynamic_batch_predict_adjustment_average',
        # # 'sjf_nonbatch',
        # 'sjf_dynamic_batch_predict',
        # 'sjf_dynamic_batch_predict_adjustment',
        # 'sjf_dynamic_batch_predict_adjustment_average',
        # 'srpt_dynamic_batch_predict',
        # 'srpt_dynamic_batch_predict_adjustment',
        # 'srpt_dynamic_batch_predict_adjustment_average',
        # 'bicriteria_dynamic_batch_predict_preemptive',
        # 'bicriteria_dynamic_batch_predict_non_preemptive',
        # 'bicriteria_dynamic_batch_predict_preemptive_adjustment',
        # 'bicriteria_dynamic_batch_predict_preemptive_adjustment_average',
    ],
    datasets=[
        (
            "prompt_engineering",
            PromptEngineeringDatasetLoader(
                PROMPT_ENGINEERING_DEFAULT_DATA_PATH,
                PROMPT_ENGINEERING_DEFAULT_MAX_DATA_SIZE,
                PROMPT_ENGINEERING_DEFAULT_REQUEST_RATE,
                PROMPT_ENGINEERING_DEFAULT_PREDICTED_LEN_STDEV
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
        #         SHAREGPT_DEFAULT_MAX_CONTEXT_WINDOW
        #     ),
        #     SHAREGPT_DEFAULT_VRAM_SLOTS
        # )
    ],
    visualization_x_axis=None
)

EXPERIMENTS = [
    default,
    # prompt_engineering_varied_slots_perfect_fixed,
    # prompt_engineering_varied_slots_perfect,
    # sharegpt_varied_slots_perfect,
    # prompt_engineering_varied_slots,
    sharegpt_varied_slots,
    # prompt_engineering_varied_stdev,
    # test_dataset,
]
