import numpy as np
import copy
from simulator import Simulator
from gpu import GPU, GPUView
from experiment import EXPERIMENTS, RESULTS_PATH

np.random.seed(42)

def main():
    for experiment in EXPERIMENTS:
        print(f"{experiment.experiment_name} experiment")
        for dataset_name, loader, vram_slots in experiment.datasets:
            print(f"{dataset_name} dataset")
            dataset = loader.load()

            for scheduler_name, scheduler in experiment.schedulers:
                dataset_copy = copy.deepcopy(dataset)

                gpu = GPU(vram_slots)

                scheduler = scheduler(GPUView(gpu))

                simulator = Simulator(dataset_copy, gpu, scheduler)
                simulator.run()

                average_latency = dataset_copy.show_average_latency(scheduler_name)
                experiment.add_result(scheduler_name, average_latency)

                if experiment.save_timeline_visualizations:
                    dataset_copy.visualize_request_history(RESULTS_PATH, experiment.experiment_name, scheduler_name, dataset_name)
                    gpu.visualize_history(RESULTS_PATH, experiment.experiment_name, scheduler_name, dataset_name)

            print() # newline
        
        experiment.visualize_results()

if __name__ == "__main__":
    main()
