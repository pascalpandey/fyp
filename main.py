import numpy as np
from loader.prompt_engineering_dataset import PromptEngineeringDatasetLoader
from scheduler.fcfs import FCFSScheduler
from simulator import Simulator
from gpu import GPU, GPUView

np.random.seed(42)

DATA_PATH = './data/prompt_engineering_dataset.csv'
DATA_SIZE = 1000
REQUEST_RATE = 1/50
VRAM_SLOTS = 1000


def main():
    loader = PromptEngineeringDatasetLoader(DATA_PATH, DATA_SIZE, REQUEST_RATE)
    dataset = loader.load()

    gpu = GPU(VRAM_SLOTS)

    scheduler = FCFSScheduler(GPUView(gpu))

    simulator = Simulator(dataset, gpu, scheduler)
    simulator.run()

    dataset.show_results()

    # TODO: visualize vram usage
    # gpu.visualize_history()


if __name__ == "__main__":
    main()
