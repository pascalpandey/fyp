from loader.prompt_engineering_dataset import PromptEngineeringDatasetLoader
from scheduler.fcfs import FCFSScheduler
from simulator import Simulator
from resources import Resources


DATA_PATH = './data/prompt_engineering_dataset.csv'
DATA_SIZE = 1000
REQUEST_RATE = 1/50
GPU_SLOTS = 1000
VRAM_SLOTS = 1000


def main():
    loader = PromptEngineeringDatasetLoader(DATA_PATH, DATA_SIZE, REQUEST_RATE)
    dataset = loader.load()

    scheduler = FCFSScheduler()
    resources = Resources(GPU_SLOTS, VRAM_SLOTS)

    simulator = Simulator(dataset, scheduler, resources)
    simulator.run()

    dataset.show_results()
    resources.visualize_history()


if __name__ == "__main__":
    main()
