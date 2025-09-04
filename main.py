from loader import Loader
from runner import Runner
from visualizer import Visualizer

def main():
	loader = Loader('./data/prompt_engineering_dataset.csv', 1000)
	dataset = loader.load()

	runner = Runner(dataset)
	result = runner.run()

	visulizer = Visualizer(result)
	visulizer.show()

if __name__ == "__main__":
	main()