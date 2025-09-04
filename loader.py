import csv
from dataset import Data, Dataset


class Loader:
    def __init__(self, path, size):
        self.path = path
        self.size = size

    def load(self):
        dataset = Dataset()
        with open(self.path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                if i >= self.size:
                    break
                data = Data(row['Prompt'], row['Response'])
                dataset.add(data)
        return dataset
