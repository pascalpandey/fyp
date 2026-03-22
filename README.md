# FYP

copy `https://www.kaggle.com/datasets/antrixsh/prompt-engineering-and-responses-dataset` to `./data`

to setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

to run

```bash
# first, uncomment the specific experiment from the EXPERIMENTS variable on experiment.py
python3 main.py
```

to test

```bash
PYTHONPATH=. pytest -s
```

to visualize dataset

```bash
python -m scripts.visualize_dataset
```

to generate graph from JSON

```bash
# first, modify the `json_path` and `output_dir` variables
python -m scripts.generate_graph
```