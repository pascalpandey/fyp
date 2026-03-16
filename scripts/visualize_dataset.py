import matplotlib.pyplot as plt
import os
import numpy as np
from loader.prompt_engineering_dataset import PromptEngineeringDatasetLoader
from loader.sharegpt_dataset import ShareGPTDatasetLoader
from experiment import (
    PROMPT_ENGINEERING_DEFAULT_DATA_PATH,
    PROMPT_ENGINEERING_DEFAULT_MAX_DATA_SIZE,
    PROMPT_ENGINEERING_DEFAULT_REQUEST_RATE,
    PROMPT_ENGINEERING_DEFAULT_PREDICTED_LEN_STDEV,
    SHAREGPT_DEFAULT_DATA_PATH,
    SHAREGPT_DEFAULT_MAX_CONVERSATION_COUNT,
    SHAREGPT_DEFAULT_CONVERSATION_RATE,
    SHAREGPT_DEFAULT_PROMPT_RATE,
    SHAREGPT_DEFAULT_PREDICTED_LEN_STDEV,
    SHAREGPT_DEFAULT_MAX_CONVERSATION_TOKEN_COUNT
)


def stats(arr):
    return np.mean(arr), np.var(arr), np.min(arr), np.max(arr)


def visualize_dataset(dataset, dataset_name):
    actual_values = np.array([req._response_len for req in dataset._requests.values()])
    predicted_values = np.array([req.predicted_response_len for req in dataset._requests.values()])
    prompt_len_values = np.array([req._prompt_len for req in dataset._requests.values()])
    diff_values = predicted_values - actual_values

    output_dir = "./results/visualize"
    os.makedirs(output_dir, exist_ok=True)

    actual_mean, actual_var, actual_min, actual_max = stats(actual_values)
    pred_mean, pred_var, pred_min, pred_max = stats(predicted_values)

    path = os.path.join(output_dir, f"{dataset_name}_actual_vs_predicted.png")

    plt.figure(figsize=(10, 6), dpi=200)

    plt.hist(actual_values, alpha=0.5, bins=50,
             label="Actual Response Length")
    plt.hist(predicted_values, alpha=0.5, bins=50,
             label="Predicted Response Length")

    plt.axvline(actual_mean, linestyle="--", linewidth=1,
                label=f"Actual Mean={actual_mean:.2f}, Actual Var={actual_var:.2f}")
    plt.axvline(pred_mean, linestyle="--", color="orange", linewidth=1,
                label=f"Pred Mean={pred_mean:.2f}, Actual Var={pred_var:.2f}")

    plt.axvline(actual_min, linestyle=":", linewidth=1,
                label=f"Actual Min={actual_min}")
    plt.axvline(actual_max, linestyle=":", linewidth=1,
                label=f"Actual Max={actual_max}")

    plt.axvline(pred_min, linestyle=":", color="orange", linewidth=1,
                label=f"Pred Min={pred_min}")
    plt.axvline(pred_max, linestyle=":", color="orange", linewidth=1,
                label=f"Pred Max={pred_max}")

    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    plt.title(f"{dataset_name} Actual vs Predicted Response Length Distribution")
    plt.legend(fontsize=8)

    plt.savefig(path, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to {path}")

    prompt_mean, prompt_var, prompt_min, prompt_max = stats(prompt_len_values)

    path = os.path.join(output_dir, f"{dataset_name}_prompt_length.png")

    plt.figure(figsize=(10, 6), dpi=200)

    plt.hist(prompt_len_values, alpha=0.7, bins=50,
             label="Prompt Length")

    plt.axvline(prompt_mean, linestyle="--", linewidth=1,
                label=f"Mean={prompt_mean:.2f}, Var={prompt_var:.2f}")
    plt.axvline(prompt_min, linestyle=":", linewidth=1,
                label=f"Min={prompt_min}")
    plt.axvline(prompt_max, linestyle=":", linewidth=1,
                label=f"Max={prompt_max}")

    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    plt.title(f"{dataset_name} Prompt Length Distribution")
    plt.legend(fontsize=8)

    plt.savefig(path, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to {path}")

    diff_mean, diff_var, diff_min, diff_max = stats(diff_values)

    path = os.path.join(output_dir, f"{dataset_name}_prediction_difference.png")

    plt.figure(figsize=(10, 6), dpi=200)

    plt.hist(diff_values, alpha=0.7, bins=50,
             label="Prediction Error (Predicted - Actual)")

    plt.axvline(diff_mean, linestyle="--", linewidth=1,
                label=f"Mean={diff_mean:.2f}, Var={diff_var:.2f}")
    plt.axvline(diff_min, linestyle=":", linewidth=1,
                label=f"Min={diff_min:.2f}")
    plt.axvline(diff_max, linestyle=":", linewidth=1,
                label=f"Max={diff_max:.2f}")

    plt.xlabel("Token Difference")
    plt.ylabel("Frequency")
    plt.title(f"{dataset_name} Prediction Error Distribution")
    plt.legend(fontsize=8)

    plt.savefig(path, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to {path}")


if __name__ == "__main__":
    datasets = [
        (
            PromptEngineeringDatasetLoader(
                PROMPT_ENGINEERING_DEFAULT_DATA_PATH,
                PROMPT_ENGINEERING_DEFAULT_MAX_DATA_SIZE,
                PROMPT_ENGINEERING_DEFAULT_REQUEST_RATE,
                PROMPT_ENGINEERING_DEFAULT_PREDICTED_LEN_STDEV
            ), 'Prompt Engineering'
        ),
        (
            ShareGPTDatasetLoader(
                SHAREGPT_DEFAULT_DATA_PATH,
                SHAREGPT_DEFAULT_MAX_CONVERSATION_COUNT,
                SHAREGPT_DEFAULT_CONVERSATION_RATE,
                SHAREGPT_DEFAULT_PROMPT_RATE,
                SHAREGPT_DEFAULT_PREDICTED_LEN_STDEV,
                SHAREGPT_DEFAULT_MAX_CONVERSATION_TOKEN_COUNT
            ), 'ShareGPT'
        )
    ]

    for dataset, dataset_name in datasets:
        visualize_dataset(dataset.load(), dataset_name)
