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


def visualize_dataset(dataset, filename):
    actual_values = np.array(
        [req._response_len for req in dataset._requests.values()]
    )
    predicted_values = np.array(
        [req.predicted_response_len for req in dataset._requests.values()]
    )
    prompt_len_values = np.array(
        [req._prompt_len for req in dataset._requests.values()]
    )

    diff_values = predicted_values - actual_values

    output_dir = "./results/visualize"
    os.makedirs(output_dir, exist_ok=True)

    actual_vs_pred_path = os.path.join(
        output_dir, f"{filename}_actual_vs_predicted.png"
    )

    plt.figure(figsize=(10, 6), dpi=200)

    plt.hist(actual_values, alpha=0.4, bins=50, label="Actual Response Length")
    plt.hist(predicted_values, alpha=0.4, bins=50, label="Predicted Response Length")

    actual_mean = np.mean(actual_values)
    actual_var = np.var(actual_values)

    pred_mean = np.mean(predicted_values)
    pred_var = np.var(predicted_values)

    plt.axvline(actual_mean, linestyle="--", linewidth=1,
                label=f"Actual Mean={actual_mean:.2f}, Var={actual_var:.2f}")
    plt.axvline(pred_mean, linestyle="--", linewidth=1,
                label=f"Pred Mean={pred_mean:.2f}, Var={pred_var:.2f}")

    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    plt.title("Actual vs Predicted Response Length Distribution")
    plt.legend(fontsize=8)

    plt.savefig(actual_vs_pred_path, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to {actual_vs_pred_path}")

    prompt_len_path = os.path.join(
        output_dir, f"{filename}_prompt_length.png"
    )

    plt.figure(figsize=(10, 6), dpi=200)

    plt.hist(prompt_len_values, alpha=0.7, bins=50, label="Prompt Length")

    prompt_mean = np.mean(prompt_len_values)
    prompt_var = np.var(prompt_len_values)

    plt.axvline(prompt_mean, linestyle="--", linewidth=1,
                label=f"Mean={prompt_mean:.2f}, Var={prompt_var:.2f}")

    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    plt.title("Prompt Length Distribution")
    plt.legend(fontsize=8)

    plt.savefig(prompt_len_path, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to {prompt_len_path}")

    diff_path = os.path.join(
        output_dir, f"{filename}_prediction_difference.png"
    )

    plt.figure(figsize=(10, 6), dpi=200)

    plt.hist(diff_values, alpha=0.7, bins=50,
             label="Prediction Error (Predicted - Actual)")

    diff_mean = np.mean(diff_values)
    diff_var = np.var(diff_values)

    plt.axvline(diff_mean, linestyle="--", linewidth=1,
                label=f"Mean={diff_mean:.2f}, Var={diff_var:.2f}")

    plt.xlabel("Token Difference")
    plt.ylabel("Frequency")
    plt.title("Prediction Error Distribution, STDEV = 0.1")
    plt.legend(fontsize=8)

    plt.savefig(diff_path, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to {diff_path}")


if __name__ == "__main__":
    datasets = [
        (
            PromptEngineeringDatasetLoader(
                PROMPT_ENGINEERING_DEFAULT_DATA_PATH,
                PROMPT_ENGINEERING_DEFAULT_MAX_DATA_SIZE,
                PROMPT_ENGINEERING_DEFAULT_REQUEST_RATE,
                PROMPT_ENGINEERING_DEFAULT_PREDICTED_LEN_STDEV
            ), 'prompt_engineering'
        ),
        (
            ShareGPTDatasetLoader(
                SHAREGPT_DEFAULT_DATA_PATH,
                SHAREGPT_DEFAULT_MAX_CONVERSATION_COUNT,
                SHAREGPT_DEFAULT_CONVERSATION_RATE,
                SHAREGPT_DEFAULT_PROMPT_RATE,
                SHAREGPT_DEFAULT_PREDICTED_LEN_STDEV,
                SHAREGPT_DEFAULT_MAX_CONVERSATION_TOKEN_COUNT
            ), 'sharegpt'
        )
    ]

    for dataset, dataset_name in datasets:
        visualize_dataset(dataset.load(), dataset_name)
