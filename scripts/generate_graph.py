import os
import json
import matplotlib.pyplot as plt

def generate_graph(json_path, output_dir):
    with open(json_path, "r") as f:
        snapshot = json.load(f)

    experiment_name = snapshot["experiment_name"]
    x_name = snapshot["x_name"]
    x = snapshot["x_values"]
    schedulers = snapshot["schedulers"]

    if output_dir is None:
        output_dir = os.path.dirname(json_path)

    os.makedirs(output_dir, exist_ok=True)

    png_path = os.path.join(output_dir, "regenerated_result.png")

    plt.figure(figsize=(10, 6), dpi=200)

    for scheduler_name, y in schedulers.items():
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

    plt.xlabel(x_name)
    plt.ylabel("Latency (time units)")
    plt.title(experiment_name)
    plt.legend(fontsize=7)
    plt.grid(True, linewidth=0.5, alpha=0.6)

    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

    print(f"Saved regenerated graph to {png_path}")

if __name__ == "__main__":
    json_path = 'results/Prompt Engineering Varied VRAM Capacity/result_data.json'
    output_dir = 'results/Prompt Engineering Varied VRAM Capacity'
    generate_graph(json_path, output_dir)