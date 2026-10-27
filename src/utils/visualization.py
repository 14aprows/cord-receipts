from pathlib import Path 
import matplotlib.pyplot as plt
import pandas as pd

def plot_training(csv_path, output_path):
    history = pd.read_csv(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12,4))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["val_token_accuracy"] * 100)
    axes[1].set_title("Validation token accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)