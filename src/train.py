from data import load_eurosat, build_tf_datasets
from model import build_cnn
from utils import plot_training
import os


def main():
    os.makedirs("../experiments/results", exist_ok=True)

    ds, class_names, num_classes = load_eurosat()
    train_ds, val_ds, _ = build_tf_datasets(ds)

    model = build_cnn(num_classes=num_classes)
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15
    )

    plot_training(
        history,
        save_path="../experiments/results/training_curves.png"
    )

    model.save("../experiments/results/cnn_eurosat_model")


if __name__ == "__main__":
    main()
