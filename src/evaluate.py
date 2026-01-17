from data import load_eurosat, build_tf_datasets
import tensorflow as tf


def main():
    ds, class_names, _ = load_eurosat()
    _, _, test_ds = build_tf_datasets(ds)

    model = tf.keras.models.load_model(
        "../experiments/results/cnn_eurosat_model"
    )

    loss, acc = model.evaluate(test_ds)
    print(f"Test Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
