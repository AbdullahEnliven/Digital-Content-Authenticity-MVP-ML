import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from app.ml.text_auth.config import get_text_auth_settings
from app.ml.text_auth.embedder import MiniLMEmbedder
from app.ml.text_auth.preprocess import preprocess_text
from app.ml.text_auth.utils import iter_text_files, read_text_file


def load_dataset(dataset_dir: Path) -> tuple[list[str], list[str]]:
    """Read and preprocess dataset text files."""
    settings = get_text_auth_settings()
    samples = iter_text_files(dataset_dir)
    if not samples:
        raise ValueError(f"No .txt files found under {dataset_dir}")

    texts: list[str] = []
    labels: list[str] = []
    skipped = 0

    for index, (path, label) in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] {label}: {path}")
        try:
            text = preprocess_text(read_text_file(path), settings)
        except ValueError as exc:
            skipped += 1
            print(f"  skipped: {exc}")
            continue
        texts.append(text)
        labels.append(label)

    if not texts:
        raise ValueError("No usable text samples found after preprocessing")
    if skipped:
        print(f"\nSkipped {skipped} text files during preprocessing.")

    return texts, labels


def train_classifier(dataset_dir: Path, output_dir: Path, test_size: float, random_state: int) -> None:
    """Train and save a Logistic Regression classifier for MiniLM embeddings."""
    settings = get_text_auth_settings()
    texts, labels = load_dataset(dataset_dir)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    class_counts = np.bincount(encoded_labels)
    stratify = encoded_labels if len(class_counts) > 1 and np.min(class_counts) >= 2 else None

    x_train_texts, x_val_texts, y_train, y_val = train_test_split(
        texts,
        encoded_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    embedder = MiniLMEmbedder(settings)
    print("\nEncoding training texts...")
    x_train = embedder.encode_texts(x_train_texts)
    print("Encoding validation texts...")
    x_val = embedder.encode_texts(x_val_texts)

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=random_state,
    )
    classifier.fit(x_train, y_train)

    predictions = classifier.predict(x_val)
    accuracy = accuracy_score(y_val, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print("\nValidation metrics")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification report")
    all_label_ids = np.arange(len(label_encoder.classes_))
    print(
        classification_report(
            y_val,
            predictions,
            labels=all_label_ids,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )
    print("Confusion matrix")
    print(confusion_matrix(y_val, predictions, labels=all_label_ids))

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, output_dir / "classifier.pkl")
    joblib.dump(label_encoder, output_dir / "label_encoder.pkl")
    print(f"\nSaved classifier to {output_dir / 'classifier.pkl'}")
    print(f"Saved label encoder to {output_dir / 'label_encoder.pkl'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the text-auth Logistic Regression classifier.")
    parser.add_argument("--dataset", required=True, type=Path, help="Dataset directory containing human/ and ai/")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("app/ml/text_auth/saved_models"),
        help="Directory where classifier.pkl and label_encoder.pkl will be written",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split size")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_classifier(
        dataset_dir=args.dataset,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
