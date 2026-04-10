import argparse
import re
from pathlib import Path

import pandas as pd


def clean_text(text: str) -> str:
    """Basic tweet text cleaning."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # remove URLs
    text = re.sub(r"\@\w+|\#", "", text)  # remove mentions and hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # keep letters and spaces only
    text = re.sub(r"\s+", " ", text).strip()  # normalize whitespace
    return text


def load_and_sample_data(
    input_path: Path,
    sample_per_class: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load Sentiment140 CSV and create balanced sample."""
    cols = ["target", "ids", "date", "flag", "user", "text"]
    df = pd.read_csv(input_path, encoding="latin-1", names=cols)

    print("Original dataset shape:", df.shape)
    print("Original target distribution:")
    print(df["target"].value_counts())

    df_neg = df[df["target"] == 0].sample(n=sample_per_class, random_state=random_state)
    df_pos = df[df["target"] == 4].sample(n=sample_per_class, random_state=random_state)
    df_sample = pd.concat([df_neg, df_pos], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)

    print("\nSampled dataset shape:", df_sample.shape)
    print("Sampled target distribution:")
    print(df_sample["target"].value_counts())

    return df_sample


def preprocess(df_sample: pd.DataFrame) -> pd.DataFrame:
    """Run preprocessing and return final dataframe."""
    print("\nChecking null values:")
    print(df_sample.isnull().sum())

    # Map labels: 0 -> 0 (negative), 4 -> 1 (positive)
    df_sample["sentiment"] = df_sample["target"].map({0: 0, 4: 1})

    print("\nCleaning text...")
    df_sample["clean_text"] = df_sample["text"].apply(clean_text)

    # Drop rows where cleaned text is empty
    before = len(df_sample)
    df_sample = df_sample[df_sample["clean_text"] != ""].copy()
    after = len(df_sample)

    print(f"Rows before dropping empty clean_text: {before}")
    print(f"Rows after dropping empty clean_text:  {after}")

    final_df = df_sample[["clean_text", "sentiment"]]
    return final_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Sentiment140 dataset into cleaned balanced CSV."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/training.1600000.processed.noemoticon.csv",
        help="Path ke file CSV mentah Sentiment140.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/cleaned/cleaned_sample.csv",
        help="Path output CSV hasil preprocessing.",
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=5000,
        help="Jumlah sampel per kelas (negatif & positif).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed random untuk reproduksibilitas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file tidak ditemukan: {input_path}\n"
            "Pastikan dataset mentah tersedia di path yang benar."
        )

    print("Loading dataset...")
    df_sample = load_and_sample_data(
        input_path=input_path,
        sample_per_class=args.sample_per_class,
        random_state=args.random_state,
    )

    final_df = preprocess(df_sample)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print("\nPreprocessing selesai.")
    print(f"Saved cleaned data to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
