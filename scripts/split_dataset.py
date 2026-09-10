
import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

DATA_PATH = r"data/processed/customer_support_pairs.csv"

OUTPUT_DIR = r"data/processed/splits"

TRAIN_PATH = os.path.join(
    OUTPUT_DIR,
    "train.csv"
)

VALIDATION_PATH = os.path.join(
    OUTPUT_DIR,
    "validation.csv"
)

TEST_PATH = os.path.join(
    OUTPUT_DIR,
    "test.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TRAIN_SIZE = 0.80
VALIDATION_SIZE = 0.10
TEST_SIZE = 0.10


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load the processed customer-support pairs."""

    print("Loading processed dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"Total pairs: {len(df):,}")

    return df


# ============================================================
# SPLIT BY CUSTOMER TWEET
# ============================================================

def split_dataset(df):
    """
    Split the dataset using unique customer tweet IDs.

    This prevents pairs belonging to the same customer tweet
    from appearing in different datasets.
    """

    print("\n" + "=" * 60)
    print("CREATING TRAIN / VALIDATION / TEST SPLITS")
    print("=" * 60)

    unique_customers = df[
        "customer_tweet_id" # sanket we are spliting by c_t_i to avoid ambuiguity 
    ].unique()

    print(
        f"\nUnique customer tweets: "
        f"{len(unique_customers):,}"
    )

    # --------------------------------------------------------
    # First split:
    # 80% Train
    # 20% Temporary
    # --------------------------------------------------------

    train_customers, temp_customers = train_test_split(
        unique_customers,
        test_size=(VALIDATION_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Second split:
    # 10% Validation
    # 10% Test
    # --------------------------------------------------------

    validation_customers, test_customers = train_test_split(
        temp_customers,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_df = df[
        df["customer_tweet_id"].isin(train_customers)
    ].copy()

    validation_df = df[
        df["customer_tweet_id"].isin(validation_customers)
    ].copy()

    test_df = df[
        df["customer_tweet_id"].isin(test_customers)
    ].copy()

    return train_df, validation_df, test_df


# ============================================================
# VALIDATE SPLITS
# ============================================================

def validate_splits(
    train_df,
    validation_df,
    test_df
):
    """Validate that there is no customer-level data leakage."""

    print("\n" + "=" * 60)
    print("SPLIT VALIDATION")
    print("=" * 60)

    train_customers = set(
        train_df["customer_tweet_id"]
    )

    validation_customers = set(
        validation_df["customer_tweet_id"]
    )

    test_customers = set(
        test_df["customer_tweet_id"]
    )

    train_validation_overlap = (
        train_customers & validation_customers
    )

    train_test_overlap = (
        train_customers & test_customers
    )

    validation_test_overlap = (
        validation_customers & test_customers
    )

    print(
        f"\nTrain ↔ Validation overlap: "
        f"{len(train_validation_overlap):,}"
    )

    print(
        f"Train ↔ Test overlap: "
        f"{len(train_test_overlap):,}"
    )

    print(
        f"Validation ↔ Test overlap: "
        f"{len(validation_test_overlap):,}"
    )

    if (
        len(train_validation_overlap) == 0
        and len(train_test_overlap) == 0
        and len(validation_test_overlap) == 0
    ):
        print(
            "\n✓ No customer-level data leakage detected."
        )
    else:
        print(
            "\n⚠ Data leakage detected!"
        )


# ============================================================
# DISPLAY SPLIT STATISTICS
# ============================================================

def display_statistics(
    train_df,
    validation_df,
    test_df
):
    """Display statistics for each split."""

    total_pairs = len(
        train_df
    ) + len(
        validation_df
    ) + len(
        test_df
    )

    print("\n" + "=" * 60)
    print("SPLIT STATISTICS")
    print("=" * 60)

    datasets = {
        "Train": train_df,
        "Validation": validation_df,
        "Test": test_df
    }

    for name, dataset in datasets.items():

        percentage = (
            len(dataset) / total_pairs
        ) * 100

        unique_customers = (
            dataset["customer_tweet_id"]
            .nunique()
        )

        print(f"\n{name}:")

        print(
            f"  Pairs: "
            f"{len(dataset):,}"
        )

        print(
            f"  Unique customer tweets: "
            f"{unique_customers:,}"
        )

        print(
            f"  Percentage of pairs: "
            f"{percentage:.2f}%"
        )


# ============================================================
# SAVE SPLITS
# ============================================================

def save_splits(
    train_df,
    validation_df,
    test_df
):
    """Save the three dataset splits."""

    print("\n" + "=" * 60)
    print("SAVING DATASET SPLITS")
    print("=" * 60)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    train_df.to_csv(
        TRAIN_PATH,
        index=False
    )

    validation_df.to_csv(
        VALIDATION_PATH,
        index=False
    )

    test_df.to_csv(
        TEST_PATH,
        index=False
    )

    print(
        f"\nTrain dataset saved to:\n"
        f"{TRAIN_PATH}"
    )

    print(
        f"\nValidation dataset saved to:\n"
        f"{VALIDATION_PATH}"
    )

    print(
        f"\nTest dataset saved to:\n"
        f"{TEST_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # Step 1: Load processed pairs
    df = load_data()

    # Step 2: Split by unique customer tweet
    (
        train_df,
        validation_df,
        test_df
    ) = split_dataset(df)

    # Step 3: Validate no leakage
    validate_splits(
        train_df,
        validation_df,
        test_df
    )

    # Step 4: Display statistics
    display_statistics(
        train_df,
        validation_df,
        test_df
    )

    # Step 5: Save datasets
    save_splits(
        train_df,
        validation_df,
        test_df
    )

    print("\n" + "=" * 60)
    print("DATASET SPLITTING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
