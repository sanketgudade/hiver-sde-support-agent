
import pandas as pd


# ============================================================
# PATH
# ============================================================

DATA_PATH = r"data/processed/customer_support_pairs.csv"


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

def load_data():
    """Load the processed customer-support pairs."""

    print("Loading processed dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


# ============================================================
# DATASET OVERVIEW
# ============================================================

def dataset_overview(df):
    """Display basic information about the processed dataset."""

    print("\n" + "=" * 60)
    print("PROCESSED DATASET OVERVIEW")
    print("=" * 60)

    print(f"\nTotal pairs: {len(df):,}")
    print(f"Unique customer tweets: {df['customer_tweet_id'].nunique():,}")
    print(f"Unique support tweets: {df['support_tweet_id'].nunique():,}")

    print("\nColumns:")
    for column in df.columns:
        print(f"- {column}")


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

def missing_value_analysis(df):
    """Check missing values in the processed dataset."""

    print("\n" + "=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)

    missing = df.isnull().sum()

    print(missing.to_string())

    total_missing = missing.sum()

    print(
        f"\nTotal missing values: "
        f"{total_missing:,}"
    )


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

def duplicate_analysis(df):
    """Check duplicate customer-support pairs."""

    print("\n" + "=" * 60)
    print("DUPLICATE ANALYSIS")
    print("=" * 60)

    duplicate_pairs = df[
        [
            "customer_tweet_id",
            "support_tweet_id"
        ]
    ].duplicated().sum()

    duplicate_customer_text = df[
        "customer_text"
    ].duplicated().sum()

    duplicate_support_text = df[
        "support_text"
    ].duplicated().sum()

    print(
        f"\nDuplicate customer-support pairs: "
        f"{duplicate_pairs:,}"
    )

    print(
        f"Duplicate customer messages: "
        f"{duplicate_customer_text:,}"
    )

    print(
        f"Duplicate support responses: "
        f"{duplicate_support_text:,}"
    )


# ============================================================
# TEXT LENGTH ANALYSIS
# ============================================================

def text_length_analysis(df):
    """Analyze customer and support message lengths."""

    print("\n" + "=" * 60)
    print("TEXT LENGTH ANALYSIS")
    print("=" * 60)

    # Character counts
    df["customer_char_count"] = (
        df["customer_text"].str.len()
    )

    df["support_char_count"] = (
        df["support_text"].str.len()
    )

    # Word counts
    df["customer_word_count"] = (
        df["customer_text"]
        .str.split()
        .str.len()
    )

    df["support_word_count"] = (
        df["support_text"]
        .str.split()
        .str.len()
    )

    print("\nCustomer message statistics:")
    print(
        df[
            [
                "customer_char_count",
                "customer_word_count"
            ]
        ]
        .describe()
        .round(2)
    )

    print("\nSupport response statistics:")
    print(
        df[
            [
                "support_char_count",
                "support_word_count"
            ]
        ]
        .describe()
        .round(2)
    )


# ============================================================
# MULTIPLE RESPONSE ANALYSIS
# ============================================================

def multiple_response_analysis(df):
    """Analyze customers having multiple support responses."""

    print("\n" + "=" * 60)
    print("MULTIPLE RESPONSE ANALYSIS")
    print("=" * 60)

    responses_per_customer = (
        df.groupby("customer_tweet_id")
        .size()
    )

    print(
        "\nAverage support responses per "
        "customer tweet: "
        f"{responses_per_customer.mean():.2f}"
    )

    print(
        "Maximum support responses for "
        "one customer tweet: "
        f"{responses_per_customer.max()}"
    )

    customers_with_multiple_responses = (
        (responses_per_customer > 1).sum()
    )

    print(
        "Customer tweets with multiple "
        "support responses: "
        f"{customers_with_multiple_responses:,}"
    )


# ============================================================
# SAMPLE PAIRS
# ============================================================

def show_sample_pairs(df):
    """Display sample customer-support pairs."""

    print("\n" + "=" * 60)
    print("SAMPLE CUSTOMER → SUPPORT PAIRS")
    print("=" * 60)

    for _, row in df.head(10).iterrows():

        print("\nCustomer:")
        print(row["customer_text"])

        print("\nSupport:")
        print(row["support_text"])

        print("-" * 60)


# ============================================================
# FINAL SUMMARY
# ============================================================

def final_summary(df):
    """Display final analysis summary."""

    print("\n" + "=" * 60)
    print("FINAL PROCESSED DATASET SUMMARY")
    print("=" * 60)

    print(f"\nTotal training pairs: {len(df):,}")

    print(
        f"Unique customer tweets: "
        f"{df['customer_tweet_id'].nunique():,}"
    )

    print(
        f"Unique support tweets: "
        f"{df['support_tweet_id'].nunique():,}"
    )

    print(
        f"Duplicate pairs: "
        f"{df[['customer_tweet_id', 'support_tweet_id']].duplicated().sum():,}"
    )

    print(
        f"Missing values: "
        f"{df.isnull().sum().sum():,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    dataset_overview(df)

    missing_value_analysis(df)

    duplicate_analysis(df)

    text_length_analysis(df)

    multiple_response_analysis(df)

    show_sample_pairs(df)

    final_summary(df)

    print("\n" + "=" * 60)
    print("PROCESSED DATASET EDA COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
