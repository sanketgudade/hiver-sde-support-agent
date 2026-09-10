import pandas as pd


DATA_PATH = r"data/raw/archive/twcs/twcs.csv"


def load_data():
    """Load the complete Twitter customer-support dataset."""
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")

    return df


def duplicate_analysis(df):
    """Check for duplicate tweet IDs and duplicate messages."""

    print("\n" + "=" * 60)
    print("DUPLICATE ANALYSIS")
    print("=" * 60)

    # Duplicate tweet IDs
    duplicate_tweet_ids = df["tweet_id"].duplicated().sum()

    print(f"\nDuplicate tweet IDs: {duplicate_tweet_ids:,}")

    # Duplicate complete rows
    duplicate_rows = df.duplicated().sum()

    print(f"Duplicate complete rows: {duplicate_rows:,}")

    # Duplicate text
    duplicate_text = df["text"].duplicated().sum()

    print(f"Duplicate text entries: {duplicate_text:,}")


def text_quality_analysis(df):
    """Check missing, empty, and very short messages."""

    print("\n" + "=" * 60)
    print("TEXT QUALITY ANALYSIS")
    print("=" * 60)

    # Missing text
    missing_text = df["text"].isna().sum()

    print(f"\nMissing text: {missing_text:,}")

    # Empty text
    empty_text = (
        df["text"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"Empty text: {empty_text:,}")

    # Very short text
    text_length = (
        df["text"]
        .fillna("")
        .str.strip()
        .str.len()
    )

    very_short_text = (text_length < 3).sum()

    print(f"Very short text (< 3 characters): {very_short_text:,}")

    # Extremely long text
    very_long_text = (text_length > 500).sum()

    print(f"Very long text (> 500 characters): {very_long_text:,}")


def relationship_quality_analysis(df):
    """Check the quality of tweet-to-tweet relationships."""

    print("\n" + "=" * 60)
    print("RELATIONSHIP QUALITY ANALYSIS")
    print("=" * 60)

    # Tweets with parent
    has_parent = df["in_response_to_tweet_id"].notna()

    print(f"\nTweets with parent tweet: {has_parent.sum():,}")

    print(
        f"Tweets without parent tweet: "
        f"{(~has_parent).sum():,}"
    )

    # Tweets with responses
    has_response = df["response_tweet_id"].notna()

    print(f"Tweets with responses: {has_response.sum():,}")

    print(
        f"Tweets without responses: "
        f"{(~has_response).sum():,}"
    )

    # Create tweet ID → inbound lookup
    tweet_type = df.set_index("tweet_id")["inbound"]

    # Only tweets having a parent
    replies = df[
        df["in_response_to_tweet_id"].notna()
    ].copy()

    # Convert parent ID
    replies["parent_id"] = (
        replies["in_response_to_tweet_id"]
        .astype("int64")
    )

    # Find parent type
    replies["parent_inbound"] = (
        replies["parent_id"].map(tweet_type)
    )

    # Parent IDs that don't exist
    invalid_parent_ids = (
        replies["parent_inbound"].isna()
    ).sum()

    print(
        f"Parent IDs not found in dataset: "
        f"{invalid_parent_ids:,}"
    )

    # Customer → Support
    customer_to_support = replies[
        (replies["parent_inbound"] == True)
        &
        (replies["inbound"] == False)
    ]

    print(
        f"Customer → Support pairs: "
        f"{len(customer_to_support):,}"
    )

    # Support → Customer
    support_to_customer = replies[
        (replies["parent_inbound"] == False)
        &
        (replies["inbound"] == True)
    ]

    print(
        f"Support → Customer pairs: "
        f"{len(support_to_customer):,}"
    )


def pair_duplicate_analysis(df):
    """Check for duplicate customer-support relationships."""

    print("\n" + "=" * 60)
    print("CUSTOMER → SUPPORT PAIR ANALYSIS")
    print("=" * 60)

    # Tweet ID → inbound status
    tweet_type = df.set_index("tweet_id")["inbound"]

    # Tweets with parent
    replies = df[
        df["in_response_to_tweet_id"].notna()
    ].copy()

    replies["parent_id"] = (
        replies["in_response_to_tweet_id"]
        .astype("int64")
    )

    replies["parent_inbound"] = (
        replies["parent_id"].map(tweet_type)
    )

    # Customer → Support
    customer_to_support = replies[
        (replies["parent_inbound"] == True)
        &
        (replies["inbound"] == False)
    ].copy()

    # Check duplicate pairs
    duplicate_pairs = customer_to_support[
        ["parent_id", "tweet_id"]
    ].duplicated().sum()

    print(
        f"Total customer → support pairs: "
        f"{len(customer_to_support):,}"
    )

    print(
        f"Duplicate customer → support pairs: "
        f"{duplicate_pairs:,}"
    )


def data_quality_summary(df):
    """Print final data quality summary."""

    print("\n" + "=" * 60)
    print("FINAL DATA QUALITY SUMMARY")
    print("=" * 60)

    duplicate_ids = df["tweet_id"].duplicated().sum()

    missing_text = df["text"].isna().sum()

    empty_text = (
        df["text"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    short_text = (
        df["text"]
        .fillna("")
        .str.strip()
        .str.len()
        .lt(3)
        .sum()
    )

    duplicate_rows = df.duplicated().sum()

    print(f"\nTotal rows: {len(df):,}")
    print(f"Duplicate tweet IDs: {duplicate_ids:,}")
    print(f"Duplicate complete rows: {duplicate_rows:,}")
    print(f"Missing text: {missing_text:,}")
    print(f"Empty text: {empty_text:,}")
    print(f"Very short text: {short_text:,}")

    print("\n===== RECOMMENDATION =====")

    if duplicate_ids == 0:
        print("✓ Tweet IDs are unique.")
    else:
        print("⚠ Duplicate tweet IDs need investigation.")

    if missing_text == 0 and empty_text == 0:
        print("✓ Text field is complete.")
    else:
        print("⚠ Missing or empty text should be removed.")

    if duplicate_rows == 0:
        print("✓ No completely duplicate rows found.")
    else:
        print("⚠ Completely duplicate rows should be removed.")

    print("\nData quality analysis completed.")


def main():

    df = load_data()

    duplicate_analysis(df)

    text_quality_analysis(df)

    relationship_quality_analysis(df)

    pair_duplicate_analysis(df)

    data_quality_summary(df)

    print("\n" + "=" * 60)
    print("EDA2 COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()