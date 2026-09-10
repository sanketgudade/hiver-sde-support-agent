import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATA_PATH = r"data/raw/archive/twcs/twcs.csv"
OUTPUT_PATH = r"data/processed/customer_support_pairs.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load the raw Twitter customer-support dataset."""

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"Rows loaded: {len(df):,}")

    return df


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(df):
    """Clean tweet text."""

    print("\n" + "=" * 60)
    print("TEXT CLEANING")
    print("=" * 60)

    # Remove leading and trailing whitespace
    df["text"] = df["text"].str.strip()

    # --------------------------------------------------------
    # Remove very short messages
    # --------------------------------------------------------

    before = len(df)

    text_length = df["text"].str.len()

    df = df[text_length >= 3].copy()

    removed_short = before - len(df)

    print(
        f"Very short messages removed: "
        f"{removed_short:,}"
    )

    # --------------------------------------------------------
    # Remove extremely long messages
    # --------------------------------------------------------

    before = len(df)

    text_length = df["text"].str.len()

    df = df[text_length <= 500].copy()

    removed_long = before - len(df)

    print(
        f"Very long messages removed: "
        f"{removed_long:,}"
    )

    print(
        f"Rows after text cleaning: "
        f"{len(df):,}"
    )

    return df


# ============================================================
# BUILD TWEET LOOKUP
# ============================================================

def build_tweet_lookup(df):
    """
    Create a lookup table using tweet_id.

    This allows us to quickly find a tweet's:
    - text
    - inbound status
    """

    print("\n" + "=" * 60)
    print("BUILDING TWEET LOOKUP")
    print("=" * 60)

    tweet_lookup = df.set_index("tweet_id")[
        ["text", "inbound"]
    ]

    print(
        f"Tweet lookup created for "
        f"{len(tweet_lookup):,} tweets."
    )

    return tweet_lookup


# ============================================================
# BUILD CUSTOMER → SUPPORT PAIRS
# ============================================================

def build_customer_support_pairs(df, tweet_lookup):
    """
    Build customer → support conversation pairs.

    Customer:
        inbound = True

    Support:
        inbound = False

    The relationship is established using:
        customer.response_tweet_id
    """

    print("\n" + "=" * 60)
    print("BUILDING CUSTOMER → SUPPORT PAIRS")
    print("=" * 60)

    # --------------------------------------------------------
    # Select customer tweets
    # --------------------------------------------------------

    customers = df[
        df["inbound"] == True
    ].copy()

    print(
        f"Customer tweets available: "
        f"{len(customers):,}"
    )

    # --------------------------------------------------------
    # Keep only customer tweets that have a response
    # --------------------------------------------------------

    customers = customers[
        customers["response_tweet_id"].notna()
    ].copy()

    print(
        f"Customer tweets with response IDs: "
        f"{len(customers):,}"
    )

    # --------------------------------------------------------
    # Convert response_tweet_id into individual IDs
    #
    # Some tweets have multiple responses:
    #
    # response_tweet_id = "5,7"
    #
    # We split them into:
    #
    # 5
    # 7
    # --------------------------------------------------------

    customers["response_tweet_id"] = (
        customers["response_tweet_id"]
        .astype(str)
        .str.split(",")
    )

    customers = customers.explode(
        "response_tweet_id"
    )

    # Remove whitespace around IDs
    customers["response_tweet_id"] = (
        customers["response_tweet_id"]
        .str.strip()
    )

    # --------------------------------------------------------
    # Convert response IDs to numeric
    # --------------------------------------------------------

    customers["response_tweet_id"] = pd.to_numeric(
        customers["response_tweet_id"],
        errors="coerce"
    )

    # Remove invalid IDs
    customers = customers[
        customers["response_tweet_id"].notna()
    ].copy()

    # Convert to integer
    customers["response_tweet_id"] = (
        customers["response_tweet_id"]
        .astype("int64")
    )

    # --------------------------------------------------------
    # Find the corresponding response tweet
    # --------------------------------------------------------

    customers["support_text"] = (
        customers["response_tweet_id"]
        .map(tweet_lookup["text"])
    )

    customers["support_inbound"] = (
        customers["response_tweet_id"]
        .map(tweet_lookup["inbound"])
    )

    # --------------------------------------------------------
    # Keep only responses that actually exist
    # and were written by support
    # --------------------------------------------------------

    pairs = customers[
        customers["support_text"].notna()
        &
        (customers["support_inbound"] == False)
    ].copy()

    print(
        f"Valid customer → support pairs: "
        f"{len(pairs):,}"
    )

    # --------------------------------------------------------
    # Create final dataset
    # --------------------------------------------------------

    pairs = pairs[
        [
            "tweet_id",
            "response_tweet_id",
            "text",
            "support_text"
        ]
    ].copy()

    # Rename columns
    pairs.rename(
        columns={
            "tweet_id": "customer_tweet_id",
            "response_tweet_id": "support_tweet_id",
            "text": "customer_text"
        },
        inplace=True
    )

    # --------------------------------------------------------
    # Remove duplicate pairs
    # --------------------------------------------------------

    before = len(pairs)

    pairs.drop_duplicates(
        subset=[
            "customer_tweet_id",
            "support_tweet_id"
        ],
        inplace=True
    )

    duplicates_removed = before - len(pairs)

    print(
        f"Duplicate pairs removed: "
        f"{duplicates_removed:,}"
    )

    print(
        f"Final pairs: "
        f"{len(pairs):,}"
    )

    return pairs


# ============================================================
# VALIDATE PAIRS
# ============================================================

def validate_pairs(pairs):
    """Validate the final customer-support dataset."""

    print("\n" + "=" * 60)
    print("PAIR VALIDATION")
    print("=" * 60)

    print(
        f"\nTotal customer → support pairs: "
        f"{len(pairs):,}"
    )

    # Missing customer text
    missing_customer = pairs[
        "customer_text"
    ].isna().sum()

    # Missing support text
    missing_support = pairs[
        "support_text"
    ].isna().sum()

    # Duplicate pairs
    duplicate_pairs = pairs[
        [
            "customer_tweet_id",
            "support_tweet_id"
        ]
    ].duplicated().sum()

    print(
        f"Missing customer text: "
        f"{missing_customer:,}"
    )

    print(
        f"Missing support text: "
        f"{missing_support:,}"
    )

    print(
        f"Duplicate pairs: "
        f"{duplicate_pairs:,}"
    )

    # --------------------------------------------------------
    # Display sample pairs
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SAMPLE CUSTOMER → SUPPORT PAIRS")
    print("=" * 60)

    for _, row in pairs.head(5).iterrows():

        print("\nCustomer:")
        print(row["customer_text"])

        print("\nSupport:")
        print(row["support_text"])

        print("-" * 60)


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_pairs(pairs):
    """Save the processed customer-support pairs."""

    print("\n" + "=" * 60)
    print("SAVING PROCESSED DATA")
    print("=" * 60)

    # Create directory if it doesn't exist
    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    pairs.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nProcessed dataset saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        f"Rows saved: "
        f"{len(pairs):,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # Step 1: Load raw data
    df = load_data()

    # Step 2: Clean text
    df = clean_text(df)

    # Step 3: Build tweet lookup
    tweet_lookup = build_tweet_lookup(df)

    # Step 4: Build customer → support pairs
    pairs = build_customer_support_pairs(
        df,
        tweet_lookup
    )

    # Step 5: Validate pairs
    validate_pairs(pairs)

    # Step 6: Save processed dataset
    save_pairs(pairs)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()