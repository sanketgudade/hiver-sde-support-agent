import pandas as pd

DATA_PATH = r"data/raw/archive/twcs/twcs.csv"


def load_data():
    """Load the complete Twitter customer-support dataset."""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print("Dataset loaded successfully.")
    return df


def dataset_overview(df):
    """Display basic dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Total tweets: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Unique authors: {df['author_id'].nunique():,}")

    print("\nCustomer tweets:")
    print(f"  {df['inbound'].sum():,}")

    print("\nSupport tweets:")
    print(f"  {(~df['inbound']).sum():,}")


def author_analysis(df):
    """Analyze the most active authors/accounts."""
    print("\n" + "=" * 60)
    print("TOP AUTHORS / SUPPORT ACCOUNTS")
    print("=" * 60)

    top_authors = df["author_id"].value_counts().head(20)
    print(top_authors.to_string())


def text_analysis(df):
    """Analyze customer and support message lengths."""
    print("\n" + "=" * 60)
    print("TEXT ANALYSIS")
    print("=" * 60)

    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()

    print("\nOverall text statistics:")
    print(df[["text_length", "word_count"]].describe().round(2))

    customer = df[df["inbound"]]
    support = df[~df["inbound"]]

    print("\nCustomer message statistics:")
    print(customer[["text_length", "word_count"]].describe().round(2))

    print("\nSupport message statistics:")
    print(support[["text_length", "word_count"]].describe().round(2))


def relationship_analysis(df):
    """Analyze tweet-to-tweet conversation relationships."""
    print("\n" + "=" * 60)
    print("CONVERSATION RELATIONSHIP ANALYSIS")
    print("=" * 60)

    has_parent = df["in_response_to_tweet_id"].notna()
    has_response = df["response_tweet_id"].notna()

    print(f"Tweets with a parent tweet: {has_parent.sum():,}")
    print(f"Tweets without a parent tweet: {(~has_parent).sum():,}")

    print(f"Tweets with responses: {has_response.sum():,}")
    print(f"Tweets without responses: {(~has_response).sum():,}")


def customer_support_pair_analysis(df):
    """Analyze direct customer-to-support tweet relationships."""
    print("\n" + "=" * 60)
    print("CUSTOMER → SUPPORT PAIR ANALYSIS")
    print("=" * 60)

    # Create a lookup from tweet ID to inbound status
    tweet_type = df.set_index("tweet_id")["inbound"]

    # Only tweets that directly respond to another tweet
    replies = df[df["in_response_to_tweet_id"].notna()].copy()

    # Convert parent tweet IDs to integers for matching
    replies["parent_id"] = replies["in_response_to_tweet_id"].astype("int64")

    # Identify the type of the parent tweet
    replies["parent_inbound"] = replies["parent_id"].map(tweet_type)

    # Direct customer → support responses
    customer_to_support = replies[
        (replies["parent_inbound"] == True) &
        (replies["inbound"] == False)
    ]

    print(f"Tweets with a parent: {len(replies):,}")
    print(f"Direct customer → support pairs: {len(customer_to_support):,}")

    customer_count = df["inbound"].sum()
    percentage = (len(customer_to_support) / customer_count) * 100

    print(
        f"Customer tweets receiving a direct support response: "
        f"{percentage:.2f}%"
    )

    print("\n===== SAMPLE CUSTOMER → SUPPORT PAIRS =====")

    for _, row in customer_to_support.head(5).iterrows():
        customer_id = row["parent_id"]

        customer_text = df.loc[
            df["tweet_id"] == customer_id, "text"
        ].iloc[0]

        support_text = row["text"]

        print(f"\nCustomer: {customer_text}")
        print(f"Support : {support_text}")


def main():
    df = load_data()

    dataset_overview(df)
    author_analysis(df)
    text_analysis(df)
    relationship_analysis(df)
    customer_support_pair_analysis(df)

    print("\n" + "=" * 60)
    print("EDA COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()