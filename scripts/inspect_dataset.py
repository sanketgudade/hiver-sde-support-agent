import pandas as pd

DATA_PATH = r"data/raw/archive/twcs/twcs.csv"


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("\n===== DATASET OVERVIEW =====")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")

    print("\n===== COLUMNS =====")
    for column in df.columns:
        print(f"- {column}")

    print("\n===== INBOUND DISTRIBUTION =====")
    print(df["inbound"].value_counts())

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== DATA TYPES =====")
    print(df.dtypes)

    print("\n===== SAMPLE CUSTOMER TWEETS =====")
    print(df[df["inbound"] == True][["tweet_id", "author_id", "text"]].head(5).to_string(index=False))

    print("\n===== SAMPLE SUPPORT TWEETS =====")
    print(df[df["inbound"] == False][["tweet_id", "author_id", "text"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()