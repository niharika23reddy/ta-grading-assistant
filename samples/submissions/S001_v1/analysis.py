import pandas as pd
import matplotlib.pyplot as plt


def clean_data(df):
    """Remove rows with missing temperature values."""
    return df.dropna(subset=["temp"])


def main():
    df = clean_data(pd.read_csv("day.csv"))
    print(df.shape)
    print(df["cnt"].mean(), df["cnt"].median())
    plt.hist(df["cnt"], bins=30)
    plt.xlabel("Daily rentals")
    plt.ylabel("Number of days")
    plt.title("Distribution of daily bike rentals")
    plt.savefig("hist.png")


if __name__ == "__main__":
    main()
