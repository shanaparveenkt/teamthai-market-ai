from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR /
    "data" /
    "processed" /
    "cleaned_market_data.csv"
)


def main():

    print("=" * 70)
    print("TEAM THAI DATA AUDIT")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    print("\nROWS:", len(df))
    print("COLUMNS:", len(df.columns))

    print("\nMISSING VALUES")
    print("-" * 70)

    missing = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    print(missing[missing > 0])

    print("\nDUPLICATE ROWS")
    print("-" * 70)

    print(df.duplicated().sum())

    print("\nTEAM THAI PRODUCTS")
    print("-" * 70)

    teamthai = df[
        df["company_group"]
        .astype(str)
        .str.lower()
        .eq("team thai")
    ]

    print(
        teamthai[
            [
                "brand_standardized",
                "product_name"
            ]
        ].to_string(index=False)
    )

    print("\nBRANDS")
    print("-" * 70)

    print(
        df["brand_standardized"]
        .value_counts()
        .head(30)
    )

    print("\nDATA QUALITY")
    print("-" * 70)

    print(
        df["data_quality_category"]
        .value_counts()
    )


if __name__ == "__main__":
    main()