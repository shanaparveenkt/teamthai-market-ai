from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MAIN INVESTIGATION
# ============================================================

def main():

    print("=" * 70)
    print("TEAM THAI DATA INVESTIGATION")
    print("=" * 70)

    input_file = PROCESSED_DIR / "cleaned_market_data.csv"

    if not input_file.exists():
        raise FileNotFoundError(
            f"\nCleaned dataset not found:\n{input_file}\n\n"
            "Run clean_data.py first."
        )

    print("\n📂 Loading cleaned dataset...")

    df = pd.read_csv(input_file)

    print(f"Rows loaded    : {len(df)}")
    print(f"Columns loaded : {len(df.columns)}")


    # ========================================================
    # BASIC DATASET INFORMATION
    # ========================================================

    print("\n" + "=" * 70)
    print("1. DATASET STRUCTURE")
    print("=" * 70)

    print(f"\nRows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumns:")

    for column in df.columns:
        print(f" - {column}")


    # ========================================================
    # MISSING VALUES
    # ========================================================

    print("\n" + "=" * 70)
    print("2. MISSING VALUE INVESTIGATION")
    print("=" * 70)

    missing = df.isna().sum()

    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:

        print("\nNo missing values found.")

    else:

        print("\nMissing values:")

        for column, count in missing.items():

            percentage = (count / len(df)) * 100

            print(
                f" - {column}: "
                f"{count} ({percentage:.2f}%)"
            )


    # ========================================================
    # DUPLICATES
    # ========================================================

    print("\n" + "=" * 70)
    print("3. DUPLICATE INVESTIGATION")
    print("=" * 70)

    duplicate_count = df.duplicated().sum()

    print(f"\nExact duplicate rows : {duplicate_count}")

    if "product_name" in df.columns:

        product_duplicates = df["product_name"].duplicated().sum()

        print(
            f"Duplicate product names : "
            f"{product_duplicates}"
        )


    # ========================================================
    # TEAM THAI INVESTIGATION
    # ========================================================

    print("\n" + "=" * 70)
    print("4. TEAM THAI PORTFOLIO INVESTIGATION")
    print("=" * 70)

    if "company_group" in df.columns:

        print("\nCompany groups:")

        print(
            df["company_group"]
            .value_counts(dropna=False)
            .to_string()
        )

    teamthai = pd.DataFrame()

    if "company_group" in df.columns:

        teamthai = df[
            df["company_group"]
            .astype(str)
            .str.lower()
            .eq("team thai")
        ]

    print(
        f"\nTeam Thai products detected : "
        f"{len(teamthai)}"
    )

    if len(teamthai) > 0:

        if "brand_standardized" in teamthai.columns:

            print("\nTeam Thai brands:")

            print(
                teamthai["brand_standardized"]
                .value_counts()
                .to_string()
            )

        display_columns = [
            "brand_standardized",
            "product_name",
            "selling_price",
            "rating",
            "rating_count"
        ]

        display_columns = [
            column
            for column in display_columns
            if column in teamthai.columns
        ]

        print("\nTeam Thai products:")

        print(
            teamthai[display_columns]
            .to_string(index=False)
        )

    else:

        print("\n⚠️ No Team Thai products detected.")


    # ========================================================
    # BRAND INVESTIGATION
    # ========================================================

    print("\n" + "=" * 70)
    print("5. BRAND INVESTIGATION")
    print("=" * 70)

    if "brand_standardized" in df.columns:

        brand_counts = (
            df["brand_standardized"]
            .value_counts()
        )

        print(
            f"\nUnique standardized brands : "
            f"{len(brand_counts)}"
        )

        print("\nTop brands:")

        print(
            brand_counts.head(20)
            .to_string()
        )


    # ========================================================
    # UNKNOWN BRANDS
    # ========================================================

    print("\n" + "=" * 70)
    print("6. UNKNOWN BRAND INVESTIGATION")
    print("=" * 70)

    if "brand_standardized" in df.columns:

        unknown = df[
            df["brand_standardized"]
            .astype(str)
            .str.lower()
            .eq("unknown")
        ]

        print(
            f"\nUnknown brand products : "
            f"{len(unknown)}"
        )

        if len(unknown) > 0:

            columns = [
                "brand",
                "product_name",
                "marketplace"
            ]

            columns = [
                column
                for column in columns
                if column in unknown.columns
            ]

            print("\nUnknown brand examples:")

            print(
                unknown[columns]
                .head(30)
                .to_string(index=False)
            )


    # ========================================================
    # PRICE INVESTIGATION
    # ========================================================

    print("\n" + "=" * 70)
    print("7. PRICE INVESTIGATION")
    print("=" * 70)

    if "selling_price" in df.columns:

        price = pd.to_numeric(
            df["selling_price"],
            errors="coerce"
        )

        print(
            f"\nMissing prices : "
            f"{price.isna().sum()}"
        )

        print(
            f"Zero prices    : "
            f"{(price <= 0).sum()}"
        )

        print(
            f"Minimum price  : "
            f"₹{price.min():.2f}"
        )

        print(
            f"Maximum price  : "
            f"₹{price.max():.2f}"
        )

        print(
            f"Average price  : "
            f"₹{price.mean():.2f}"
        )


    # ========================================================
    # RATING INVESTIGATION
    # ========================================================

    print("\n" + "=" * 70)
    print("8. RATING INVESTIGATION")
    print("=" * 70)

    if "rating" in df.columns:

        rating = pd.to_numeric(
            df["rating"],
            errors="coerce"
        )

        invalid_ratings = (
            (rating < 0) |
            (rating > 5)
        ).sum()

        print(
            f"\nMissing ratings : "
            f"{rating.isna().sum()}"
        )

        print(
            f"Invalid ratings : "
            f"{invalid_ratings}"
        )

        print(
            f"Average rating  : "
            f"{rating.mean():.2f}"
        )


    # ========================================================
    # REVIEW INVESTIGATION
    # ========================================================

    print("\n" + "=" * 70)
    print("9. REVIEW INVESTIGATION")
    print("=" * 70)

    if "rating_count" in df.columns:

        reviews = pd.to_numeric(
            df["rating_count"],
            errors="coerce"
        ).fillna(0)

        print(
            f"\nProducts with reviews : "
            f"{(reviews > 0).sum()}"
        )

        print(
            f"Products without reviews : "
            f"{(reviews == 0).sum()}"
        )

        print(
            f"Maximum review count : "
            f"{reviews.max():,.0f}"
        )


    # ========================================================
    # PACK SIZE INVESTIGATION
    # ========================================================

    print("\n" + "=" * 70)
    print("10. PACK SIZE INVESTIGATION")
    print("=" * 70)

    if "pack_size_ml" in df.columns:

        pack = pd.to_numeric(
            df["pack_size_ml"],
            errors="coerce"
        )

        print(
            f"\nMissing pack sizes : "
            f"{pack.isna().sum()}"
        )

        print(
            f"Minimum pack size  : "
            f"{pack.min():.0f} ml"
        )

        print(
            f"Maximum pack size  : "
            f"{pack.max():.0f} ml"
        )


    # ========================================================
    # MARKETPLACE INVESTIGATION
    # ========================================================

    print("\n" + "=" * 70)
    print("11. MARKETPLACE INVESTIGATION")
    print("=" * 70)

    if "marketplace" in df.columns:

        print("\nMarketplace distribution:")

        print(
            df["marketplace"]
            .value_counts(dropna=False)
            .to_string()
        )

    else:

        print(
            "\n⚠️ Marketplace column not available."
        )


    # ========================================================
    # PRODUCT CATEGORY
    # ========================================================

    print("\n" + "=" * 70)
    print("12. PRODUCT CATEGORY INVESTIGATION")
    print("=" * 70)

    if "product_category" in df.columns:

        print(
            "\nProduct categories:"
        )

        print(
            df["product_category"]
            .value_counts()
            .to_string()
        )


    # ========================================================
    # DATA QUALITY
    # ========================================================

    print("\n" + "=" * 70)
    print("13. DATA QUALITY INVESTIGATION")
    print("=" * 70)

    if "data_quality_category" in df.columns:

        print(
            "\nData quality categories:"
        )

        print(
            df["data_quality_category"]
            .value_counts()
            .to_string()
        )


    # ========================================================
    # SAVE INVESTIGATION REPORT
    # ========================================================

    report_file = OUTPUT_DIR / "data_investigation_report.txt"

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "TEAM THAI DATA INVESTIGATION REPORT\n"
        )

        file.write("=" * 60 + "\n\n")

        file.write(
            f"Rows: {len(df)}\n"
        )

        file.write(
            f"Columns: {len(df.columns)}\n\n"
        )

        if "company_group" in df.columns:

            file.write(
                "COMPANY GROUPS\n"
            )

            file.write(
                df["company_group"]
                .value_counts()
                .to_string()
            )

            file.write("\n\n")

        if "brand_standardized" in df.columns:

            file.write(
                "BRAND COUNTS\n"
            )

            file.write(
                df["brand_standardized"]
                .value_counts()
                .to_string()
            )

            file.write("\n\n")

        file.write(
            "MISSING VALUES\n"
        )

        file.write(
            df.isna()
            .sum()
            .to_string()
        )

        file.write("\n")


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETE")
    print("=" * 70)

    print("\n📁 Investigation report:")
    print(report_file)

    print("\nNext step:")
    print("Review Team Thai brand detection before market analysis.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()