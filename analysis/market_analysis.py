from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR /
    "data" /
    "processed" /
    "cleaned_market_data.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 70)
    print("TEAM THAI LIQUID DETERGENT MARKET ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    # ---------------------------------------------------------
    # MARKET DEFINITION
    # ---------------------------------------------------------

    market = df[
        df["product_category"]
        .astype(str)
        .str.lower()
        .eq("liquid detergent")
    ].copy()

    print(f"\nLiquid detergent products : {len(market)}")

    # ---------------------------------------------------------
    # BASIC MARKET DATA
    # ---------------------------------------------------------

    market_summary = pd.DataFrame([{
        "total_products": len(market),
        "unique_brands": market["brand_standardized"].nunique(),
        "marketplaces": market["marketplace"].nunique()
        if "marketplace" in market.columns else 0,
        "average_price": market["selling_price"].mean(),
        "median_price": market["selling_price"].median(),
        "minimum_price": market["selling_price"].min(),
        "maximum_price": market["selling_price"].max(),
        "average_rating": market["rating"].mean()
    }])

    market_summary.to_csv(
        OUTPUT_DIR / "market_analysis.csv",
        index=False
    )

    # ---------------------------------------------------------
    # BRAND ANALYSIS
    # ---------------------------------------------------------

    brand_analysis = (
        market
        .groupby("brand_standardized")
        .agg(
            products=("product_name", "count"),
            average_price=("selling_price", "mean"),
            median_price=("selling_price", "median"),
            average_rating=("rating", "mean"),
            total_reviews=("rating_count", "sum"),
            average_reviews=("rating_count", "mean"),
            average_discount=(
                "discount_pct_calculated",
                "mean"
            ),
            average_price_per_100ml=(
                "price_per_100ml",
                "mean"
            )
        )
        .reset_index()
    )

    brand_analysis["product_share_pct"] = (
        brand_analysis["products"] /
        len(market) * 100
    )

    brand_analysis.to_csv(
        OUTPUT_DIR / "brand_competitor_analysis.csv",
        index=False
    )

    # ---------------------------------------------------------
    # MARKETPLACE
    # ---------------------------------------------------------

    if "marketplace" in market.columns:

        marketplace = (
            market
            .groupby("marketplace")
            .agg(
                products=("product_name", "count"),
                brands=("brand_standardized", "nunique"),
                average_price=("selling_price", "mean"),
                average_rating=("rating", "mean")
            )
            .reset_index()
        )

        marketplace["market_share_pct"] = (
            marketplace["products"] /
            len(market) * 100
        )

        marketplace.to_csv(
            OUTPUT_DIR / "marketplace_summary.csv",
            index=False
        )

    # ---------------------------------------------------------
    # PRICE SEGMENTS
    # ---------------------------------------------------------

    def price_segment(price):

        if price < 150:
            return "Budget (<₹150)"

        elif price < 300:
            return "Mass Market (₹150–299)"

        elif price < 600:
            return "Mid Premium (₹300–599)"

        elif price < 1000:
            return "Premium (₹600–999)"

        return "High Premium (₹1000+)"

    market["price_segment"] = (
        market["selling_price"]
        .apply(price_segment)
    )

    price_segments = (
        market
        .groupby("price_segment")
        .agg(
            products=("product_name", "count"),
            brands=("brand_standardized", "nunique"),
            average_price=("selling_price", "mean"),
            median_price=("selling_price", "median"),
            average_rating=("rating", "mean")
        )
        .reset_index()
    )

    price_segments["market_share_pct"] = (
        price_segments["products"] /
        len(market) * 100
    )

    price_segments.to_csv(
        OUTPUT_DIR / "price_segment_analysis.csv",
        index=False
    )

    # ---------------------------------------------------------
    # TOP REVIEWED
    # ---------------------------------------------------------

    top_reviews = (
        market
        .sort_values(
            "rating_count",
            ascending=False
        )
        .head(20)
    )

    top_reviews.to_csv(
        OUTPUT_DIR / "top_review_products.csv",
        index=False
    )

    # ---------------------------------------------------------
    # TOP RATED
    # ---------------------------------------------------------

    top_rated = (
        market[
            market["rating_count"] >= 10
        ]
        .sort_values(
            ["rating", "rating_count"],
            ascending=[False, False]
        )
        .head(20)
    )

    top_rated.to_csv(
        OUTPUT_DIR / "top_rated_products.csv",
        index=False
    )

    # ---------------------------------------------------------
    # DISCOUNTS
    # ---------------------------------------------------------

    top_discounted = (
        market
        .sort_values(
            "discount_pct_calculated",
            ascending=False
        )
        .head(20)
    )

    top_discounted.to_csv(
        OUTPUT_DIR / "top_discounted_products.csv",
        index=False
    )

    # ---------------------------------------------------------
    # LOWEST NORMALIZED PRICE
    # ---------------------------------------------------------

    normalized = (
        market[
            market["price_per_100ml"].notna()
        ]
        .sort_values(
            "price_per_100ml"
        )
        .head(20)
    )

    normalized.to_csv(
        OUTPUT_DIR /
        "lowest_normalized_price_products.csv",
        index=False
    )

    # ---------------------------------------------------------
    # TEAM THAI PORTFOLIO
    # ---------------------------------------------------------

    teamthai = market[
        market["company_group"]
        .astype(str)
        .str.lower()
        .eq("team thai")
    ].copy()

    teamthai.to_csv(
        OUTPUT_DIR / "teamthai_portfolio.csv",
        index=False
    )

    print("\nTeam Thai products found:",
          len(teamthai))

    if len(teamthai) > 0:

        print(
            teamthai[
                [
                    "brand_standardized",
                    "product_name",
                    "selling_price",
                    "rating"
                ]
            ].to_string(index=False)
        )

    # ---------------------------------------------------------
    # COMPETITOR VS TEAM THAI
    # ---------------------------------------------------------

    comparison = (
        market
        .groupby("company_group")
        .agg(
            products=("product_name", "count"),
            brands=("brand_standardized", "nunique"),
            average_price=("selling_price", "mean"),
            average_rating=("rating", "mean"),
            total_reviews=("rating_count", "sum")
        )
        .reset_index()
    )

    comparison.to_csv(
        OUTPUT_DIR / "teamthai_vs_market.csv",
        index=False
    )

    # ---------------------------------------------------------
    # EXECUTIVE DATA
    # ---------------------------------------------------------

    summary = f"""
TEAM THAI MARKET DATA SUMMARY

Total liquid detergent products: {len(market)}
Unique brands: {market["brand_standardized"].nunique()}
Average price: ₹{market["selling_price"].mean():.2f}
Median price: ₹{market["selling_price"].median():.2f}
Average rating: {market["rating"].mean():.2f}

Team Thai products detected: {len(teamthai)}

Team Thai brands detected:
{", ".join(sorted(teamthai["brand_standardized"].dropna().unique()))}

Largest price segment:
{price_segments.sort_values("products", ascending=False).iloc[0]["price_segment"]}

Largest competitor brands by product count:
{", ".join(
    brand_analysis
    .sort_values("products", ascending=False)
    .head(10)["brand_standardized"]
    .tolist()
)}
"""

    with open(
        OUTPUT_DIR / "executive_summary.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(summary)

    print("\nMARKET ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()