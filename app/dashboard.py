from pathlib import Path
import pandas as pd
import streamlit as st

from components import (
    metric_card,
    section_title,
    show_dataframe
)


BASE_DIR = Path(__file__).resolve().parent.parent

ANALYSIS_DIR = (
    BASE_DIR /
    "data" /
    "analysis"
)

OUTPUT_DIR = (
    BASE_DIR /
    "outputs"
)


def load_csv(filename):

    path = ANALYSIS_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def render_dashboard():

    st.title("Team Thai Market Intelligence")

    st.caption(
        "AI-powered liquid detergent market analysis"
    )

    # -----------------------------------------------------
    # MARKET OVERVIEW
    # -----------------------------------------------------

    section_title("Market Overview")

    market = load_csv(
        "market_analysis.csv"
    )

    if not market.empty:

        row = market.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            metric_card(
                "Products",
                int(row["total_products"])
            )

        with col2:
            metric_card(
                "Brands",
                int(row["unique_brands"])
            )

        with col3:
            metric_card(
                "Average Price",
                f"₹{row['average_price']:.0f}"
            )

        with col4:
            metric_card(
                "Average Rating",
                f"{row['average_rating']:.2f}"
            )

    # -----------------------------------------------------
    # TEAM THAI
    # -----------------------------------------------------

    section_title(
        "Team Thai Portfolio"
    )

    teamthai = load_csv(
        "teamthai_portfolio.csv"
    )

    if teamthai.empty:

        st.warning(
            "No Team Thai marketplace products "
            "were found in the current dataset."
        )

    else:

        show_dataframe(
            teamthai[
                [
                    "brand_standardized",
                    "product_name",
                    "selling_price",
                    "rating",
                    "rating_count",
                    "marketplace"
                ]
            ]
        )

    # -----------------------------------------------------
    # COMPETITORS
    # -----------------------------------------------------

    section_title(
        "Competitor Landscape"
    )

    competitors = load_csv(
        "brand_competitor_analysis.csv"
    )

    if not competitors.empty:

        competitors = (
            competitors
            .sort_values(
                "products",
                ascending=False
            )
            .head(15)
        )

        show_dataframe(
            competitors
        )

    # -----------------------------------------------------
    # PRICE SEGMENTS
    # -----------------------------------------------------

    section_title(
        "Price Segments"
    )

    prices = load_csv(
        "price_segment_analysis.csv"
    )

    if not prices.empty:

        show_dataframe(
            prices
        )

    # -----------------------------------------------------
    # AI INSIGHTS
    # -----------------------------------------------------

    section_title(
        "AI Market Intelligence"
    )

    report_file = (
        OUTPUT_DIR /
        "executive_report.txt"
    )

    if report_file.exists():

        report = report_file.read_text(
            encoding="utf-8"
        )

        st.markdown(report)

    else:

        st.info(
            "AI insights have not been generated yet."
        )