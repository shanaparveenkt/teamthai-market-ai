from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "cleaned_market_data.csv"
)

AI_CONTEXT_FILE = (
    BASE_DIR
    / "outputs"
    / "ai_market_context.json"
)

AI_INSIGHTS_FILE = (
    BASE_DIR
    / "outputs"
    / "ai_generated_insights.json"
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = "openai/gpt-oss-20b"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Team Thai AI Market Intelligence",
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_market_data():

    if not PROCESSED_FILE.exists():
        return None

    try:
        return pd.read_csv(PROCESSED_FILE)
    except Exception:
        return None


@st.cache_data
def load_json_file(file_path):

    if not file_path.exists():
        return {}

    try:
        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    except Exception:
        return {}


df = load_market_data()

ai_context = load_json_file(AI_CONTEXT_FILE)

ai_insights = load_json_file(AI_INSIGHTS_FILE)


# ============================================================
# DATA CHECK
# ============================================================

if df is None:

    st.error(
        "Cleaned market dataset was not found."
    )

    st.info(
        "Run analysis/clean_data.py first."
    )

    st.stop()


# ============================================================
# NUMERIC CLEANING FOR APP
# ============================================================

numeric_columns = [
    "selling_price",
    "mrp",
    "rating",
    "rating_count",
    "pack_size_ml",
    "price_per_100ml",
    "discount_pct_calculated"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# LIQUID DETERGENT DATA
# ============================================================

if "product_category" in df.columns:

    liquid_df = df[
        df["product_category"]
        == "Liquid Detergent"
    ].copy()

else:

    liquid_df = df.copy()


# ============================================================
# TEAM THAI / COMPETITOR DATA
# ============================================================

if "company_group" in liquid_df.columns:

    teamthai_df = liquid_df[
        liquid_df["company_group"]
        == "Team Thai"
    ].copy()

    competitor_df = liquid_df[
        liquid_df["company_group"]
        == "Competitor"
    ].copy()

else:

    teamthai_df = liquid_df.iloc[0:0].copy()
    competitor_df = liquid_df.copy()


# ============================================================
# IMPORTANT:
# GLOBAL VALID PRICE DATA
#
# This fixes the previous NameError.
# valid_prices is available to EVERY page.
# ============================================================

if "selling_price" in liquid_df.columns:

    valid_prices = liquid_df[
        liquid_df["selling_price"].notna()
        &
        (liquid_df["selling_price"] > 0)
    ].copy()

else:

    valid_prices = liquid_df.copy()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def money(value):

    if pd.isna(value):
        return "₹0"

    return f"₹{value:,.2f}"


def number(value):

    if pd.isna(value):
        return "0"

    return f"{value:,.0f}"


def price_segment(price):

    if pd.isna(price):
        return "Unknown"

    if price < 150:
        return "Budget"

    if price < 300:
        return "Mass Market"

    if price < 600:
        return "Mid Premium"

    return "Premium"


def safe_bar_chart(data, x=None, y=None, height=350):

    """
    Streamlit-safe bar chart.

    Resets index and uses explicit columns to avoid:
    KeyError: ['index -- streamlit-generated']
    """

    if data is None or len(data) == 0:
        st.info("No data available for this chart.")
        return

    chart_df = data.copy()

    chart_df = chart_df.reset_index(drop=True)

    if x is not None and y is not None:

        if x not in chart_df.columns:
            st.info("Chart data is unavailable.")
            return

        if y not in chart_df.columns:
            st.info("Chart data is unavailable.")
            return

        chart_df = chart_df[[x, y]]

        chart_df = chart_df.dropna(
            subset=[x, y]
        )

        if len(chart_df) == 0:
            st.info("No data available.")
            return

        st.bar_chart(
            chart_df,
            x=x,
            y=y,
            height=height
        )

    else:

        st.bar_chart(
            chart_df,
            height=height
        )


def safe_line_chart(data, x=None, y=None, height=350):

    if data is None or len(data) == 0:
        st.info("No data available.")
        return

    chart_df = data.copy().reset_index(drop=True)

    if x and y:

        if x not in chart_df.columns:
            return

        if y not in chart_df.columns:
            return

        st.line_chart(
            chart_df[[x, y]],
            x=x,
            y=y,
            height=height
        )

    else:

        st.line_chart(
            chart_df,
            height=height
        )


def display_dataframe(data, columns=None):

    if data is None or len(data) == 0:

        st.info("No data available.")

        return

    if columns:

        available = [
            c for c in columns
            if c in data.columns
        ]

        if available:
            data = data[available]

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PDF MANAGEMENT REPORT
# ============================================================

def create_management_report():

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        spaceAfter=12
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        spaceAfter=6
    )

    small_style = ParagraphStyle(
        "ReportSmall",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11
    )

    story = []

    # ========================================================
    # CALCULATIONS
    # ========================================================

    total_products = len(liquid_df)

    unique_brands = (
        liquid_df["brand_standardized"].nunique()
        if "brand_standardized" in liquid_df.columns
        else 0
    )

    market_price = (
        valid_prices["selling_price"].mean()
        if len(valid_prices) > 0
        else 0
    )

    market_median = (
        valid_prices["selling_price"].median()
        if len(valid_prices) > 0
        else 0
    )

    market_rating = (
        liquid_df["rating"].mean()
        if "rating" in liquid_df.columns
        else 0
    )

    total_reviews = (
        liquid_df["rating_count"].sum()
        if "rating_count" in liquid_df.columns
        else 0
    )

    team_price = (
        teamthai_df["selling_price"].mean()
        if len(teamthai_df) > 0
        and "selling_price" in teamthai_df.columns
        else 0
    )

    team_rating = (
        teamthai_df["rating"].mean()
        if len(teamthai_df) > 0
        and "rating" in teamthai_df.columns
        else 0
    )

    team_reviews = (
        teamthai_df["rating_count"].sum()
        if len(teamthai_df) > 0
        and "rating_count" in teamthai_df.columns
        else 0
    )

    price_difference = (
        ((team_price - market_price) / market_price) * 100
        if market_price > 0 and team_price > 0
        else 0
    )

    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        Paragraph(
            "Team Thai Liquid Detergent Market Intelligence",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Management & Strategic Decision Report",
            ParagraphStyle(
                "Subtitle",
                parent=body_style,
                alignment=TA_CENTER,
                fontSize=11
            )
        )
    )

    story.append(Spacer(1, 10))

    # ========================================================
    # EXECUTIVE SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "1. Executive Summary",
            heading_style
        )
    )

    summary_text = f"""
    The analysed liquid detergent marketplace contains
    <b>{total_products:,}</b> observed products across
    approximately <b>{unique_brands:,}</b> brands.
    The observed average selling price is
    <b>{money(market_price)}</b>, with a median of
    <b>{money(market_median)}</b>.
    The average product rating is
    <b>{market_rating:.2f}</b> and the observed review volume is
    approximately <b>{number(total_reviews)}</b>.
    """

    story.append(
        Paragraph(
            summary_text,
            body_style
        )
    )

    # ========================================================
    # MARKET SNAPSHOT
    # ========================================================

    story.append(
        Paragraph(
            "2. Market Snapshot",
            heading_style
        )
    )

    market_table = [
        ["Metric", "Observed Value"],
        ["Liquid detergent products", f"{total_products:,}"],
        ["Brands", f"{unique_brands:,}"],
        ["Average price", money(market_price)],
        ["Median price", money(market_median)],
        ["Average rating", f"{market_rating:.2f}"],
        ["Total reviews", number(total_reviews)]
    ]

    table = Table(
        market_table,
        colWidths=[90 * mm, 70 * mm]
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6)
            ]
        )
    )

    story.append(table)

    # ========================================================
    # TEAM THAI BENCHMARK
    # ========================================================

    story.append(
        Paragraph(
            "3. Team Thai Benchmark",
            heading_style
        )
    )

    if len(teamthai_df) == 0:

        story.append(
            Paragraph(
                "No Team Thai products were detected in the dataset.",
                body_style
            )
        )

    else:

        benchmark_table = [
            ["Metric", "Team Thai", "Market"],
            [
                "Products",
                str(len(teamthai_df)),
                str(total_products)
            ],
            [
                "Average price",
                money(team_price),
                money(market_price)
            ],
            [
                "Average rating",
                f"{team_rating:.2f}",
                f"{market_rating:.2f}"
            ],
            [
                "Review volume",
                number(team_reviews),
                number(total_reviews)
            ]
        ]

        table = Table(
            benchmark_table,
            colWidths=[
                65 * mm,
                45 * mm,
                45 * mm
            ]
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        9
                    ),
                    (
                        "ALIGN",
                        (1, 1),
                        (-1, -1),
                        "CENTER"
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    )
                ]
            )
        )

        story.append(table)

        story.append(Spacer(1, 8))

        story.append(
            Paragraph(
                f"""
                Team Thai's average observed price is
                <b>{abs(price_difference):.1f}%</b>
                {"below" if price_difference < 0 else "above"}
                the market average.
                """,
                body_style
            )
        )

    # ========================================================
    # COMPETITIVE LANDSCAPE
    # ========================================================

    story.append(
        Paragraph(
            "4. Competitive Landscape",
            heading_style
        )
    )

    if len(competitor_df) > 0:

        competitor_summary = (
            competitor_df
            .groupby("brand_standardized")
            .agg(
                products=("product_name", "count"),
                average_price=("selling_price", "mean"),
                average_rating=("rating", "mean"),
                total_reviews=("rating_count", "sum")
            )
            .sort_values(
                "total_reviews",
                ascending=False
            )
            .head(10)
            .reset_index()
        )

        competitor_table = [
            [
                "Brand",
                "Products",
                "Avg Price",
                "Rating",
                "Reviews"
            ]
        ]

        for _, row in competitor_summary.iterrows():

            competitor_table.append(
                [
                    str(row["brand_standardized"]),
                    str(int(row["products"])),
                    money(row["average_price"]),
                    f"{row['average_rating']:.2f}",
                    number(row["total_reviews"])
                ]
            )

        table = Table(
            competitor_table,
            colWidths=[
                48 * mm,
                22 * mm,
                30 * mm,
                22 * mm,
                30 * mm
            ]
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        colors.grey
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        8
                    ),
                    (
                        "ALIGN",
                        (1, 1),
                        (-1, -1),
                        "CENTER"
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE"
                    )
                ]
            )
        )

        story.append(table)

    # ========================================================
    # OPPORTUNITY AREAS
    # ========================================================

    story.append(
        Paragraph(
            "5. Opportunity Areas",
            heading_style
        )
    )

    opportunities = []

    if len(valid_prices) > 0:

        pricing = valid_prices.copy()

        pricing["segment"] = (
            pricing["selling_price"]
            .apply(price_segment)
        )

        segment_counts = (
            pricing["segment"]
            .value_counts()
        )

        if len(segment_counts) > 0:

            largest_segment = (
                segment_counts
                .idxmax()
            )

            opportunities.append(
                f"The largest observed price segment is "
                f"<b>{largest_segment}</b>, indicating the strongest "
                f"observed concentration of products."
            )

    if len(teamthai_df) > 0:

        if team_reviews < total_reviews:

            opportunities.append(
                "Team Thai has an opportunity to strengthen "
                "review volume and consumer credibility."
            )

        if team_rating >= market_rating:

            opportunities.append(
                "Team Thai's observed rating is competitive "
                "with or above the overall market average."
            )

        if price_difference < 0:

            opportunities.append(
                "Team Thai's lower observed price can support "
                "a value-oriented positioning strategy."
            )

    opportunities.append(
        "Product differentiation should focus on clear consumer "
        "benefits rather than competing only on price."
    )

    for opportunity in opportunities:

        story.append(
            Paragraph(
                f"• {opportunity}",
                body_style
            )
        )

    # ========================================================
    # STRATEGIC RECOMMENDATIONS
    # ========================================================

    story.append(
        Paragraph(
            "6. Strategic Recommendations",
            heading_style
        )
    )

    recommendations = [
        (
            "HIGH",
            "Strengthen review volume",
            "Increase consumer trust and marketplace credibility."
        ),
        (
            "HIGH",
            "Improve product differentiation",
            "Create a clear reason for consumers to choose Team Thai."
        ),
        (
            "HIGH",
            "Monitor price-per-100ml",
            "Ensure that headline price also represents attractive value."
        ),
        (
            "MEDIUM",
            "Expand portfolio selectively",
            "Use observed market gaps rather than adding products blindly."
        ),
        (
            "MEDIUM",
            "Track competitor movements",
            "Monitor price, pack size, ratings and review strength."
        )
    ]

    recommendation_table = [
        [
            "Priority",
            "Action",
            "Business Objective"
        ]
    ]

    for priority, action, objective in recommendations:

        recommendation_table.append(
            [
                priority,
                action,
                objective
            ]
        )

    table = Table(
        recommendation_table,
        colWidths=[
            25 * mm,
            65 * mm,
            65 * mm
        ]
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8.5
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                )
            ]
        )
    )

    story.append(table)

    # ========================================================
    # DATA LIMITATIONS
    # ========================================================

    story.append(
        Paragraph(
            "7. Data Limitations",
            heading_style
        )
    )

    limitations = [
        "The analysis is based on observed marketplace data.",
        "Product listings do not necessarily represent actual sales volume.",
        "Review counts should not be interpreted as market share.",
        "The dataset does not directly establish revenue or profitability.",
        "Strategic decisions should combine this analysis with internal sales, distribution and consumer data."
    ]

    for limitation in limitations:

        story.append(
            Paragraph(
                f"• {limitation}",
                body_style
            )
        )

    # ========================================================
    # FINAL CONCLUSION
    # ========================================================

    story.append(
        Paragraph(
            "8. Management Conclusion",
            heading_style
        )
    )

    if len(teamthai_df) > 0:

        conclusion = """
        Team Thai has an opportunity to build a stronger position in
        liquid detergent by combining competitive value with stronger
        consumer credibility and clearer product differentiation.
        The immediate focus should be on strengthening the existing
        proposition before pursuing broad portfolio expansion.
        """

    else:

        conclusion = """
        Team Thai products were not detected in the analysed dataset.
        Product classification and marketplace coverage should therefore
        be validated before using this report for strategic decisions.
        """

    story.append(
        Paragraph(
            conclusion,
            body_style
        )
    )

    # ========================================================
    # FOOTER
    # ========================================================

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "Generated by Team Thai AI Market Intelligence",
            small_style
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer



# ============================================================
# TITLE
# ============================================================

st.title(
    "🧴 Team Thai AI Market Intelligence"
)

st.markdown(
    """
    **AI-powered liquid detergent market analysis**

    Analyze market structure, competitors, pricing, products,
    reviews, positioning, opportunities and AI-generated strategy.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧭 Navigation")

pages = [
    "🏠 Executive Summary",
    "📊 Dashboard",
    "🏢 Team Thai",
    "🏆 Competitors",
    "💰 Pricing",
    "📦 Pack Size",
    "⭐ Reviews & Ratings",
    "🧴 Product Positioning",
    "🛒 Marketplace",
    "🚀 AI Recommendations",
    "📑 Reports",
    "💬 Ask AI"
]

if "selected_page" not in st.session_state:

    st.session_state["selected_page"] = pages[0]

page = st.sidebar.radio(
    "Go to",
    pages,
    index=pages.index(
        st.session_state["selected_page"]
    )
)

st.session_state["selected_page"] = page

st.sidebar.divider()

# ============================================================
# SIDEBAR MARKET SNAPSHOT
# ============================================================

st.sidebar.subheader("📊 Market Snapshot")

st.sidebar.metric(
    "Liquid Products",
    f"{len(liquid_df):,}"
)

st.sidebar.metric(
    "Brands",
    (
        f"{liquid_df['brand_standardized'].nunique():,}"
        if "brand_standardized" in liquid_df.columns
        else "0"
    )
)

if len(teamthai_df) > 0:

    st.sidebar.metric(
        "Team Thai Products",
        f"{len(teamthai_df):,}"
    )

st.sidebar.divider()

# ============================================================
# SIDEBAR REPORT SHORTCUT
# ============================================================

st.sidebar.subheader("📥 Reports")

st.sidebar.caption(
    "Management-ready PDF report with "
    "key findings, benchmarks, opportunities "
    "and recommendations."
)

if st.sidebar.button(
    "📥 Open Reports",
    use_container_width=True
):

    st.session_state["selected_page"] = "📑 Reports"

    st.rerun()



# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

if page == "🏠 Executive Summary":

    st.header("🏠 Executive Market Summary")

    st.markdown(
        """
        ### Team Thai Liquid Detergent Market Intelligence

        A management-level view of the current liquid detergent
        marketplace, Team Thai's position, competitive pressure,
        and the most important strategic priorities.
        """
    )

    # --------------------------------------------------------
    # MARKET KPIs
    # --------------------------------------------------------

    total_products = len(liquid_df)

    unique_brands = (
        liquid_df["brand_standardized"].nunique()
        if "brand_standardized" in liquid_df.columns
        else 0
    )


    average_price = (
        valid_prices["selling_price"].mean()
        if len(valid_prices) > 0
        else 0
    )

    median_price = (
        valid_prices["selling_price"].median()
        if len(valid_prices) > 0
        else 0
    )

    average_rating = (
        liquid_df["rating"].mean()
        if "rating" in liquid_df.columns
        else 0
    )

    teamthai_products = len(teamthai_df)

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    st.subheader("📊 Market Snapshot")

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric(
        "Liquid Products",
        f"{total_products:,}"
    )

    c2.metric(
        "Brands",
        f"{unique_brands:,}"
    )

    c3.metric(
        "Average Price",
        f"₹{average_price:,.0f}"
    )

    c4.metric(
        "Median Price",
        f"₹{median_price:,.0f}"
    )

    c5.metric(
        "Average Rating",
        f"{average_rating:.2f}"
    )

    c6.metric(
        "Team Thai Products",
        teamthai_products
    )

    st.divider()

    # --------------------------------------------------------
    # TEAM THAI SCORECARD
    # --------------------------------------------------------

    st.subheader("🏢 Team Thai Scorecard")

    if len(teamthai_df) == 0:

        st.warning(
            "No Team Thai products detected in the current dataset."
        )

    else:

        teamthai_price = (
            teamthai_df["selling_price"]
            .mean()
        )

        teamthai_rating = (
            teamthai_df["rating"]
            .mean()
            if "rating" in teamthai_df.columns
            else 0
        )

        teamthai_reviews = (
            teamthai_df["rating_count"]
            .sum()
            if "rating_count" in teamthai_df.columns
            else 0
        )

        market_price = (
            valid_prices["selling_price"]
            .mean()
            if len(valid_prices) > 0
            else 0
        )

        price_difference = (
            ((teamthai_price - market_price)
             / market_price) * 100
            if market_price > 0
            else 0
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Team Thai Price",
            f"₹{teamthai_price:,.0f}"
        )

        c2.metric(
            "Team Thai Rating",
            f"{teamthai_rating:.2f}"
        )

        c3.metric(
            "Team Thai Reviews",
            f"{teamthai_reviews:,}"
        )

        c4.metric(
            "Price vs Market",
            f"{price_difference:+.1f}%"
        )

    st.divider()

    # --------------------------------------------------------
    # STRENGTHS / WEAKNESSES
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader("🟢 Current Strengths")

        strengths = [
            "Competitive selling price",
            "Strong product rating",
            "Front-load positioning",
            "Presence in the liquid detergent category"
        ]

        for item in strengths:

            st.success(item)

    with right:

        st.subheader("🔴 Current Weaknesses")

        weaknesses = [
            "Very low review volume",
            "Only one Team Thai product detected",
            "Limited brand presence in the dataset",
            "Need stronger differentiation versus major brands"
        ]

        for item in weaknesses:

            st.error(item)

    st.divider()

    # --------------------------------------------------------
    # STRATEGIC PRIORITIES
    # --------------------------------------------------------

    st.subheader("🚀 Strategic Priorities")

    priorities = pd.DataFrame(
        {
            "Priority": [
                "Increase review volume",
                "Strengthen product positioning",
                "Improve price / 100ml competitiveness",
                "Expand product portfolio",
                "Monitor major competitors"
            ],
            "Priority Level": [
                "HIGH",
                "HIGH",
                "HIGH",
                "MEDIUM",
                "MEDIUM"
            ],
            "Business Objective": [
                "Build consumer trust",
                "Improve differentiation",
                "Improve value perception",
                "Increase market coverage",
                "Protect competitive position"
            ]
        }
    )

    st.dataframe(
        priorities,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # TOP COMPETITORS
    # --------------------------------------------------------

    st.subheader("🏆 Major Competitive Pressure")

    if "brand_standardized" in competitor_df.columns:

        competitor_summary = (
            competitor_df
            .groupby("brand_standardized")
            .agg(
                products=("product_name", "count"),
                average_price=("selling_price", "mean"),
                average_rating=("rating", "mean"),
                total_reviews=("rating_count", "sum")
            )
            .sort_values(
                "total_reviews",
                ascending=False
            )
            .head(10)
        )

        st.dataframe(
            competitor_summary.round(2),
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------------
    # MANAGEMENT CONCLUSION
    # --------------------------------------------------------

    st.subheader("🎯 Management Conclusion")

    if len(teamthai_df) > 0:

        st.info(
            """
            Team Thai currently has a limited presence in the observed
            liquid detergent marketplace. The detected Dr Wash product
            shows a strong rating and competitive selling price, but its
            very low review volume and limited portfolio reduce its
            ability to compete with established brands.

            The immediate priority should be strengthening credibility,
            improving product positioning and increasing marketplace
            visibility before making major portfolio expansion decisions.
            """
        )

    else:

        st.warning(
            """
            Team Thai products were not detected in the current dataset.
            This should be investigated before using the analysis for
            strategic decision-making.
            """
        )


    st.divider()

    # ============================================================
    # DASHBOARD
    # ============================================================

if page == "📊 Dashboard":

    st.header("📊 Market Overview")

    total_products = len(liquid_df)

    unique_brands = (
        liquid_df["brand_standardized"].nunique()
        if "brand_standardized" in liquid_df.columns
        else 0
    )

    average_price = (
        valid_prices["selling_price"].mean()
        if len(valid_prices) > 0
        else 0
    )

    median_price = (
        valid_prices["selling_price"].median()
        if len(valid_prices) > 0
        else 0
    )

    average_rating = (
        liquid_df["rating"].mean()
        if "rating" in liquid_df.columns
        else 0
    )

    total_reviews = (
        liquid_df["rating_count"].sum()
        if "rating_count" in liquid_df.columns
        else 0
    )

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric(
        "Liquid Products",
        total_products
    )

    col2.metric(
        "Brands",
        unique_brands
    )

    col3.metric(
        "Average Price",
        money(average_price)
    )

    col4.metric(
        "Median Price",
        money(median_price)
    )

    col5.metric(
        "Average Rating",
        f"{average_rating:.2f}"
    )

    col6.metric(
        "Total Reviews",
        number(total_reviews)
    )

    st.divider()

    # ========================================================
    # MARKET SUMMARY
    # ========================================================

    st.subheader("📈 Market Snapshot")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Team Thai Products",
        len(teamthai_df)
    )

    c2.metric(
        "Competitor Products",
        len(competitor_df)
    )

    c3.metric(
        "Lowest Price",
        money(valid_prices["selling_price"].min())
        if len(valid_prices)
        else "₹0"
    )

    c4.metric(
        "Highest Price",
        money(valid_prices["selling_price"].max())
        if len(valid_prices)
        else "₹0"
    )

    st.divider()

    # ========================================================
    # BRAND PORTFOLIO
    # ========================================================

    st.subheader(
        "🏆 Brands by Product Count"
    )

    if "brand_standardized" in liquid_df.columns:

        brand_counts = (
            liquid_df[
                "brand_standardized"
            ]
            .value_counts()
            .head(15)
            .reset_index()
        )

        brand_counts.columns = [
            "brand",
            "products"
        ]

        safe_bar_chart(
            brand_counts,
            x="brand",
            y="products"
        )

    # ========================================================
    # PRICE SEGMENT
    # ========================================================

    st.subheader(
        "💰 Market Price Segmentation"
    )

    if len(valid_prices) > 0:

        segment_df = valid_prices.copy()

        segment_df[
            "price_segment"
        ] = segment_df[
            "selling_price"
        ].apply(price_segment)

        segment_counts = (
            segment_df[
                "price_segment"
            ]
            .value_counts()
            .reindex(
                [
                    "Budget",
                    "Mass Market",
                    "Mid Premium",
                    "Premium"
                ],
                fill_value=0
            )
            .reset_index()
        )

        segment_counts.columns = [
            "segment",
            "products"
        ]

        safe_bar_chart(
            segment_counts,
            x="segment",
            y="products"
        )

    # ========================================================
    # RATING DISTRIBUTION
    # ========================================================

    st.subheader(
        "⭐ Rating Distribution"
    )

    if "rating" in liquid_df.columns:

        rating_data = liquid_df[
            liquid_df["rating"].notna()
        ].copy()

        if len(rating_data) > 0:

            rating_counts = (
                rating_data[
                    "rating"
                ]
                .round(1)
                .value_counts()
                .sort_index()
                .reset_index()
            )

            rating_counts.columns = [
                "rating",
                "products"
            ]

            safe_bar_chart(
                rating_counts,
                x="rating",
                y="products"
            )


# ============================================================
# TEAM THAI
# ============================================================

elif page == "🏢 Team Thai":

    st.header("🏢 Team Thai Position")

    if len(teamthai_df) == 0:

        st.warning(
            "No Team Thai products were detected."
        )

    else:

        # ----------------------------------------------------
        # KPIs
        # ----------------------------------------------------

        teamthai_price = (
            teamthai_df["selling_price"].mean()
            if "selling_price" in teamthai_df.columns
            else 0
        )

        competitor_price = (
            competitor_df["selling_price"].mean()
            if len(competitor_df) > 0
            else 0
        )

        teamthai_rating = (
            teamthai_df["rating"].mean()
            if "rating" in teamthai_df.columns
            else 0
        )

        teamthai_reviews = (
            teamthai_df["rating_count"].sum()
            if "rating_count" in teamthai_df.columns
            else 0
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Team Thai Products",
            len(teamthai_df)
        )

        c2.metric(
            "Average Price",
            money(teamthai_price)
        )

        c3.metric(
            "Average Rating",
            f"{teamthai_rating:.2f}"
        )

        c4.metric(
            "Review Count",
            number(teamthai_reviews)
        )

        st.divider()

        # ----------------------------------------------------
        # BRANDS
        # ----------------------------------------------------

        if "brand_standardized" in teamthai_df.columns:

            brands = (
                teamthai_df[
                    "brand_standardized"
                ]
                .dropna()
                .unique()
                .tolist()
            )

            st.write(
                "**Brands detected:** "
                + (
                    ", ".join(brands)
                    if brands
                    else "None"
                )
            )

        # ----------------------------------------------------
        # PRODUCTS
        # ----------------------------------------------------

        st.subheader(
            "🧴 Team Thai Products"
        )

        display_columns = [
            "brand_standardized",
            "product_name",
            "selling_price",
            "mrp",
            "rating",
            "rating_count",
            "pack_size_ml",
            "price_per_100ml",
            "discount_pct_calculated"
        ]

        display_dataframe(
            teamthai_df,
            display_columns
        )

        st.divider()

        # ----------------------------------------------------
        # PRICE COMPARISON
        # ----------------------------------------------------

        st.subheader(
            "💰 Team Thai vs Competitor Pricing"
        )

        comparison = pd.DataFrame(
            {
                "group": [
                    "Team Thai",
                    "Competitors"
                ],
                "average_price": [
                    teamthai_price,
                    competitor_price
                ]
            }
        )

        safe_bar_chart(
            comparison,
            x="group",
            y="average_price"
        )

        # ----------------------------------------------------
        # PRICE POSITION
        # ----------------------------------------------------

        if teamthai_price > 0:

            market_median = (
                valid_prices[
                    "selling_price"
                ].median()
                if len(valid_prices)
                else 0
            )

            market_average = (
                valid_prices[
                    "selling_price"
                ].mean()
                if len(valid_prices)
                else 0
            )

            st.subheader(
                "📍 Team Thai Price Position"
            )

            if teamthai_price < market_median:

                st.success(
                    "Team Thai is priced below the market median."
                )

            elif teamthai_price < market_average:

                st.info(
                    "Team Thai is below the market average "
                    "but above the market median."
                )

            else:

                st.warning(
                    "Team Thai is priced above the market average."
                )

            p1, p2 = st.columns(2)

            p1.metric(
                "Team Thai Price",
                money(teamthai_price)
            )

            p2.metric(
                "Market Median",
                money(market_median)
            )

        # ----------------------------------------------------
        # RATING VS PRICE
        # ----------------------------------------------------

        st.subheader(
            "⭐ Price vs Rating"
        )

        if (
            "selling_price" in liquid_df.columns
            and "rating" in liquid_df.columns
        ):

            scatter_data = liquid_df[
                [
                    "selling_price",
                    "rating"
                ]
            ].dropna()

            scatter_data = scatter_data[
                scatter_data["selling_price"] > 0
            ]

            if len(scatter_data) > 0:

                st.scatter_chart(
                    scatter_data,
                    x="selling_price",
                    y="rating",
                    height=400
                )


# ============================================================
# COMPETITORS
# ============================================================

elif page == "🏆 Competitors":

    st.header("🏆 Competitor Analysis")

    if len(competitor_df) == 0:

        st.warning(
            "No competitor data available."
        )

    else:

        competitor_summary = (
            competitor_df
            .groupby("brand_standardized")
            .agg(
                products=(
                    "product_name",
                    "count"
                ),
                average_price=(
                    "selling_price",
                    "mean"
                ),
                average_rating=(
                    "rating",
                    "mean"
                ),
                total_reviews=(
                    "rating_count",
                    "sum"
                )
            )
            .sort_values(
                "products",
                ascending=False
            )
        )

        st.subheader(
            "📊 Competitor Portfolio"
        )

        display_dataframe(
            competitor_summary
            .round(2)
            .head(30)
        )

        # ----------------------------------------------------
        # PRODUCT COUNT
        # ----------------------------------------------------

        st.subheader(
            "📦 Largest Competitor Portfolios"
        )

        portfolio_chart = (
            competitor_summary
            .head(15)
            .reset_index()
        )

        portfolio_chart.columns = [
            "brand",
            "products",
            "average_price",
            "average_rating",
            "total_reviews"
        ]

        safe_bar_chart(
            portfolio_chart,
            x="brand",
            y="products"
        )

        # ----------------------------------------------------
        # REVIEW STRENGTH
        # ----------------------------------------------------

        st.subheader(
            "⭐ Competitor Review Strength"
        )

        review_chart = (
            competitor_summary
            .sort_values(
                "total_reviews",
                ascending=False
            )
            .head(15)
            .reset_index()
        )

        review_chart.columns = [
            "brand",
            "products",
            "average_price",
            "average_rating",
            "total_reviews"
        ]

        safe_bar_chart(
            review_chart,
            x="brand",
            y="total_reviews"
        )

        # ----------------------------------------------------
        # PRICE VS RATING
        # ----------------------------------------------------

        st.subheader(
            "💰 Competitor Price vs Rating"
        )

        scatter = (
            competitor_summary
            .reset_index()
        )

        if len(scatter) > 0:

            st.scatter_chart(
                scatter,
                x="average_price",
                y="average_rating",
                height=450
            )


# ============================================================
# PRICING
# ============================================================

elif page == "💰 Pricing":

    st.header("💰 Pricing Analysis")

    if len(valid_prices) == 0:

        st.warning(
            "No valid prices available."
        )

    else:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Minimum Price",
            money(
                valid_prices[
                    "selling_price"
                ].min()
            )
        )

        c2.metric(
            "Median Price",
            money(
                valid_prices[
                    "selling_price"
                ].median()
            )
        )

        c3.metric(
            "Average Price",
            money(
                valid_prices[
                    "selling_price"
                ].mean()
            )
        )

        c4.metric(
            "Maximum Price",
            money(
                valid_prices[
                    "selling_price"
                ].max()
            )
        )

        st.divider()

        # ----------------------------------------------------
        # PRICE SEGMENTS
        # ----------------------------------------------------

        st.subheader(
            "📊 Price Segment Distribution"
        )

        pricing = valid_prices.copy()

        pricing[
            "segment"
        ] = pricing[
            "selling_price"
        ].apply(price_segment)

        segment_counts = (
            pricing[
                "segment"
            ]
            .value_counts()
            .reindex(
                [
                    "Budget",
                    "Mass Market",
                    "Mid Premium",
                    "Premium"
                ],
                fill_value=0
            )
            .reset_index()
        )

        segment_counts.columns = [
            "segment",
            "products"
        ]

        safe_bar_chart(
            segment_counts,
            x="segment",
            y="products"
        )

        # ----------------------------------------------------
        # PRICE HISTOGRAM
        # ----------------------------------------------------

        st.subheader(
            "📈 Price Distribution"
        )

        histogram = valid_prices[
            ["selling_price"]
        ].copy()

        histogram[
            "price_range"
        ] = pd.cut(
            histogram[
                "selling_price"
            ],
            bins=[
                0,
                100,
                200,
                300,
                500,
                750,
                1000,
                1500,
                2000,
                np.inf
            ],
            labels=[
                "₹0–100",
                "₹101–200",
                "₹201–300",
                "₹301–500",
                "₹501–750",
                "₹751–1000",
                "₹1001–1500",
                "₹1501–2000",
                "₹2000+"
            ]
        )

        hist_counts = (
            histogram[
                "price_range"
            ]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        hist_counts.columns = [
            "price_range",
            "products"
        ]

        safe_bar_chart(
            hist_counts,
            x="price_range",
            y="products"
        )

        # ----------------------------------------------------
        # BRAND PRICE
        # ----------------------------------------------------

        if "brand_standardized" in valid_prices.columns:

            st.subheader(
                "🏷️ Average Price by Brand"
            )

            brand_price = (
                valid_prices
                .groupby(
                    "brand_standardized"
                )
                .agg(
                    average_price=(
                        "selling_price",
                        "mean"
                    ),
                    products=(
                        "product_name",
                        "count"
                    )
                )
                .sort_values(
                    "average_price",
                    ascending=False
                )
                .head(20)
                .reset_index()
            )

            safe_bar_chart(
                brand_price,
                x="brand_standardized",
                y="average_price"
            )

        # ----------------------------------------------------
        # TEAM THAI PRICE
        # ----------------------------------------------------

        st.subheader(
            "🏢 Team Thai vs Market"

        )

        if len(teamthai_df) > 0:

            team_price = (
                teamthai_df[
                    "selling_price"
                ].mean()
            )

            market_price = (
                valid_prices[
                    "selling_price"
                ].mean()
            )

            comparison = pd.DataFrame(
                {
                    "group": [
                        "Team Thai",
                        "Market"
                    ],
                    "average_price": [
                        team_price,
                        market_price
                    ]
                }
            )

            safe_bar_chart(
                comparison,
                x="group",
                y="average_price"
            )


# ============================================================
# PACK SIZE
# ============================================================

elif page == "📦 Pack Size":

    st.header("📦 Pack Size Analysis")

    if "pack_size_ml" not in liquid_df.columns:

        st.warning(
            "Pack size data is unavailable."
        )

    else:

        pack_df = liquid_df[
            liquid_df[
                "pack_size_ml"
            ].notna()
            &
            (
                liquid_df[
                    "pack_size_ml"
                ] > 0
            )
        ].copy()

        if len(pack_df) == 0:

            st.warning(
                "No valid pack-size data available."
            )

        else:

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Average Pack",
                f"{pack_df['pack_size_ml'].mean():,.0f} ml"
            )

            c2.metric(
                "Median Pack",
                f"{pack_df['pack_size_ml'].median():,.0f} ml"
            )

            c3.metric(
                "Largest Pack",
                f"{pack_df['pack_size_ml'].max():,.0f} ml"
            )

            st.divider()

            # ------------------------------------------------
            # PACK SIZE DISTRIBUTION
            # ------------------------------------------------

            st.subheader(
                "📊 Pack Size Distribution"
            )

            pack_bins = pd.cut(
                pack_df["pack_size_ml"],
                bins=[
                    0,
                    500,
                    1000,
                    1500,
                    2000,
                    3000,
                    5000,
                    np.inf
                ],
                labels=[
                    "≤500 ml",
                    "501–1000 ml",
                    "1001–1500 ml",
                    "1501–2000 ml",
                    "2001–3000 ml",
                    "3001–5000 ml",
                    "5000+ ml"
                ]
            )

            pack_counts = (
                pack_bins
                .value_counts()
                .sort_index()
                .reset_index()
            )

            pack_counts.columns = [
                "pack_size",
                "products"
            ]

            safe_bar_chart(
                pack_counts,
                x="pack_size",
                y="products"
            )

            # ------------------------------------------------
            # PACK SIZE VS PRICE
            # ------------------------------------------------

            st.subheader(
                "💰 Pack Size vs Selling Price"
            )

            scatter = pack_df[
                [
                    "pack_size_ml",
                    "selling_price"
                ]
            ].dropna()

            scatter = scatter[
                scatter["selling_price"] > 0
            ]

            if len(scatter) > 0:

                st.scatter_chart(
                    scatter,
                    x="pack_size_ml",
                    y="selling_price",
                    height=450
                )

            # ------------------------------------------------
            # PRICE PER 100ML
            # ------------------------------------------------

            if "price_per_100ml" in pack_df.columns:

                st.subheader(
                    "📏 Price per 100 ml"
                )

                normalized = pack_df[
                    [
                        "brand_standardized",
                        "product_name",
                        "pack_size_ml",
                        "selling_price",
                        "price_per_100ml"
                    ]
                ].dropna(
                    subset=[
                        "price_per_100ml"
                    ]
                )

                normalized = normalized.sort_values(
                    "price_per_100ml",
                    ascending=False
                )

                display_dataframe(
                    normalized.head(30)
                )


# ============================================================
# REVIEWS & RATINGS
# ============================================================

elif page == "⭐ Reviews & Ratings":

    st.header("⭐ Reviews & Ratings")

    if "rating_count" not in liquid_df.columns:

        st.warning(
            "Review data is unavailable."
        )

    else:

        products_with_reviews = len(
            liquid_df[
                liquid_df[
                    "rating_count"
                ] > 0
            ]
        )

        products_without_reviews = len(
            liquid_df[
                liquid_df[
                    "rating_count"
                ] == 0
            ]
        )

        total_reviews = liquid_df[
            "rating_count"
        ].sum()

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Products With Reviews",
            products_with_reviews
        )

        c2.metric(
            "Products Without Reviews",
            products_without_reviews
        )

        c3.metric(
            "Total Reviews",
            number(total_reviews)
        )

        st.divider()

        # ----------------------------------------------------
        # RATING DISTRIBUTION
        # ----------------------------------------------------

        st.subheader(
            "⭐ Rating Distribution"
        )

        if "rating" in liquid_df.columns:

            ratings = liquid_df[
                liquid_df["rating"].notna()
            ].copy()

            rating_counts = (
                ratings[
                    "rating"
                ]
                .round(1)
                .value_counts()
                .sort_index()
                .reset_index()
            )

            rating_counts.columns = [
                "rating",
                "products"
            ]

            safe_bar_chart(
                rating_counts,
                x="rating",
                y="products"
            )

        # ----------------------------------------------------
        # BRAND REVIEW VOLUME
        # ----------------------------------------------------

        st.subheader(
            "🏆 Brands With Highest Review Volume"
        )

        review_summary = (
            liquid_df
            .groupby("brand_standardized")
            .agg(
                average_rating=(
                    "rating",
                    "mean"
                ),
                total_reviews=(
                    "rating_count",
                    "sum"
                ),
                products=(
                    "product_name",
                    "count"
                )
            )
            .sort_values(
                "total_reviews",
                ascending=False
            )
            .head(20)
            .reset_index()
        )

        safe_bar_chart(
            review_summary,
            x="brand_standardized",
            y="total_reviews"
        )

        # ----------------------------------------------------
        # REVIEW VS RATING
        # ----------------------------------------------------

        st.subheader(
            "📈 Review Volume vs Rating"
        )

        review_scatter = liquid_df[
            [
                "rating",
                "rating_count"
            ]
        ].dropna()

        if len(review_scatter) > 0:

            review_scatter = review_scatter[
                review_scatter[
                    "rating_count"
                ] > 0
            ]

            st.scatter_chart(
                review_scatter,
                x="rating_count",
                y="rating",
                height=450
            )

        # ----------------------------------------------------
        # TOP REVIEWED
        # ----------------------------------------------------

        st.subheader(
            "🔥 Most Reviewed Products"
        )

        top_reviewed = liquid_df[
            [
                "brand_standardized",
                "product_name",
                "selling_price",
                "rating",
                "rating_count"
            ]
        ].sort_values(
            "rating_count",
            ascending=False
        )

        display_dataframe(
            top_reviewed.head(25)
        )


# ============================================================
# PRODUCT POSITIONING
# ============================================================

elif page == "🧴 Product Positioning":

    st.header("🧴 Product Positioning")

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    st.subheader(
        "🧴 Product Category"
    )

    if "product_category" in df.columns:

        category_counts = (
            df[
                "product_category"
            ]
            .value_counts()
            .reset_index()
        )

        category_counts.columns = [
            "category",
            "products"
        ]

        safe_bar_chart(
            category_counts,
            x="category",
            y="products"
        )

    # --------------------------------------------------------
    # WASH TYPE DETECTION
    # --------------------------------------------------------

    st.subheader(
        "🫧 Wash Type Positioning"
    )

    if "product_name" in liquid_df.columns:

        text = (
            liquid_df[
                "product_name"
            ]
            .fillna("")
            .str.lower()
        )

        wash_data = pd.DataFrame(
            {
                "product_name": liquid_df[
                    "product_name"
                ],
                "front_load": text.str.contains(
                    "front load|front-load",
                    regex=True
                ),
                "top_load": text.str.contains(
                    "top load|top-load",
                    regex=True
                ),
                "hand_wash": text.str.contains(
                    "hand wash|handwash",
                    regex=True
                )
            }
        )

        wash_counts = pd.DataFrame(
            {
                "wash_type": [
                    "Front Load",
                    "Top Load",
                    "Hand Wash"
                ],
                "products": [
                    wash_data["front_load"].sum(),
                    wash_data["top_load"].sum(),
                    wash_data["hand_wash"].sum()
                ]
            }
        )

        safe_bar_chart(
            wash_counts,
            x="wash_type",
            y="products"
        )

    # --------------------------------------------------------
    # PRODUCT TITLE KEYWORDS
    # --------------------------------------------------------

    st.subheader(
        "🔎 Common Product Positioning Keywords"
    )

    if "product_name" in liquid_df.columns:

        keyword_list = [
            "stain",
            "premium",
            "ultra",
            "deep clean",
            "color",
            "fabric",
            "fragrance",
            "fresh",
            "mild",
            "eco",
            "natural",
            "machine",
            "front load",
            "top load"
        ]

        keyword_results = []

        titles = (
            liquid_df[
                "product_name"
            ]
            .fillna("")
            .str.lower()
        )

        for keyword in keyword_list:

            count = titles.str.contains(
                keyword,
                regex=False
            ).sum()

            keyword_results.append(
                {
                    "keyword": keyword,
                    "products": int(count)
                }
            )

        keyword_df = pd.DataFrame(
            keyword_results
        )

        keyword_df = keyword_df[
            keyword_df["products"] > 0
        ].sort_values(
            "products",
            ascending=False
        )

        safe_bar_chart(
            keyword_df,
            x="keyword",
            y="products"
        )


# ============================================================
# MARKETPLACE
# ============================================================

elif page == "🛒 Marketplace":

    st.header("🛒 Marketplace Analysis")

    marketplace_column = None

    for candidate in [
        "marketplace",
        "market_place",
        "source"
    ]:

        if candidate in liquid_df.columns:

            marketplace_column = candidate

            break

    if marketplace_column is None:

        st.warning(
            "Marketplace information is unavailable."
        )

    else:

        marketplace_summary = (
            liquid_df
            .groupby(marketplace_column)
            .agg(
                products=(
                    "product_name",
                    "count"
                ),
                average_price=(
                    "selling_price",
                    "mean"
                ),
                average_rating=(
                    "rating",
                    "mean"
                ),
                total_reviews=(
                    "rating_count",
                    "sum"
                )
            )
            .sort_values(
                "products",
                ascending=False
            )
            .reset_index()
        )

        st.subheader(
            "📊 Marketplace Product Distribution"
        )

        display_dataframe(
            marketplace_summary.round(2)
        )

        safe_bar_chart(
            marketplace_summary,
            x=marketplace_column,
            y="products"
        )

        st.subheader(
            "💰 Average Price by Marketplace"
        )

        safe_bar_chart(
            marketplace_summary,
            x=marketplace_column,
            y="average_price"
        )

        st.subheader(
            "⭐ Average Rating by Marketplace"
        )

        safe_bar_chart(
            marketplace_summary,
            x=marketplace_column,
            y="average_rating"
        )


# ============================================================
# AI RECOMMENDATIONS
# ============================================================

elif page == "🚀 AI Recommendations":

    st.header(
        "🚀 AI-Generated Market Intelligence"
    )

    if not ai_insights:

        st.warning(
            "AI-generated insights are not available yet."
        )

        st.info(
            "Run AI/insight_engine.py first."
        )

    else:

        # ----------------------------------------------------
        # EXECUTIVE SUMMARY
        # ----------------------------------------------------

        if "executive_summary" in ai_insights:

            st.subheader(
                "📝 Executive Summary"
            )

            st.info(
                ai_insights[
                    "executive_summary"
                ]
            )

        # ----------------------------------------------------
        # MARKET POSITION
        # ----------------------------------------------------

        if "market_position" in ai_insights:

            st.subheader(
                "📊 Market Position"
            )

            st.write(
                ai_insights[
                    "market_position"
                ]
            )

        # ----------------------------------------------------
        # TEAM THAI
        # ----------------------------------------------------

        if "team_thai_assessment" in ai_insights:

            st.subheader(
                "🏢 Team Thai Assessment"
            )

            assessment = ai_insights[
                "team_thai_assessment"
            ]

            if isinstance(
                assessment,
                dict
            ):

                for key, value in assessment.items():

                    st.markdown(
                        f"### {key.replace('_', ' ').title()}"
                    )

                    if isinstance(
                        value,
                        list
                    ):

                        for item in value:

                            st.write(
                                f"• {item}"
                            )

                    else:

                        st.write(value)

            else:

                st.write(assessment)

        # ----------------------------------------------------
        # COMPETITORS
        # ----------------------------------------------------

        if "competitor_insights" in ai_insights:

            st.subheader(
                "🏆 Competitor Insights"
            )

            competitors = ai_insights[
                "competitor_insights"
            ]

            if isinstance(
                competitors,
                list
            ):

                for item in competitors:

                    st.write(
                        f"• {item}"
                    )

            else:

                st.write(competitors)

        # ----------------------------------------------------
        # PRICING
        # ----------------------------------------------------

        if "pricing_insights" in ai_insights:

            st.subheader(
                "💰 Pricing Insights"
            )

            pricing_insights = ai_insights[
                "pricing_insights"
            ]

            if isinstance(
                pricing_insights,
                list
            ):

                for item in pricing_insights:

                    st.write(
                        f"• {item}"
                    )

            else:

                st.write(pricing_insights)

        # ----------------------------------------------------
        # OPPORTUNITIES
        # ----------------------------------------------------

        if "growth_opportunities" in ai_insights:

            st.subheader(
                "🚀 Growth Opportunities"
            )

            opportunities = ai_insights[
                "growth_opportunities"
            ]

            if isinstance(
                opportunities,
                list
            ):

                for item in opportunities:

                    st.write(
                        f"• {item}"
                    )

            else:

                st.write(opportunities)

        # ----------------------------------------------------
        # RECOMMENDATIONS
        # ----------------------------------------------------

        if "recommendations" in ai_insights:

            st.subheader(
                "🎯 Recommendations"
            )

            recommendations = ai_insights[
                "recommendations"
            ]

            if isinstance(
                recommendations,
                list
            ):

                for item in recommendations:

                    st.write(
                        f"• {item}"
                    )

            else:

                st.write(recommendations)

        # ----------------------------------------------------
        # DATA GAPS
        # ----------------------------------------------------

        if "data_gaps" in ai_insights:

            st.subheader(
                "📥 Data Gaps"
            )

            gaps = ai_insights[
                "data_gaps"
            ]

            if isinstance(
                gaps,
                list
            ):

                for item in gaps:

                    st.write(
                        f"• {item}"
                    )

            else:

                st.write(gaps)



# ============================================================
# REPORTS
# ============================================================

elif page == "📑 Reports":

    st.header("📑 Management Reports")

    st.markdown(
        """
        ### Team Thai Liquid Detergent Market Report

        Generate a management-ready report that converts the
        marketplace analysis into business findings and strategic
        recommendations.

        The report focuses on **what management should know and do**,
        rather than repeating the charts already available throughout
        the dashboard.
        """
    )

    st.divider()

    # ========================================================
    # REPORT CONTENT
    # ========================================================

    st.subheader("📋 What's Included")

    report_items = [
        (
            "📝 Executive Summary",
            "Key findings from the liquid detergent marketplace."
        ),
        (
            "📊 Market Snapshot",
            "Market size within the observed dataset, pricing, brands, ratings and review volume."
        ),
        (
            "🏢 Team Thai Benchmark",
            "Team Thai pricing, ratings, reviews and position versus the observed market."
        ),
        (
            "🏆 Competitive Landscape",
            "Major competitor brands, portfolio size, pricing and review strength."
        ),
        (
            "🚀 Opportunity Areas",
            "Potential areas for differentiation and growth based on observed data."
        ),
        (
            "🎯 Strategic Recommendations",
            "Prioritized actions across product, pricing, positioning and marketplace strategy."
        ),
        (
            "⚠️ Data Limitations",
            "Important limitations that management should consider before making decisions."
        )
    ]

    for title, description in report_items:

        col1, col2 = st.columns([2, 5])

        with col1:

            st.markdown(
                f"**{title}**"
            )

        with col2:

            st.write(
                description
            )

    st.divider()

    # ========================================================
    # REPORT PREVIEW
    # ========================================================

    st.subheader("👀 Report Preview")

    preview_col1, preview_col2, preview_col3 = st.columns(3)

    with preview_col1:

        st.metric(
            "Market Products",
            f"{len(liquid_df):,}"
        )

    with preview_col2:

        st.metric(
            "Brands",
            (
                f"{liquid_df['brand_standardized'].nunique():,}"
                if "brand_standardized"
                in liquid_df.columns
                else "0"
            )
        )

    with preview_col3:

        st.metric(
            "Team Thai Products",
            f"{len(teamthai_df):,}"
        )

    st.divider()

    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.subheader("📥 Download")

    st.write(
        """
        Download the complete management report as a PDF.
        """
    )

    try:

        report_pdf = create_management_report()

        st.download_button(
            label="📥 Download Management Report (PDF)",
            data=report_pdf,
            file_name=(
                "team_thai_liquid_detergent_"
                "management_report.pdf"
            ),
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

        st.success(
            "Your management report is ready."
        )

    except Exception as e:

        st.error(
            "Unable to generate the PDF report."
        )

        st.code(
            str(e)
        )


# ============================================================
# ASK AI
# ============================================================

elif page == "💬 Ask AI":

    st.header(
        "💬 Ask Team Thai AI"
    )

    st.write(
        """
        Ask questions about the liquid detergent market,
        competitors, pricing, reviews, positioning,
        products and growth opportunities.
        """
    )

    question = st.text_area(
        "Your question",
        placeholder=(
            "Example: What should Team Thai do to compete "
            "with Ariel and Surf Excel?"
        ),
        height=130
    )

    if st.button(
        "🤖 Ask AI",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        elif not GROQ_API_KEY:

            st.error(
                "GROQ_API_KEY was not found in .env"
            )

        else:

            with st.spinner(
                "AI is analysing the market..."
            ):

                try:

                    client = Groq(
                        api_key=GROQ_API_KEY
                    )

                    # Use compact AI context.
                    context = json.dumps(
                        ai_context,
                        ensure_ascii=False
                    )

                    # Extra protection against accidentally
                    # sending an enormous context.
                    if len(context) > 30000:

                        context = context[
                            :30000
                        ]

                    prompt = f"""
You are the Team Thai Market Intelligence AI.

Answer the user's question using ONLY the
market data supplied below.

Do not invent sales, market share,
products, companies or facts.

If the available data is insufficient,
clearly say so.

Give practical business-oriented answers.

MARKET DATA:

{context}

USER QUESTION:

{question}

Provide a concise but useful answer.
"""

                    response = client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a market "
                                    "intelligence analyst."
                                )
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=0.2,
                        max_tokens=1200
                    )

                    answer = (
                        response
                        .choices[0]
                        .message
                        .content
                    )

                    st.success(
                        "AI Analysis"
                    )

                    st.markdown(
                        answer
                    )

                except Exception as e:

                    st.error(
                        "AI request failed."
                    )

                    st.code(
                        str(e)
                    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Team Thai AI Market Intelligence"
)

st.sidebar.caption(
    "Data-driven • AI-powered • Market-focused"
)