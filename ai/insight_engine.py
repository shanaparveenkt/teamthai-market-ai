from pathlib import Path
import json
import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ANALYSIS_DIR = DATA_DIR / "analysis"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "openai/gpt-oss-20b"

CLEANED_FILE = (
    PROCESSED_DIR / "cleaned_market_data.csv"
)

CONTEXT_FILE = (
    OUTPUT_DIR / "ai_market_context.json"
)

AI_OUTPUT_FILE = (
    OUTPUT_DIR / "ai_generated_insights.json"
)


# ============================================================
# HELPERS
# ============================================================

def title(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def section(text):
    print("\n" + "-" * 70)
    print(text)
    print("-" * 70)


def clean_value(value):

    if pd.isna(value):
        return None

    if isinstance(value, float):

        if pd.isna(value):
            return None

        return round(value, 2)

    text = str(value)

    # Remove very long URLs
    if text.startswith("http"):
        return None

    # Limit unnecessarily long text
    if len(text) > 180:
        text = text[:180] + "..."

    return text


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(filename):

    path = ANALYSIS_DIR / filename

    if not path.exists():

        print(
            f"⚠️ Missing: {filename}"
        )

        return None

    try:

        df = pd.read_csv(path)

        print(
            f"✅ {filename:<45} "
            f"{len(df)} records"
        )

        return df

    except Exception as e:

        print(
            f"⚠️ Error reading {filename}: {e}"
        )

        return None


# ============================================================
# REDUCE DATAFRAME
# ============================================================

def reduce_dataframe(
    df,
    columns,
    rows=5
):

    if df is None or df.empty:
        return []

    available = [
        c for c in columns
        if c in df.columns
    ]

    if not available:
        return []

    temp = df[available].head(rows).copy()

    records = []

    for _, row in temp.iterrows():

        record = {}

        for column in available:

            value = clean_value(
                row[column]
            )

            if value is not None:
                record[column] = value

        records.append(record)

    return records


# ============================================================
# LOAD CLEANED DATA
# ============================================================

def load_cleaned_data():

    if not CLEANED_FILE.exists():

        raise FileNotFoundError(
            f"\nCleaned dataset not found:\n"
            f"{CLEANED_FILE}"
        )

    return pd.read_csv(
        CLEANED_FILE
    )


# ============================================================
# MARKET SUMMARY
# ============================================================

def market_summary(df):

    liquid = df[
        df["product_category"]
        .astype(str)
        .str.lower()
        .eq("liquid detergent")
    ].copy()

    prices = pd.to_numeric(
        liquid["selling_price"],
        errors="coerce"
    )

    ratings = pd.to_numeric(
        liquid["rating"],
        errors="coerce"
    )

    teamthai = liquid[
        liquid["company_group"]
        .astype(str)
        .str.lower()
        .eq("team thai")
    ].copy()

    summary = {

        "liquid_detergent_products":
            int(len(liquid)),

        "unique_brands":
            int(
                liquid[
                    "brand_standardized"
                ]
                .nunique()
            ),

        "average_price":
            round(
                float(prices.mean()),
                2
            ),

        "median_price":
            round(
                float(prices.median()),
                2
            ),

        "minimum_price":
            round(
                float(prices.min()),
                2
            ),

        "maximum_price":
            round(
                float(prices.max()),
                2
            ),

        "average_rating":
            round(
                float(ratings.mean()),
                2
            ),

        "team_thai_products":
            int(len(teamthai)),

        "team_thai_brands":
            sorted(
                teamthai[
                    "brand_standardized"
                ]
                .dropna()
                .unique()
                .tolist()
            )
    }

    return liquid, teamthai, summary


# ============================================================
# TEAM THAI PRODUCTS
# ============================================================

def teamthai_products(teamthai):

    return reduce_dataframe(

        teamthai,

        [
            "brand_standardized",
            "product_name",
            "pack_size",
            "selling_price",
            "mrp",
            "rating",
            "rating_count",
            "price_per_100ml"
        ],

        rows=10
    )


# ============================================================
# COMPETITOR SUMMARY
# ============================================================

def competitor_summary(liquid):

    competitors = liquid[
        liquid["company_group"]
        .astype(str)
        .str.lower()
        .ne("team thai")
    ].copy()

    grouped = (
        competitors
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
        .reset_index()
    )

    grouped["average_price"] = (
        grouped["average_price"]
        .round(2)
    )

    grouped["average_rating"] = (
        grouped["average_rating"]
        .round(2)
    )

    grouped["total_reviews"] = (
        grouped["total_reviews"]
        .astype(int)
    )

    grouped = grouped.sort_values(
        "products",
        ascending=False
    )

    return reduce_dataframe(

        grouped,

        [
            "brand_standardized",
            "products",
            "average_price",
            "average_rating",
            "total_reviews"
        ],

        rows=10
    )


# ============================================================
# LOAD ANALYSIS FILES
# ============================================================

def load_analysis():

    files = {

        "brand_competitors":
            "brand_competitor_analysis.csv",

        "marketplace":
            "marketplace_summary.csv",

        "price_segments":
            "price_segment_analysis.csv",

        "positioning":
            "product_positioning_analysis.csv",

        "wash_type":
            "wash_type_analysis.csv",

        "review_strength":
            "review_strength_analysis.csv",

        "premium_market":
            "premium_market_analysis.csv",

        "opportunities":
            "market_opportunity_signals.csv",

        "top_rated":
            "top_rated_products.csv",

        "top_reviewed":
            "top_review_products.csv",

        "top_discounted":
            "top_discounted_products.csv",

        "lowest_price":
            "lowest_normalized_price_products.csv"
    }

    result = {}

    section(
        "LOADING ANALYSIS MODULES"
    )

    for key, filename in files.items():

        result[key] = load_csv(
            filename
        )

    return result


# ============================================================
# EXTRACT ONLY IMPORTANT INFORMATION
# ============================================================

def extract_analysis(modules):

    data = {}

    # --------------------------------------------------------
    # BRAND COMPETITION
    # --------------------------------------------------------

    data["brand_competitors"] = reduce_dataframe(

        modules["brand_competitors"],

        [
            "brand_standardized",
            "product_count",
            "average_price",
            "average_rating",
            "total_reviews"
        ],

        rows=8
    )

    # --------------------------------------------------------
    # MARKETPLACE
    # --------------------------------------------------------

    data["marketplace"] = reduce_dataframe(

        modules["marketplace"],

        [
            "marketplace",
            "product_count",
            "average_price",
            "average_rating"
        ],

        rows=5
    )

    # --------------------------------------------------------
    # PRICE SEGMENTS
    # --------------------------------------------------------

    data["price_segments"] = reduce_dataframe(

        modules["price_segments"],

        [
            "price_segment",
            "product_count",
            "percentage",
            "average_rating"
        ],

        rows=5
    )

    # --------------------------------------------------------
    # POSITIONING
    # --------------------------------------------------------

    data["positioning"] = reduce_dataframe(

        modules["positioning"],

        [
            "positioning",
            "product_count",
            "percentage"
        ],

        rows=8
    )

    # --------------------------------------------------------
    # WASH TYPE
    # --------------------------------------------------------

    data["wash_type"] = reduce_dataframe(

        modules["wash_type"],

        [
            "wash_type",
            "product_count",
            "percentage"
        ],

        rows=6
    )

    # --------------------------------------------------------
    # REVIEWS
    # --------------------------------------------------------

    data["review_strength"] = reduce_dataframe(

        modules["review_strength"],

        [
            "review_strength",
            "product_count",
            "percentage",
            "average_rating"
        ],

        rows=6
    )

    # --------------------------------------------------------
    # PREMIUM
    # --------------------------------------------------------

    data["premium_market"] = reduce_dataframe(

        modules["premium_market"],

        [
            "brand_standardized",
            "product_name",
            "selling_price",
            "rating",
            "rating_count"
        ],

        rows=5
    )

    # --------------------------------------------------------
    # OPPORTUNITIES
    # --------------------------------------------------------

    data["opportunities"] = reduce_dataframe(

        modules["opportunities"],

        [
            "opportunity",
            "signal",
            "priority"
        ],

        rows=5
    )

    # --------------------------------------------------------
    # TOP RATED
    # --------------------------------------------------------

    data["top_rated"] = reduce_dataframe(

        modules["top_rated"],

        [
            "brand_standardized",
            "product_name",
            "selling_price",
            "rating",
            "rating_count"
        ],

        rows=5
    )

    # --------------------------------------------------------
    # TOP REVIEWED
    # --------------------------------------------------------

    data["top_reviewed"] = reduce_dataframe(

        modules["top_reviewed"],

        [
            "brand_standardized",
            "product_name",
            "selling_price",
            "rating",
            "rating_count"
        ],

        rows=5
    )

    # --------------------------------------------------------
    # TOP DISCOUNTED
    # --------------------------------------------------------

    data["top_discounted"] = reduce_dataframe(

        modules["top_discounted"],

        [
            "brand_standardized",
            "product_name",
            "selling_price",
            "mrp",
            "discount_pct"
        ],

        rows=5
    )

    # --------------------------------------------------------
    # LOWEST NORMALIZED PRICE
    # --------------------------------------------------------

    data["lowest_price"] = reduce_dataframe(

        modules["lowest_price"],

        [
            "brand_standardized",
            "product_name",
            "selling_price",
            "pack_size",
            "price_per_100ml"
        ],

        rows=5
    )

    return data


# ============================================================
# BUILD AI CONTEXT
# ============================================================

def build_context(
    df,
    liquid,
    teamthai,
    summary,
    modules
):

    context = {

        "market": summary,

        "team_thai": {
            "products":
                teamthai_products(
                    teamthai
                )
        },

        "top_competitors":
            competitor_summary(
                liquid
            ),

        "analysis":
            extract_analysis(
                modules
            )
    }

    return context


# ============================================================
# SAVE CONTEXT
# ============================================================

def save_context(context):

    with open(
        CONTEXT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            context,
            file,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(context):

    data = json.dumps(
        context,
        ensure_ascii=False
    )

    prompt = f"""
You are Team Thai's AI Market Intelligence Analyst.

Analyze the available Indian liquid-detergent marketplace data.

IMPORTANT RULES:

1. Use ONLY the supplied data.
2. Do not invent facts.
3. Do not invent Team Thai products.
4. The current dataset contains only the Team Thai products
   detected in the collected marketplace data.
5. Do not claim market share unless explicitly provided.
6. Ratings and review counts represent marketplace signals,
   not total sales.
7. Separate observations from recommendations.
8. Give practical business recommendations.
9. Focus on pricing, competitors, product positioning,
   wash type, reviews, marketplace presence and opportunities.

Return ONLY valid JSON.

Required structure:

{{
  "executive_summary": "...",

  "market_position": [
    "...",
    "...",
    "..."
  ],

  "team_thai_assessment": {{
    "current_position": "...",
    "strengths": ["..."],
    "weaknesses": ["..."],
    "opportunities": ["..."]
  }},

  "competitor_insights": [
    {{
      "competitor": "...",
      "observation": "...",
      "implication": "..."
    }}
  ],

  "pricing_insights": [
    "..."
  ],

  "positioning_insights": [
    "..."
  ],

  "review_insights": [
    "..."
  ],

  "growth_opportunities": [
    {{
      "opportunity": "...",
      "evidence": "...",
      "priority": "High"
    }}
  ],

  "recommendations": [
    {{
      "recommendation": "...",
      "reason": "...",
      "priority": "High"
    }}
  ],

  "data_gaps": [
    "..."
  ]
}}

AVAILABLE MARKET DATA:

{data}
"""

    return prompt


# ============================================================
# CALL GROQ
# ============================================================

def call_groq(prompt):

    client = Groq(
        api_key=GROQ_API_KEY
    )

    print(
        "\n🧠 Connecting to Groq..."
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        "\n🤖 Sending compact market context to AI..."
    )

    print(
        "Please wait..."
    )

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[

            {
                "role": "system",

                "content":
                    "You are a precise business "
                    "market intelligence analyst. "
                    "Never invent data."
            },

            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2,

        max_tokens=3000,

        response_format={
            "type": "json_object"
        }
    )

    return response.choices[0].message.content


# ============================================================
# SAVE AI OUTPUT
# ============================================================

def save_ai_output(response):

    try:

        result = json.loads(
            response
        )

    except json.JSONDecodeError:

        result = {
            "raw_response":
                response
        }

    with open(
        AI_OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False
        )

    return result


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(result):

    title(
        "AI-GENERATED MARKET INTELLIGENCE"
    )

    if "executive_summary" in result:

        section(
            "📝 EXECUTIVE SUMMARY"
        )

        print(
            result["executive_summary"]
        )

    if "market_position" in result:

        section(
            "📊 MARKET POSITION"
        )

        for item in result[
            "market_position"
        ]:

            print(
                f"• {item}"
            )

    if "team_thai_assessment" in result:

        assessment = (
            result[
                "team_thai_assessment"
            ]
        )

        section(
            "🏢 TEAM THAI ASSESSMENT"
        )

        print(
            "\nCurrent position:"
        )

        print(
            assessment.get(
                "current_position",
                ""
            )
        )

        print(
            "\nStrengths:"
        )

        for item in assessment.get(
            "strengths",
            []
        ):

            print(
                f"• {item}"
            )

        print(
            "\nWeaknesses:"
        )

        for item in assessment.get(
            "weaknesses",
            []
        ):

            print(
                f"• {item}"
            )

        print(
            "\nOpportunities:"
        )

        for item in assessment.get(
            "opportunities",
            []
        ):

            print(
                f"• {item}"
            )

    if "competitor_insights" in result:

        section(
            "🏆 COMPETITOR INSIGHTS"
        )

        for item in result[
            "competitor_insights"
        ]:

            print(
                f"\n{item.get('competitor', '')}"
            )

            print(
                f"Observation: "
                f"{item.get('observation', '')}"
            )

            print(
                f"Implication: "
                f"{item.get('implication', '')}"
            )

    if "pricing_insights" in result:

        section(
            "💰 PRICING INSIGHTS"
        )

        for item in result[
            "pricing_insights"
        ]:

            print(
                f"• {item}"
            )

    if "positioning_insights" in result:

        section(
            "🧴 PRODUCT POSITIONING"
        )

        for item in result[
            "positioning_insights"
        ]:

            print(
                f"• {item}"
            )

    if "review_insights" in result:

        section(
            "⭐ CUSTOMER REVIEW INSIGHTS"
        )

        for item in result[
            "review_insights"
        ]:

            print(
                f"• {item}"
            )

    if "growth_opportunities" in result:

        section(
            "🚀 GROWTH OPPORTUNITIES"
        )

        for item in result[
            "growth_opportunities"
        ]:

            print(
                f"\n[{item.get('priority', 'Medium')}] "
                f"{item.get('opportunity', '')}"
            )

            print(
                f"Evidence: "
                f"{item.get('evidence', '')}"
            )

    if "recommendations" in result:

        section(
            "🎯 RECOMMENDATIONS"
        )

        for item in result[
            "recommendations"
        ]:

            print(
                f"\n[{item.get('priority', 'Medium')}] "
                f"{item.get('recommendation', '')}"
            )

            print(
                f"Reason: "
                f"{item.get('reason', '')}"
            )

    if "data_gaps" in result:

        section(
            "📥 DATA GAPS"
        )

        for item in result[
            "data_gaps"
        ]:

            print(
                f"• {item}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    title(
        "TEAM THAI AI INSIGHT ENGINE"
    )

    if not GROQ_API_KEY:

        print(
            "❌ GROQ_API_KEY not found."
        )

        print(
            "\nCheck your .env file."
        )

        return

    print(
        "🔐 Groq API key detected."
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print(
        "\n📂 Loading cleaned dataset..."
    )

    df = load_cleaned_data()

    print(
        f"Rows loaded    : {len(df)}"
    )

    print(
        f"Columns loaded : {len(df.columns)}"
    )

    # --------------------------------------------------------
    # MARKET SUMMARY
    # --------------------------------------------------------

    print(
        "\n🧠 Building compact AI market context..."
    )

    liquid, teamthai, summary = (
        market_summary(df)
    )

    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    modules = load_analysis()

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    context = build_context(
        df,
        liquid,
        teamthai,
        summary,
        modules
    )

    save_context(
        context
    )

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = build_prompt(
        context
    )

    chars = len(prompt)

    estimated_tokens = (
        chars / 4
    )

    section(
        "LLM CONTEXT SIZE"
    )

    print(
        f"Prompt characters : {chars:,}"
    )

    print(
        f"Estimated tokens  : "
        f"{estimated_tokens:,.0f}"
    )

    if estimated_tokens > 7000:

        print(
            "\n❌ Context is still too large."
        )

        print(
            "The AI request will NOT be sent."
        )

        print(
            "Reduce the analysis data further."
        )

        return

    print(
        "\n✅ Context is safely below "
        "the 8,000-token limit."
    )

    # --------------------------------------------------------
    # GROQ
    # --------------------------------------------------------

    try:

        response = call_groq(
            prompt
        )

        result = save_ai_output(
            response
        )

        display_result(
            result
        )

    except Exception as e:

        title(
            "❌ AI ENGINE ERROR"
        )

        print(
            type(e).__name__
            + ": "
            + str(e)
        )

        print(
            "\nYour API key and connection "
            "are working if this is a "
            "token/rate-limit error."
        )

        return

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    title(
        "AI INSIGHT GENERATION COMPLETE"
    )

    print(
        "\n📁 AI Context:"
    )

    print(
        CONTEXT_FILE
    )

    print(
        "\n📁 AI Generated Insights:"
    )

    print(
        AI_OUTPUT_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()