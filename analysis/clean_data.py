from pathlib import Path
import pandas as pd
import numpy as np
import re


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""

    text = str(value)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# PRICE CLEANING
# ============================================================

def clean_price(value):
    if pd.isna(value):
        return np.nan

    text = str(value)

    text = text.replace(",", "")
    text = text.replace("₹", "")
    text = text.replace("$", "")

    numbers = re.findall(r"\d+(?:\.\d+)?", text)

    if not numbers:
        return np.nan

    return float(numbers[0])


# ============================================================
# RATING CLEANING
# ============================================================

def clean_rating(value):
    if pd.isna(value):
        return np.nan

    text = str(value)

    match = re.search(r"\d+(?:\.\d+)?", text)

    if not match:
        return np.nan

    rating = float(match.group())

    if 0 <= rating <= 5:
        return rating

    return np.nan


# ============================================================
# REVIEW COUNT
# ============================================================

def clean_rating_count(value):

    if pd.isna(value):
        return 0

    text = str(value).replace(",", "")

    match = re.search(r"\d+", text)

    if not match:
        return 0

    return int(match.group())


# ============================================================
# BRAND STANDARDIZATION
# ============================================================

def standardize_brand(brand, product_name=""):

    brand_text = clean_text(brand).lower()
    product_text = clean_text(product_name).lower()

    combined = f"{brand_text} {product_text}"

    # --------------------------------------------------------
    # TEAM THAI BRANDS
    # --------------------------------------------------------

    teamthai_brands = {
        "dr wash": "Dr Wash",
        "dr.wash": "Dr Wash",
        "drwash": "Dr Wash",

        "vi wash": "Vi Wash",
        "viwash": "Vi Wash",

        "sunplus": "Sunplus",

        "roz": "Roz",

        "iva": "Iva",

        # Useful variations
        "iva liquid": "Iva",
        "sun plus": "Sunplus",
    }

    for key, value in teamthai_brands.items():

        if key in combined:
            return value

    # --------------------------------------------------------
    # COMPETITOR BRANDS
    # --------------------------------------------------------

    brand_map = {

        "ariel": "Ariel",

        "surf excel": "Surf Excel",
        "surfexcel": "Surf Excel",

        "tide": "Tide",

        "henko": "Henko",

        "rin": "Rin",

        "godrej fab": "Godrej Fab",
        "godrej ezee": "Godrej Ezee",

        "ujala": "Ujala",

        "ifb": "IFB",

        "ibf": "IBF",

        "safewash": "Safewash",

        "beco": "Beco",

        "flisko": "Flisko",

        "purela": "Purela",

        "koparo": "Koparo",

        "genteel": "Genteel",

        "presto": "Presto",

        "bosch": "Bosch",

        "k2square": "K2Square",

        "k2 square": "K2Square",

        "born good": "Born Good",

        "bubblenut": "Bubblenut",

        "vanish": "Vanish",

        "ghadi": "Ghadi",

        "mr white": "Mr White",

    }

    for key, value in brand_map.items():

        if key in brand_text:
            return value

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    if not brand_text:
        return "Unknown"

    return brand_text.title()


# ============================================================
# COMPANY CLASSIFICATION
# ============================================================

def classify_company(standardized_brand):

    if pd.isna(standardized_brand):
        return "Competitor"

    teamthai_brands = {
        "Dr Wash",
        "Vi Wash",
        "Sunplus",
        "Roz",
        "Iva",
    }

    if standardized_brand in teamthai_brands:
        return "Team Thai"

    return "Competitor"


# ============================================================
# PRODUCT CATEGORY
# ============================================================

def identify_category(product_name):

    if pd.isna(product_name):
        return "Other Laundry Product"

    text = str(product_name).lower()

    # Exclude non-liquid products first
    if "detergent sheet" in text:
        return "Detergent Sheets"

    if "washing machine cleaner" in text:
        return "Washing Machine Cleaner"

    # Liquid detergent indicators
    liquid_keywords = [
        "liquid detergent",
        "laundry liquid",
        "washing liquid",
        "liquid laundry",
        "liquid wash",
    ]

    for keyword in liquid_keywords:

        if keyword in text:
            return "Liquid Detergent"

    # Detergent + liquid measurement
    if "detergent" in text:

        units = [
            " ml",
            "ml ",
            " l",
            "litre",
            "liter",
            "litres",
            "liters",
        ]

        if any(unit in text for unit in units):
            return "Liquid Detergent"

    return "Other Laundry Product"


# ============================================================
# PACK SIZE EXTRACTION
# ============================================================

def extract_pack_size(value):

    if pd.isna(value):
        return np.nan

    text = str(value).lower()

    # Supports:
    # 500 ml
    # 1 L
    # 5L
    # 2 kg
    # 500 g
    # 10 x 100 ml

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*"
        r"(ml|l|litre|litres|liter|liters|kg|g)\b",
        text
    )

    if not match:
        return np.nan

    number = float(match.group(1))
    unit = match.group(2)

    if unit in [
        "l",
        "litre",
        "litres",
        "liter",
        "liters"
    ]:
        return number * 1000

    if unit == "kg":
        return number * 1000

    if unit == "g":
        return number

    return number


# ============================================================
# MAIN CLEANING PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("TEAM THAI MARKET DATA CLEANING")
    print("=" * 70)

    source_file = RAW_DIR / "marketplace_products.csv"

    if not source_file.exists():

        raise FileNotFoundError(
            f"\nMarketplace dataset not found:\n{source_file}\n"
        )

    print("\n📂 Loading marketplace dataset...")

    df = pd.read_csv(source_file)

    original_rows = len(df)

    print(f"Rows loaded    : {len(df)}")
    print(f"Columns loaded : {len(df.columns)}")

    # ========================================================
    # TEXT CLEANING
    # ========================================================

    print("\n🧹 Cleaning text fields...")

    text_columns = [
        "brand",
        "product_name",
        "marketplace",
        "source",
    ]

    for column in text_columns:

        if column in df.columns:

            df[column] = df[column].apply(clean_text)

    # ========================================================
    # BRAND STANDARDIZATION
    # ========================================================

    print("\n🏷️ Standardizing brands...")

    if "brand" in df.columns:

        product_column = (
            df["product_name"]
            if "product_name" in df.columns
            else pd.Series("", index=df.index)
        )

        df["brand_standardized"] = [
            standardize_brand(
                brand,
                product
            )
            for brand, product
            in zip(df["brand"], product_column)
        ]

        df["company_group"] = (
            df["brand_standardized"]
            .apply(classify_company)
        )

    # ========================================================
    # PRICE
    # ========================================================

    print("\n💰 Cleaning prices...")

    if "selling_price" in df.columns:

        df["selling_price"] = (
            df["selling_price"]
            .apply(clean_price)
        )

    if "mrp" in df.columns:

        df["mrp"] = (
            df["mrp"]
            .apply(clean_price)
        )

    # ========================================================
    # DISCOUNT
    # ========================================================

    print("\n🏷️ Recalculating discounts...")

    if (
        "selling_price" in df.columns
        and "mrp" in df.columns
    ):

        valid_discount = (
            (df["mrp"] > 0)
            &
            (df["selling_price"] >= 0)
            &
            (df["selling_price"] <= df["mrp"])
        )

        df["discount_pct_calculated"] = np.where(
            valid_discount,
            (
                (df["mrp"] - df["selling_price"])
                / df["mrp"]
            ) * 100,
            np.nan
        )

    # ========================================================
    # RATINGS
    # ========================================================

    print("\n⭐ Cleaning ratings...")

    if "rating" in df.columns:

        df["rating"] = (
            df["rating"]
            .apply(clean_rating)
        )

    if "rating_count" in df.columns:

        df["rating_count"] = (
            df["rating_count"]
            .apply(clean_rating_count)
        )

    # ========================================================
    # PACK SIZE
    # ========================================================

    print("\n📦 Processing pack sizes...")

    if "pack_size" in df.columns:

        df["pack_size_ml"] = (
            df["pack_size"]
            .apply(extract_pack_size)
        )

    elif "product_name" in df.columns:

        df["pack_size_ml"] = (
            df["product_name"]
            .apply(extract_pack_size)
        )

    # ========================================================
    # PRODUCT CATEGORY
    # ========================================================

    print("\n🧴 Identifying product categories...")

    if "product_name" in df.columns:

        df["product_category"] = (
            df["product_name"]
            .apply(identify_category)
        )

    # ========================================================
    # NORMALIZED PRICE
    # ========================================================

    print("\n📏 Calculating normalized price...")

    df["price_per_100ml"] = np.where(

        (
            (df["pack_size_ml"] > 0)
            &
            (df["selling_price"] >= 0)
        ),

        (
            df["selling_price"]
            / df["pack_size_ml"]
        ) * 100,

        np.nan
    )

    # ========================================================
    # DATA QUALITY
    # ========================================================

    print("\n📊 Creating data-quality score...")

    quality_score = pd.Series(
        100.0,
        index=df.index
    )

    quality_columns = [
        "brand",
        "product_name",
        "selling_price",
        "pack_size_ml",
    ]

    for column in quality_columns:

        if column in df.columns:

            missing = (
                df[column]
                .isna()
                |
                (df[column].astype(str).str.strip() == "")
            )

            quality_score -= (
                missing.astype(int) * 10
            )

    df["data_quality_score"] = (
        quality_score.clip(0, 100)
    )

    df["data_quality_category"] = pd.cut(
        df["data_quality_score"],
        bins=[-1, 59, 79, 100],
        labels=[
            "Needs Review",
            "Good",
            "Excellent"
        ]
    )

    # ========================================================
    # DUPLICATE CHECK
    # ========================================================

    print("\n🔁 Checking product duplicates...")

    duplicate_columns = [
        column
        for column in [
            "product_name",
            "marketplace"
        ]
        if column in df.columns
    ]

    if duplicate_columns:

        df["is_duplicate"] = (
            df.duplicated(
                subset=duplicate_columns,
                keep=False
            )
        )

    else:

        df["is_duplicate"] = False

    # ========================================================
    # SAVE
    # ========================================================

    output_file = (
        PROCESSED_DIR
        / "cleaned_market_data.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("CLEANING COMPLETE")
    print("=" * 70)

    print("\n📊 DATASET RESULT")
    print("-" * 70)

    print(f"Original rows : {original_rows}")
    print(f"Cleaned rows  : {len(df)}")
    print(f"Rows removed  : {original_rows - len(df)}")
    print(f"Final columns : {len(df.columns)}")

    if "company_group" in df.columns:

        print("\n🏢 COMPANY GROUP")
        print("-" * 70)

        print(
            df["company_group"]
            .value_counts()
            .to_string()
        )

        if "brand_standardized" in df.columns:

           print("\n🏷️ TEAM THAI PRODUCTS")
        print("-" * 70)

        teamthai = df[
            df["company_group"] == "Team Thai"
        ]

        if len(teamthai) > 0:

            columns_to_show = [
                "brand_standardized",
                "product_name",
                "selling_price"
            ]

            # Add columns only if they actually exist
            optional_columns = [
                "marketplace",
                "market_place",
                "rating",
                "rating_count"
            ]

            for column in optional_columns:
                if column in teamthai.columns:
                    columns_to_show.append(column)

            print(
                teamthai[columns_to_show]
                .to_string(index=False)
            )

        else:

            print("No Team Thai products detected.")

    if "product_category" in df.columns:

        print("\n🧴 PRODUCT CATEGORY")
        print("-" * 70)

        print(
            df["product_category"]
            .value_counts()
            .to_string()
        )

    print("\n📁 OUTPUT FILE")
    print("-" * 70)

    print(output_file)

    print("\nOriginal master/raw dataset was NOT modified.")

    print("\n" + "=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()