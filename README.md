# 🧴 Team Thai AI Market Intelligence

### AI-Powered Liquid Detergent Market Analysis & Strategic Intelligence Platform

> A data-driven market intelligence application designed to analyze the liquid detergent marketplace, evaluate Team Thai's competitive position, understand pricing and consumer engagement, and generate AI-powered strategic recommendations.

---

## 🚀 Live Application

🌐 **Streamlit Dashboard:**  
_Add your deployed Streamlit URL here_

---

## 📌 Project Overview

The **Team Thai AI Market Intelligence** platform transforms raw liquid detergent marketplace data into actionable business insights.

The application combines:

- 📊 Market analytics
- 🏢 Team Thai brand analysis
- 🏆 Competitor intelligence
- 💰 Pricing analysis
- 📦 Pack-size analysis
- ⭐ Ratings & review analysis
- 🧴 Product positioning analysis
- 🛒 Marketplace analysis
- 🤖 Generative AI insights
- 💬 AI-powered market Q&A
- 📄 Downloadable management reports

The goal is to help decision-makers understand **where Team Thai stands in the market, how competitors are positioned, and where the strongest growth opportunities exist.**

---

# 🎯 Business Questions Addressed

The platform is designed to answer questions such as:

### Market

- How large is the observed liquid detergent product landscape?
- Which brands have the strongest marketplace presence?
- What are the dominant price segments?
- What is the typical market price?

### Competition

- Which brands have the largest product portfolios?
- Which competitors have the strongest review volumes?
- How does Team Thai compare with major competitors?
- Where is competitive pressure strongest?

### Pricing

- Is Team Thai priced above or below the market?
- What price segments dominate the marketplace?
- How does price vary by brand?
- What is the price-per-100ml position?

### Consumer Engagement

- Which brands generate the highest review volume?
- Which products have strong ratings?
- Are high-rated products also highly reviewed?
- Where does Team Thai need stronger consumer credibility?

### Strategy

- What are Team Thai's current strengths?
- What are the major weaknesses?
- Which growth opportunities should be prioritized?
- What actions should management consider?

---

# 🧠 Key Features

## 📊 Market Overview

A high-level view of the liquid detergent marketplace including:

- Total products
- Number of brands
- Average price
- Median price
- Average rating
- Total review volume
- Brand product presence
- Price segmentation
- Rating distribution

---

## 🏢 Team Thai Position

Dedicated analysis of Team Thai's current marketplace position.

Includes:

- Team Thai product portfolio
- Average price
- Average rating
- Review volume
- Price comparison with competitors
- Price positioning versus market
- Product-level analysis
- Price vs rating analysis

---

## 🏆 Competitor Intelligence

Competitive benchmarking across major detergent brands.

Analyzes:

- Product portfolio size
- Average selling price
- Average rating
- Total reviews
- Competitive review strength
- Price vs rating relationships

---

## 💰 Pricing Intelligence

Understand how products are positioned across the market.

Includes:

- Minimum price
- Maximum price
- Median price
- Average price
- Price segmentation
- Price distribution
- Brand-level pricing
- Team Thai vs market pricing

### Price Segments

| Segment | Price Range |
|---|---:|
| Budget | < ₹150 |
| Mass Market | ₹150–₹299 |
| Mid Premium | ₹300–₹599 |
| Premium | ₹600+ |

---

## 📦 Pack Size Analysis

Analyze product size and value positioning.

Includes:

- Average pack size
- Median pack size
- Largest pack
- Pack-size distribution
- Pack size vs selling price
- Price per 100ml

This helps identify differences between **headline price** and **true unit economics/value perception**.

---

## ⭐ Reviews & Ratings

Consumer engagement analysis based on available marketplace data.

Includes:

- Products with reviews
- Products without reviews
- Total reviews
- Rating distribution
- Brand review volume
- Review volume vs rating
- Most reviewed products

---

## 🧴 Product Positioning

Analyze how brands position their products through product titles and descriptions.

The application detects positioning themes such as:

- 🫧 Front Load
- 🫧 Top Load
- 🧺 Hand Wash
- ✨ Stain Removal
- 🌸 Fragrance
- 🌿 Natural / Eco
- 💎 Premium
- 🧴 Fabric Care
- 🧼 Deep Cleaning

This provides a view of the **messaging and positioning landscape**.

---

## 🛒 Marketplace Analysis

Where marketplace information is available, the application analyzes:

- Product distribution
- Average price
- Average rating
- Review volume
- Marketplace-level differences

---

# 🤖 AI-Powered Market Intelligence

The platform uses generative AI to convert structured market data into business-oriented insights.

AI analysis covers:

- 📝 Executive summary
- 📊 Market position
- 🏢 Team Thai assessment
- 🏆 Competitor insights
- 💰 Pricing insights
- 🚀 Growth opportunities
- 🎯 Strategic recommendations
- 📥 Data gaps

The AI is instructed to work only with the supplied market context and avoid inventing unsupported market facts.

---

# 💬 Ask Team Thai AI

The interactive **Ask AI** feature allows users to ask questions such as:

> "What should Team Thai do to compete with major detergent brands?"

> "Is Team Thai competitively priced?"

> "Which competitors have the strongest consumer engagement?"

> "What are the biggest opportunities for Team Thai?"

The AI responds using the available market intelligence context.

---

# 📄 Management Reports

The application also provides a downloadable market-analysis report.

The report can be used for:

- Management discussions
- Business presentations
- Strategic planning
- Competitive reviews
- Internal decision-making

---

# 🏗️ Project Architecture

```text
                    ┌─────────────────────┐
                    │   Market Data       │
                    │   CSV Dataset       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Cleaning     │
                    │   clean_data.py     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Market Analysis     │
                    │ + Derived Metrics   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ AI Context          │
                    │ insight_engine.py   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ AI Generated        │
                    │ Market Insights     │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌──────────────────────────────────┐
              │       Streamlit Application      │
              │                                  │
              │ Dashboard • Competitors          │
              │ Pricing • Reviews • Positioning  │
              │ AI Recommendations • Reports     │
              └──────────────────────────────────┘
