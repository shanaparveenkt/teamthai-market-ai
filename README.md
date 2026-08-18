# Team Thai Market Intelligence AI

An AI-powered market intelligence application for analysing
Team Thai's liquid detergent portfolio and its competitive market.

## Architecture

Raw Data
↓
Python Data Cleaning
↓
Market Analysis
↓
Structured Market Intelligence
↓
LLM
↓
AI Insights
↓
Streamlit Dashboard
↓
n8n Automation

## Main Components

### Data

Contains Team Thai and marketplace datasets.

### Analysis

Python scripts for:

- data cleaning
- validation
- competitor analysis
- price analysis
- product analysis
- marketplace analysis

### AI

The AI layer:

- receives structured analysis
- interprets market patterns
- identifies opportunities
- identifies risks
- generates recommendations

### App

Streamlit dashboard for presenting:

- market overview
- Team Thai portfolio
- competitors
- pricing
- customer signals
- AI insights

### n8n

Automation layer for running the workflow automatically.

## Running the project

Install dependencies:

pip install -r requirements.txt

Run cleaning:

python analysis/clean_data.py

Run market analysis:

python analysis/market_analysis.py

Run AI:

python AI/insight_engine.py

Run dashboard:

streamlit run app/app.py