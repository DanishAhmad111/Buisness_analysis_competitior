# 🧲 AI Competitor Intelligence Agent Team

The AI Competitor Intelligence Agent Team is a powerful competitor analysis tool powered by Firecrawl . This app helps businesses analyze their competitors by extracting structured data from competitor websites and generating actionable insights using AI.

## Features

- **Multi-Agent System**
    - **Firecrawl Agent**: Specializes in crawling and summarizing competitor websites
    - **Analysis Agent**: Generates detailed competitive analysis reports
    - **Comparison Agent**: Creates structured comparisons between competitors

- **Competitor Discovery**:
  - Finds similar companies using URL matching with Exa AI 
  - Discovers competitors based on business descriptions
  - Automatically extracts relevant competitor URLs

- **Comprehensive Analysis**:
  - Provides structured analysis reports with:
    - Market gaps and opportunities
    - Competitor weaknesses
    - Recommended features
    - Pricing strategies
    - Growth opportunities
    - Actionable recommendations

- **Interactive Analysis**: Users can input either their company URL or description for analysis

## Requirements

The application requires the following Python libraries:

- `agno`
- `exa-py`
- `streamlit`
- `pandas`
- `firecrawl-py`

You'll also need API keys for:
- ChatGroq
- Firecrawl
- Exa

## Usage

1. Enter your API keys in the sidebar
2. Input either:
   - Your company's website URL
   - A description of your company
3. Click "Analyze Competitors" to generate:
   - Competitor comparison table
   - Detailed analysis report
   - Strategic recommendations
