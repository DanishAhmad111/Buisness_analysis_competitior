import streamlit as st
from exa_py import Exa
from agno.agent import Agent
from agno.tools.firecrawl import FirecrawlTools
from langchain_groq import ChatGroq as GroqChat
from agno.tools.duckduckgo import DuckDuckGoTools
import pandas as pd
import requests
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import time
from functools import lru_cache

# Streamlit UI
st.set_page_config(page_title="AI Competitor Intelligence Agent Team", layout="wide")

# Sidebar for API keys
st.sidebar.title("API Keys")
groq_api_key = st.sidebar.text_input("Chat Groq API Key", type="password")
firecrawl_api_key = st.sidebar.text_input("Firecrawl API Key", type="password")
exa_api_key = st.sidebar.text_input("Exa API Key", type="password")

# Store API keys in session state
if groq_api_key and firecrawl_api_key and exa_api_key:
    st.session_state.groq_api_key = groq_api_key
    st.session_state.firecrawl_api_key = firecrawl_api_key
    st.session_state.exa_api_key = exa_api_key
else:
    st.sidebar.warning("Please enter all required API keys to proceed.")

# Main UI
st.title("🧲 AI Competitor Intelligence Agent Team")
st.info(
    """
    This app helps businesses analyze their competitors by extracting structured data from competitor websites and generating insights using AI.
    - Provide a **URL** or a **description** of your company.
    - The app will fetch competitor URLs, extract relevant information, and generate a detailed analysis report.
    """
)
st.success("For better results, provide both URL and a 5-6 word description of your company!")

# Input fields for URL and description
url = st.text_input("Enter your company URL :")
description = st.text_area("Enter a description of your company (if URL is not available):")

class CompetitorDataSchema(BaseModel):
    company_name: str = Field(description="Name of the company")
    pricing: str = Field(description="Pricing details, tiers, and plans")
    key_features: List[str] = Field(description="Main features and capabilities of the product/service")
    tech_stack: List[str] = Field(description="Technologies, frameworks, and tools used")
    marketing_focus: str = Field(description="Main marketing angles and target audience")
    customer_feedback: str = Field(description="Customer testimonials, reviews, and feedback")

def get_competitor_urls(url: str = None, description: str = None, exa=None) -> list[str]:
    if not url and not description:
        raise ValueError("Please provide either a URL or a description.")
    # Use Exa AI for competitor search
    try:
        # Extract company name from URL or use description
        company_name = ""
        if url:
            # Extract domain name without TLD
            import re
            domain_match = re.search(r'(?:https?://)?(?:www\.)?([\w-]+)(?:\.[\w-]+)+', url)
            if domain_match:
                company_name = domain_match.group(1)
        
        # Construct a query specifically for competitors
        query = f"top competitors of {company_name if company_name else description}"
        st.info(f"Searching for: {query}")
        
        result = exa.search(
            query,
            type="neural",
            category="company",
            use_autoprompt=True,
            num_results=3
        )
        
        # Filter out any URLs that might be from the original company
        competitor_urls = []
        for item in result.results:
            # Skip if the URL contains the original company name
            if company_name and company_name.lower() in item.url.lower():
                continue
            competitor_urls.append(item.url)
        
        # If we filtered out too many, get more results
        if len(competitor_urls) < 3 and company_name:
            additional_results = exa.search(
                f"competitors of {company_name} company",
                type="neural",
                category="company",
                use_autoprompt=True,
                num_results=5
            )
            
            for item in additional_results.results:
                if company_name and company_name.lower() in item.url.lower():
                    continue
                if item.url not in competitor_urls:
                    competitor_urls.append(item.url)
                if len(competitor_urls) >= 3:
                    break
        
        return competitor_urls[:3]  # Return up to 3 competitor URLs
    except Exception as e:
        st.error(f"Error fetching competitor URLs from Exa: {str(e)}")
        return []

def extract_competitor_info(competitor_url: str, firecrawl_api_key=None) -> Optional[dict]:
    # Check if we have this URL in our session cache
    cache_key = f"{competitor_url}_{firecrawl_api_key}"
    if 'cache_data' in st.session_state and cache_key in st.session_state.cache_data:
        return st.session_state.cache_data[cache_key]
    try:
        # Initialize FirecrawlApp with API key
        app = FirecrawlApp(api_key=firecrawl_api_key)
        # Limit to exactly 5 main pages for faster scraping
        url_patterns = [competitor_url]  # Homepage
        # Add only the most important subpages
        main_pages = ["/about", "/pricing", "/features", "/products"]
        for subpage in main_pages:
            url_patterns.append(competitor_url.rstrip("/") + subpage)
        url_patterns = url_patterns[:5]  # Ensure max 5 URLs
        # Optimized extraction prompt for faster and more focused extraction
        extraction_prompt = """
        Extract ONLY the following key information about the company (be concise and specific):
        - Company name
        - Pricing: List the main pricing tiers and costs
        - Key features: List 3-5 main product capabilities
        - Tech stack: List any technologies mentioned
        - Marketing focus: Main target audience and value proposition
        - Customer feedback and testimonials
        Analyze the provided website pages to provide comprehensive information for each field.
        """
        response = app.extract(
            url_patterns,
            {
                'prompt': extraction_prompt,
                'schema': CompetitorDataSchema.model_json_schema(),
            }
        )
        if response.get('success') and response.get('data'):
            extracted_info = response['data']
            # Create JSON structure
            competitor_json = {
                "competitor_url": competitor_url,
                "company_name": extracted_info.get('company_name', 'N/A'),
                "pricing": extracted_info.get('pricing', 'N/A'),
                "key_features": extracted_info.get('key_features', [])[:5],  # Top 5 features
                "tech_stack": extracted_info.get('tech_stack', [])[:5],      # Top 5 tech stack items
                "marketing_focus": extracted_info.get('marketing_focus', 'N/A'),
                "customer_feedback": extracted_info.get('customer_feedback', 'N/A')
            }
            # Store in session state cache
            if 'cache_data' not in st.session_state:
                st.session_state.cache_data = {}
            st.session_state.cache_data[cache_key] = competitor_json
            return competitor_json
        else:
            st.error(f"Error extracting data from {competitor_url}: {response.get('error', 'Unknown error')}")
            return None
    except Exception as e:
        st.warning(f"Exception while extracting data from {competitor_url}: {str(e)}")
        return None

def generate_comparison_report(competitor_data: list, groq_api_key=None) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt
    import streamlit as st
    from io import BytesIO
    # Use direct API call to Groq instead of Agent to avoid pickling issues
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage, SystemMessage

    # Format the competitor data for the prompt
    formatted_data = json.dumps(competitor_data, indent=2)
    print("Comparison Data:", formatted_data)  # For debugging

    try:
        # Create a direct chat client using llama-3.3-70b-versatile model
        chat = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

        # Create the system and human messages
        system_message = SystemMessage(content="You are an expert business analyst who creates detailed comparison tables.")
        human_message = HumanMessage(content=f"""Create a detailed comparison table of the following competitors based on the JSON data:

        {formatted_data}

        Format the comparison as a markdown table with the following columns:
        - Company Name
        - Pricing
        - Key Features
        - Tech Stack
        - Marketing Focus
        - Customer Feedback

        Make sure the table is well-formatted in markdown and includes all competitors.
        Highlight key differences between competitors.
        """)

        # Get the response
        response = chat([system_message, human_message])
        markdown_table = response.content

        # Display the markdown table directly
        st.header("Business Comparison Table")
        st.markdown(markdown_table)

        # --- Business Visualizations ---
        # Convert competitor_data to DataFrame for charts
        df = pd.DataFrame(competitor_data)
        st.header("Key Business Graphs")
        # Feature count bar chart
        if "key_features" in df.columns:
            df["feature_count"] = df["key_features"].apply(lambda x: len(x) if isinstance(x, list) else 0)
            fig1, ax1 = plt.subplots()
            ax1.bar(df["company_name"], df["feature_count"], color="skyblue")
            ax1.set_ylabel("Number of Key Features")
            ax1.set_title("Feature Count by Competitor")
            st.pyplot(fig1)
        # Pricing text chart (if pricing is structured, otherwise show as table)
        if "pricing" in df.columns:
            st.subheader("Pricing Overview")
            st.dataframe(df[["company_name", "pricing"]])

    except Exception as e:
        st.error(f"Error creating comparison table: {str(e)}")
        st.write("Please check your Groq API key and try again.")
        return

    # Use direct API call to Groq instead of Agent to avoid pickling issues
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage, SystemMessage
    
    # Format the competitor data for the prompt
    formatted_data = json.dumps(competitor_data, indent=2)
    print("Comparison Data:", formatted_data)  # For debugging
    
    try:
        # Create a direct chat client using llama-3.3-70b-versatile model
        chat = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
        
        # Create the system and human messages
        system_message = SystemMessage(content="You are an expert business analyst who creates detailed comparison tables.")
        human_message = HumanMessage(content=f"""Create a detailed comparison table of the following competitors based on the JSON data:
        
        {formatted_data}
        
        Format the comparison as a markdown table with the following columns:
        - Company Name
        - Pricing
        - Key Features
        - Tech Stack
        - Marketing Focus
        - Customer Feedback
        
        Make sure the table is well-formatted in markdown and includes all competitors.
        Highlight key differences between competitors.
        """)
        
        # Get the response
        response = chat([system_message, human_message])
        markdown_table = response.content
        
        # Display the markdown table directly
        st.subheader("Competitor Comparison")
        st.markdown(markdown_table)
        
    except Exception as e:
        st.error(f"Error creating comparison table: {str(e)}")
        st.write("Please check your Groq API key and try again.")
        return

def generate_analysis_report(competitor_data: list, groq_api_key=None):
    # Use direct API call to Groq instead of Agent to avoid pickling issues
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage, SystemMessage
    import streamlit as st
    # Format the competitor data for the prompt
    formatted_data = json.dumps(competitor_data, indent=2)
    print("Analysis Data:", formatted_data)  # For debugging

    try:
        # Create a direct chat client using llama-3.3-70b-versatile model
        chat = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

        # Create the system and human messages
        system_message = SystemMessage(content="You are an expert business strategist. Provide only concise, bullet-pointed business insights and actionable recommendations. Avoid narrative or storytelling. Output should be strictly business-oriented.")
        human_message = HumanMessage(content=f"""Analyze the following competitor data in JSON format and identify market opportunities to improve my own company:

        {formatted_data}

        Tasks:
        1. Identify market gaps and opportunities based on competitor offerings
        2. Analyze competitor weaknesses that we can capitalize on
        3. Recommend unique features or capabilities we should develop
        4. Suggest pricing and positioning strategies to gain competitive advantage
        5. Outline specific growth opportunities in underserved market segments
        6. Provide actionable recommendations for product development and go-to-market strategy

        Output format:
        - Use bullet points only
        - Separate sections for: Market Gaps, Competitor Weaknesses, Unique Feature Recommendations, Pricing/Positioning, Growth Opportunities, Actionable Steps
        - No storytelling or narrative, only concise business insights
        """)

        # Get the response
        response = chat([system_message, human_message])
        return response.content

    except Exception as e:
        st.error(f"Error generating analysis report: {str(e)}")
        return "Error generating analysis report. Please check your Groq API key and try again."

    # Use direct API call to Groq instead of Agent to avoid pickling issues
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage, SystemMessage
    
    # Format the competitor data for the prompt
    formatted_data = json.dumps(competitor_data, indent=2)
    print("Analysis Data:", formatted_data)  # For debugging
    
    try:
        # Create a direct chat client using llama-3.3-70b-versatile model
        chat = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
        
        # Create the system and human messages
        system_message = SystemMessage(content="You are an expert business strategist who provides insightful competitive analysis and actionable recommendations.")
        human_message = HumanMessage(content=f"""Analyze the following competitor data in JSON format and identify market opportunities to improve my own company:
        
        {formatted_data}

        Tasks:
        1. Identify market gaps and opportunities based on competitor offerings
        2. Analyze competitor weaknesses that we can capitalize on
        3. Recommend unique features or capabilities we should develop
        4. Suggest pricing and positioning strategies to gain competitive advantage
        5. Outline specific growth opportunities in underserved market segments
        6. Provide actionable recommendations for product development and go-to-market strategy

        Focus on finding opportunities where we can differentiate and do better than competitors.
        Highlight any unmet customer needs or pain points we can address.
        """)
        
        # Get the response
        response = chat([system_message, human_message])
        return response.content
        
    except Exception as e:
        st.error(f"Error generating analysis report: {str(e)}")
        return "Error generating analysis report. Please check your Groq API key and try again."

# Initialize cache for session
if 'cache_data' not in st.session_state:
    st.session_state.cache_data = {}

if "groq_api_key" in st.session_state and "firecrawl_api_key" in st.session_state and "exa_api_key" in st.session_state:
    # Initialize Exa for competitor discovery
    exa = Exa(api_key=st.session_state.exa_api_key)
    if st.button("Analyze Competitors"):
        if url or description:
            try:
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Fetch competitor URLs
                status_text.text("Step 1/3: Discovering competitors...")
                with st.spinner("Fetching competitor URLs..."):
                    competitor_urls = get_competitor_urls(url=url, description=description, exa=exa)
                    
                    if not competitor_urls:
                        st.error("No competitor URLs found. Try a different URL or more specific description.")
                        progress_bar.progress(100)
                        status_text.text("Analysis failed")
                    else:
                        st.write(f"Found {len(competitor_urls)} competitors: {', '.join(competitor_urls)}")
                        progress_bar.progress(33)
                
                # Step 2: Extract competitor data
                status_text.text("Step 2/3: Analyzing competitor websites...")
                competitor_data = []
                total_urls = len(competitor_urls)
                for i, comp_url in enumerate(competitor_urls):
                    try:
                        with st.spinner(f"Analyzing Competitor {i+1}/{total_urls}: {comp_url}..."):
                            start_time = time.time()
                            competitor_info = extract_competitor_info(comp_url, firecrawl_api_key=st.session_state.firecrawl_api_key)
                            elapsed_time = time.time() - start_time
                            st.write(f"✅ Analyzed {comp_url} in {elapsed_time:.1f} seconds")
                            if competitor_info is not None:
                                competitor_data.append(competitor_info)
                    except Exception as e:
                        st.warning(f"Could not analyze {comp_url}: {str(e)}")
                        continue
                
                progress_bar.progress(66)
                status_text.text("Step 3/3: Generating reports...")
                
                if competitor_data:
                    with st.spinner("Generating business comparison table and graphs..."):
                        generate_comparison_report(competitor_data, groq_api_key=st.session_state.groq_api_key)
                    with st.spinner("Generating business insights report..."):
                        analysis_report = generate_analysis_report(competitor_data, groq_api_key=st.session_state.groq_api_key)
                        st.header("Business Insights & Recommendations")
                        st.markdown(analysis_report)
                
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    st.success("✅ Analysis complete! Review the business comparison table, graphs, and insights above.")
                else:
                    progress_bar.progress(100)
                    status_text.text("Analysis failed")
                    st.error("Could not extract data from any competitor URLs. Please try with a different URL or description.")
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
        else:
            st.error("Please provide either a URL or a description.")