
import os
import time

from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv is not installed. Please install it to load environment variables from a .env file.")

tavilyKey = os.getenv("TAVILY_API_KEY") 
api_key = os.getenv("OPENAI_API_KEY")   

search_tool = TavilySearchResults(max_results=5, tavily_api_key=tavilyKey)
llmBrain = ChatOpenAI(
    model_name="gpt-4o",
    openai_api_key=api_key,
    temperature=0.5,
    max_retries=3,
    request_timeout=30,
)


def run_single_agent(companies: list) -> dict:
    start_time = time.time()
    print(f"Running single agent for companies: {companies} at {start_time} seconds")

    all_research = ""
    for company in companies:
        data = search_tool.invoke(company)
        for r in data:
            if isinstance(r, dict):
                title = r.get("title", "")
                url = r.get("url", "")
                content = r.get("content", "")
            else:
                title = ""
                url = ""
                content = str(r)

            print(f"Company: {company}, Title: {title}, URL: {url}")
            all_research += f"- {content[:300]}\n"

    mega_prompt = f"""You are an investment analyst. Based on the research below, produce a
COMPLETE investment analysis report with ALL of the following:
1. FINANCIAL COMPARISON - Side-by-side comparison of revenue, growth, margins
2. RISK ASSESSMENT - Key risks for each company

Companies: {', '.join(companies)}

Research Data:
{all_research[:8000]}

Produce the FULL report now. Be thorough and specific with numbers."""

    response = llmBrain.invoke([HumanMessage(content=mega_prompt)])

    elapsed = time.time() - start_time

    print(f"\n⏱️  Total time: {elapsed:.1f}s")
    print(f"📄 Output length: {len(response.content)} chars")

    return {
        "output": response.content,
        "time": elapsed,
        "output_length": len(response.content),
        "approach": "Single Agent"
    }

    