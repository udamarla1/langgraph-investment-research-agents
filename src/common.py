import os

from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI

load_dotenv()


def create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model_name="gpt-4o",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.5,
        max_retries=3,
        request_timeout=30,
    )


def create_search_tool() -> TavilySearchResults:
    return TavilySearchResults(
        max_results=5,
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
    )
