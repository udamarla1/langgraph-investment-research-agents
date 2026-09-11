import json
import os
import time
from typing import Annotated, TypedDict, Literal, List, Optional
from dotenv import load_dotenv
import langchain_core
from pydantic import BaseModel, Field

# LangChain core
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# Tavily search tool for real-time web research
from langchain_community.tools.tavily_search import TavilySearchResults

# LangGraph for multi-agent orchestration
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
import oneAgent

print('test success')

toResearch = ['TESLA', 'NVIDIA', 'MICROSOFT', 'APPLE', 'AMAZON']


research_results = oneAgent.run_single_agent(toResearch)
print(f"Research results: {research_results}")
