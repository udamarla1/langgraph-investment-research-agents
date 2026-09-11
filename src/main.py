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

#print('test success')

toResearch = ['TESLA', 'NVIDIA', 'MICROSOFT', 'APPLE', 'AMAZON']


research_results = oneAgent.run_single_agent(toResearch)
#print(f"Research results: {research_results}")

eval_prompt = f"""You are a senior investment analyst reviewing a junior analyst's report.
Score this report on a scale of 1-10 for each category. Be brutally honest.

Categories:
1. Research Depth (Are specific numbers, dates, and sources cited?)
2. Financial Analysis (Are metrics compared with actual data, not generic statements?)
3. Risk Assessment (Are risks specific and non-obvious, not just generic disclaimers?)
4. Recommendation Quality (Is the buy/hold/sell backed by the analysis above?)
5. Overall Professionalism (Would you send this to a client?)

For each category, give:
- Score (1-10)
- One specific weakness you found

Report to evaluate:
{research_results['output'][:6000]}
"""

eval_tool = oneAgent.llmBrain.invoke([HumanMessage(content=eval_prompt)])
print(f"Evaluation results: {eval_tool.content}")