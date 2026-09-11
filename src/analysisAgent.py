import common
from langchain_core.messages import HumanMessage
from stateData import InvestmentState


def analysis_agent(state: InvestmentState) -> dict:

   analysis_prompt = f"""You are a senior investment analyst. Analyze the research findings for the companies: Tesla and NVIDIA.
Provide a comparative financial analysis, risk assessment, and a final investment memo. Use specific numbers, dates, and sources from the research findings. Be concise but thorough.

Research Findings from Tesla:
{state['research_tesla']}

Research Findings from NVIDIA:
{state['research_nvidia']}
"""
   response = common.create_llm().invoke([
       HumanMessage(content=analysis_prompt)
   ])
   analysis = response.content

   return {
       "financial_analysis": analysis,
       "risk_assessment": analysis,
       "investment_memo": analysis,
       "current_agent": "analysis",
   }