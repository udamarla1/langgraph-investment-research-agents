
import common
from langchain_core.messages import HumanMessage
from stateData import InvestmentState


def writer_node(state: InvestmentState) -> dict:
    """
    Writer node that generates a summary report based on the research results.

    Args:
        state (stateData): The current state of the investment analysis.

    Returns:
        dict: A dictionary containing the summary report.
    """
    # Retrieve research results from the state
    research_tesla = state.get('research_tesla', '')
    research_nvidia = state.get('research_nvidia', '')
    investment_memo = state.get('investment_memo', '')
    risk_assessment = state.get('risk_assessment', '')
    financial_analysis = state.get('financial_analysis', '')

    # Combine research results into a single prompt for the writer
    writer_prompt = f"""You are an investment analyst. Based on the following analysis, generate a concise and
informative summary report for investment analysis.

Financial analysis:
{financial_analysis}

Risk assessment:
{risk_assessment}

Investment memo:
{investment_memo}
"""

    response = common.create_llm().invoke([
        HumanMessage(content=writer_prompt)
    ])

    return {
        "investment_memo": response.content,
        "current_agent": "writer",
    }