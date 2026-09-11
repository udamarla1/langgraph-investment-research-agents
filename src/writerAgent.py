
import common
import stateData


def writer_node(state: stateData) -> dict:
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
    writer_prompt = f"""You are an investment analyst. Based on the following research results, generate a concise and
    informative summary report for investment analysis"""

    response = common.create_llm().invoke([
        stateData.HumanMessage(content=writer_prompt)
    ])

    return {
        "investment_memo": response['summary_report'],
        "time": response['time'],
        "output_length": response['output_length'],
        "approach": response['approach']}