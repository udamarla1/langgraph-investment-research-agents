from langchain_core.messages import HumanMessage

import common


def create_research_agent(company: str):
    def research_node(state: dict) -> dict:
        query = f"Research {company} for investment analysis."
        data = common.create_search_tool().invoke(query)

        research_prompt = f"""You are an investment analyst. Conduct thorough research on the company: {company}.
Provide specific numbers, dates, and sources. Focus on financial performance, market position, and
recent news or developments for {company}.

Search results:
{data}
"""

        research_result = common.create_llm().invoke([
            HumanMessage(content=research_prompt)
        ])
        output = research_result.content

        state_key = f"research_{company.lower()}"
        return {
            state_key: output,
            "current_agent": f"research_{company.lower()}",
        }

    return research_node


company_agents = {
    "TESLA": create_research_agent("TESLA"),
    "NVIDIA": create_research_agent("NVIDIA"),
}
