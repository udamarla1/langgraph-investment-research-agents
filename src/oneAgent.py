import time

from langchain_core.messages import HumanMessage


class SingleAgent:
    def __init__(self, company: str, llm_instance, search_tool_instance):
        self.company = company
        self.llm = llm_instance
        self.search_tool = search_tool_instance

    def run(self) -> dict:
        start_time = time.time()
        data = self.search_tool.invoke(self.company)
        research = []

        for result in data:
            if isinstance(result, dict):
                title = result.get("title", "")
                url = result.get("url", "")
                content = result.get("content", "")
            else:
                title = ""
                url = ""
                content = str(result)

            print(f"Company: {self.company}, Title: {title}, URL: {url}")
            research.append(content[:300])

        prompt = f"""You are an investment analyst. Analyze {self.company} using the research below.

Include:
1. Financial performance, including revenue, growth, and margins
2. Market position and recent developments
3. Key investment risks

Research Data:
{chr(10).join(f'- {item}' for item in research)[:8000]}

Use specific numbers and dates where available."""

        response = self.llm.invoke([
            HumanMessage(content=prompt)
        ])
        elapsed = time.time() - start_time

        return {
            "company": self.company,
            "output": response.content,
            "time": elapsed,
            "output_length": len(response.content),
            "approach": "Single Agent",
        }


def run_single_agent(companies: list, llm_instance, search_tool_instance) -> list:
    agents = [
        SingleAgent(company, llm_instance, search_tool_instance)
        for company in companies
    ]
    return [agent.run() for agent in agents]
