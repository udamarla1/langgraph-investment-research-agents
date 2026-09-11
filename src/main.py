from langchain_core.messages import HumanMessage

from common import create_llm, create_search_tool
from oneAgent import run_single_agent

#print('test success')

toResearch = ['TESLA', 'NVIDIA', 'MICROSOFT', 'APPLE', 'AMAZON']


llm = create_llm()
search_tool = create_search_tool()

research_results = run_single_agent(
	companies=toResearch,
	llm_instance=llm,
	search_tool_instance=search_tool,
)
#print(f"Research results: {research_results}")
combined_report = "\n\n".join(
	f"{result['company']}:\n{result['output']}"
	for result in research_results
)

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
{combined_report[:6000]}
"""

eval_tool = llm.invoke([HumanMessage(content=eval_prompt)])
print(f"Evaluation results: {eval_tool.content}")