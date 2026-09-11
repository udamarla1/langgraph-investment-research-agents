import stateData
import common

def analysis_agent(state: stateData) -> dict:

   analysis_prompt = f"""You are a senior investment analyst. Analyze the research findings for the companies: Tesla and NVIDIA.
   Provide a comparative financial analysis, risk assessment, and a final investment memo. Use specific numbers, dates, and sources from the research findings. Be concise but thorough.
   Research Findings from Tesla:state['research_tesla']
   Research Findings from NVIDIA:state['research_nvidia']
   """
   response = common.create_llm().invoke([
       stateData.HumanMessage(content=analysis_prompt)
   ])

   return {
       "financial_analysis": response['financial_analysis'],
       "risk_assessment": response['risk_assessment'],
       "investment_memo": response['investment_memo'],
       "time": response['time'],
       "output_length": response['output_length'],
       "approach": response['approach']
   }