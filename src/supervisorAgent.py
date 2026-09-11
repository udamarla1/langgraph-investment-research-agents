from langgraph.graph import END, START, StateGraph

from stateData import InvestmentState
import analysisAgent
import researchAgent
import writerAgent


workflow = StateGraph(InvestmentState)

workflow.add_node("research_tesla", researchAgent.create_research_agent("TESLA"))
workflow.add_node("research_nvidia", researchAgent.create_research_agent("NVIDIA"))
workflow.add_node("analysis", analysisAgent.analysis_agent)
workflow.add_node("writer", writerAgent.writer_node)

workflow.add_edge(START, "research_tesla")
workflow.add_edge("research_tesla", "research_nvidia")
workflow.add_edge("research_nvidia", "analysis")
workflow.add_edge("analysis", "writer")
workflow.add_edge("writer", END)

app = workflow.compile()

try:
    # LangGraph can render the workflow as a Mermaid diagram
    graph_image = app.get_graph().draw_mermaid()
    print("📊 Graph Visualization (Mermaid format):")
    print(graph_image)
except Exception as e:
    print(f"Graph visualization not available: {e}")
    print("\nManual visualization:")
    print("START → research_tesla → research_apple → research_nvidia")
    print("  → financial_analyst → risk_assessor → report_writer → END")

