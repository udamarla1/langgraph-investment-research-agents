from types import SimpleNamespace

import analysisAgent
import oneAgent
import writerAgent


class FakeSearchTool:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.results


class FakeLLM:
    def __init__(self, content="test response"):
        self.content = content
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return SimpleNamespace(content=self.content)


def test_single_agent_injects_and_uses_shared_clients():
    search_tool = FakeSearchTool([
        {"title": "Tesla result", "url": "https://example.com", "content": "Revenue grew."},
        "A plain text result",
    ])
    llm = FakeLLM("Tesla report")

    result = oneAgent.SingleAgent("TESLA", llm, search_tool).run()

    assert search_tool.queries == ["TESLA"]
    assert result["company"] == "TESLA"
    assert result["output"] == "Tesla report"
    assert result["output_length"] == len("Tesla report")
    assert "Revenue grew." in llm.messages[0][0].content
    assert "A plain text result" in llm.messages[0][0].content


def test_run_single_agent_creates_one_agent_per_company():
    search_tool = FakeSearchTool([])
    llm = FakeLLM()

    results = oneAgent.run_single_agent(["TESLA", "NVIDIA"], llm, search_tool)

    assert [result["company"] for result in results] == ["TESLA", "NVIDIA"]
    assert search_tool.queries == ["TESLA", "NVIDIA"]


def test_analysis_agent_reads_ai_message_content(monkeypatch):
    llm = FakeLLM("comparative analysis")
    monkeypatch.setattr(analysisAgent.common, "create_llm", lambda: llm)

    result = analysisAgent.analysis_agent({
        "research_tesla": "Tesla findings",
        "research_nvidia": "NVIDIA findings",
    })

    assert result["financial_analysis"] == "comparative analysis"
    assert result["risk_assessment"] == "comparative analysis"
    assert result["investment_memo"] == "comparative analysis"
    prompt = llm.messages[0][0].content
    assert "Tesla findings" in prompt
    assert "NVIDIA findings" in prompt


def test_writer_node_reads_ai_message_content(monkeypatch):
    llm = FakeLLM("summary report")
    monkeypatch.setattr(writerAgent.common, "create_llm", lambda: llm)

    result = writerAgent.writer_node({
        "financial_analysis": "Financial analysis",
        "risk_assessment": "Risk assessment",
        "investment_memo": "Investment memo",
    })

    assert result == {
        "investment_memo": "summary report",
        "current_agent": "writer",
    }
    assert "Financial analysis" in llm.messages[0][0].content


def test_supervisor_graph_compiles():
    import supervisorAgent

    node_names = set(supervisorAgent.app.get_graph().nodes)

    assert {"research_tesla", "research_nvidia", "analysis", "writer"}.issubset(node_names)
