# LangGraph Investment Research Agents

A Python project that compares investment research workflows using LangChain and LangGraph. It includes:

- A single-agent workflow that researches companies and creates reports.
- A LangGraph workflow with research, analysis, and writer nodes.
- Shared LLM and Tavily client factories in `src/common.py`.
- Tests that use fake clients, so they do not call OpenAI or Tavily.

## Project Structure

```text
.
├── src/
│   ├── common.py          Shared LLM and search-tool factories
│   ├── main.py            Application entry point
│   ├── oneAgent.py        Dependency-injected single-agent workflow
│   ├── researchAgent.py   LangGraph research node factory
│   ├── analysisAgent.py   LangGraph analysis node
│   ├── writerAgent.py     LangGraph writer node
│   ├── supervisorAgent.py Graph definition and compilation
│   └── stateData.py       InvestmentState type definition
├── tests/
│   ├── conftest.py        Adds src/ to the test import path
│   └── test_agents.py     Unit tests with fake LLM and search clients
├── .env                   Local API keys; never commit this file
├── requirements.txt       Python dependencies
└── readme.md              Project documentation
```

## Setup

Use Python 3.11 or newer and run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS and Linux, activation is:

```bash
source .venv/bin/activate
```

The `bin` directory is created automatically by `venv`. Do not create it manually.

In VS Code, select `.venv/bin/python` with `Python: Select Interpreter`.

## Environment Variables

Create a `.env` file in the repository root:

```env
OPENAI_API_KEY=your-openai-key
TAVILY_API_KEY=your-tavily-key
```

Do not commit real API keys. The project loads these values through `python-dotenv`.

## Run Tests

Tests use fake clients and do not spend API credits:

```bash
source .venv/bin/activate
python -m pytest -q
```

The tests cover:

- Injecting and using an LLM and search tool in `SingleAgent`.
- Dictionary and string search results.
- Passing company-specific research into prompts.
- Reading LangChain responses through `.content`.
- Compiling the supervisor graph.

## Run the Application

Run the main entry point from the repository root:

```bash
.venv/bin/python src/main.py
```

Or, after activating the environment:

```bash
python src/main.py
```

This performs live Tavily and OpenAI calls. It runs both the single-agent workflow and the LangGraph workflow, so it may take some time and consume API credits.

To check only that the supervisor graph compiles without running the graph:

```bash
.venv/bin/python src/supervisorAgent.py
```

## Architecture

`src/common.py` is the shared client layer. It creates one `ChatOpenAI` client and one `TavilySearchResults` client for the application.

`src/main.py` is the composition root. It creates the shared dependencies and passes them into the single-agent workflow:

```python
llm = create_llm()
search_tool = create_search_tool()

results = run_single_agent(
    companies=companies,
    llm_instance=llm,
    search_tool_instance=search_tool,
)
```

`SingleAgent` receives those dependencies through its constructor:

```python
agent = SingleAgent(
    company="TESLA",
    llm_instance=llm,
    search_tool_instance=search_tool,
)
```

The agent then owns the behavior:

```python
search_results = self.search_tool.invoke(self.company)
response = self.llm.invoke(messages)
text = response.content
```

This keeps client creation separate from agent behavior and makes the agent easy to test with fake clients.

## LangGraph Flow

The supervisor graph currently runs nodes in this order:

```text
START
  -> research_tesla
  -> research_nvidia
  -> analysis
  -> writer
  -> END
```

Each node returns a dictionary containing updates to the shared `InvestmentState`.

Important LangChain detail: `llm.invoke(...)` returns an `AIMessage`, not a dictionary. Read generated text with:

```python
response.content
```

Only use dictionary indexing when you have explicitly requested a structured dictionary response.

## Troubleshooting

### Import cannot be resolved

Make sure VS Code is using the project environment:

```text
.venv/bin/python
```

Then run `Developer: Reload Window`.

### `ModuleNotFoundError: No module named 'src'`

When running `src/main.py` directly, use imports relative to the `src` directory, such as:

```python
import oneAgent
```

Run the command from the repository root:

```bash
python src/main.py
```

### `AttributeError` on search results

Tavily results can be dictionaries or strings. The agent handles both formats before reading title, URL, and content fields.

### API errors

Confirm both keys are present in `.env`, then verify that the selected Python interpreter is the one where the packages were installed:

```bash
which python
python -m pip show langgraph langchain-openai langchain-community
```
