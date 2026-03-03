import os
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
from deepagents import create_deep_agent

# --------------------
# MODEL
# --------------------
os.environ["OPENAI_API_KEY"] = "your_key"
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --------------------
# CHECKPOINT STORE
# --------------------
checkpointer = SqliteSaver.from_conn_string("sqlite:///production_runs.db")

# --------------------
# ROLE AGENTS
# --------------------

pm = create_deep_agent(
    model=model,
    system_prompt="You are a Product Manager. Write PRD. Persist to filesystem.",
)

architect = create_deep_agent(
    model=model,
    system_prompt="You are an Architect. Produce system architecture. Persist.",
)

engineer = create_deep_agent(
    model=model,
    system_prompt="You are an Engineer. Produce implementation plan. Persist.",
)

supervisor = create_deep_agent(
    model=model,
    tools=[
        pm.as_tool("pm"),
        architect.as_tool("architect"),
        engineer.as_tool("engineer"),
    ],
    system_prompt="""
    You are Supervisor.
    - Create todo list
    - Delegate sequentially
    - Validate outputs exist
    - Integrate final result
    """,
)

# --------------------
# GRAPH DEFINITION
# --------------------

class AgentState(dict):
    pass

workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor)
workflow.set_entry_point("supervisor")
workflow.add_edge("supervisor", END)

graph = workflow.compile(checkpointer=checkpointer)

# --------------------
# RUN WITH THREAD ID
# --------------------

config = {"configurable": {"thread_id": "run_001"}}

result = graph.invoke(
    {"messages": [{"role": "user", "content": "Build adaptive chess training SaaS"}]},
    config=config,
)

print(result["messages"][-1].content)
