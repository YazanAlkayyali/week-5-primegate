import os
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_agent import retrieve_chapter_context, SYSTEM_PROMPT
from lookup_tool import lookup_leave_balance
from IPython.display import Image, display

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


tools=[retrieve_chapter_context, lookup_leave_balance]

model=ChatGroq(model="openai/gpt-oss-20b", api_key=GROQ_API_KEY).bind_tools(tools)

SYSTEM_MESSAGE=SystemMessage(content=SYSTEM_PROMPT)


def agent_node(state: AgentState) -> AgentState:
    """The reasoning node: calls the LLM with the system message + history."""
    response = model.invoke([SYSTEM_MESSAGE] + state["messages"])
    return {"messages": [response]}


tool_node=ToolNode(tools)


def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """Explicit router: makes the retrieve/lookup vs. answer/clarify
    decision visible as graph control flow, instead of it living only
    inside the LLM's system prompt like it does in the Part 1 agent.
    """
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    return "continue"


graph=StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "tools", "end": END},
)
graph.add_edge("tools", "agent")  

app=graph.compile()

png_data = app.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png_data)

print("Graph saved successfully as 'graph.png' in your current directory!")

def ask(user_input: str) -> str:
    """Run one turn through the graph and return the final text reply."""
    result = app.invoke({"messages": [{"role": "user", "content": user_input}]})
    return result["messages"][-1].content