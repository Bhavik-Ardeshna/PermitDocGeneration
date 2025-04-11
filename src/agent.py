import os
import uuid
from typing import Literal

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_together import ChatTogether
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.tools import reg_hybrid_search, sam_hybrid_search
from src.nodes import postprocessing_node

load_dotenv()


# Define the tools for the agent to use


class Agent:

    def __init__(self, pool):
        self.tools = [sam_hybrid_search, reg_hybrid_search]

        self.tool_node = ToolNode(self.tools)
        self.model = ChatTogether(
            together_api_key=os.getenv("TOGETHER_API_KEY"),
            # api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("INFERENCE_LLM_BASE_URL"),
            model=os.getenv("MODEL_NAME"),
            temperature=0,
            streaming=True,
        ).bind_tools(self.tools)

        # Define a new graph
        self.workflow = StateGraph(MessagesState)

        # Initialize memory to persist state between graph runs
        self.checkpointer = AsyncPostgresSaver(pool)

    # Define the function that determines whether to continue or not
    def should_continue(
        self, state: MessagesState
    ) -> Literal["tools", "postprocess", END]:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        elif (
            "sam_hybrid_search" in last_message.content
            or "reg_hybrid_search" in last_message.content
        ):
            return "postprocess"
        return END

    async def call_model(self, state: MessagesState, config: RunnableConfig):
        messages = state["messages"][-10:]
        response = await self.model.ainvoke(messages, config)
        # We return a list, because this will get added to the existing list
        return {"messages": response}

    def create_agent(self):

        # Define the two nodes we will cycle between
        self.workflow.add_node("agent", self.call_model)
        self.workflow.add_node("tools", self.tool_node)
        self.workflow.add_node("postprocess", postprocessing_node)

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        self.workflow.add_edge(START, "agent")

        # We now add a conditional edge
        self.workflow.add_conditional_edges(
            # First, we define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            # Next, we pass in the function that will determine which node is called next.
            self.should_continue,
        )

        # We now add a normal edge from `tools` to `agent`.
        # This means that after `tools` is called, `agent` node is called next.
        self.workflow.add_edge("tools", "agent")
        self.workflow.add_conditional_edges("postprocess", self.should_continue)

        # Finally, we compile it!
        # This compiles it into a LangChain Runnable
        lang_app = self.workflow.compile(checkpointer=self.checkpointer)
        return lang_app
