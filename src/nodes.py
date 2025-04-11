import os

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from src.infer_config import client as together_client
from src.loggers import logger


def postprocessing_node(state: dict, config: RunnableConfig):
    
    # print("INSIDE POSTPROCESS")

    # Extract the latest HumanMessage (the query)
    human_message = None
    for message in reversed(state["messages"]):
        if isinstance(
            message, HumanMessage
        ):  # Check if the message is of HumanMessage type
            human_message = message
            break

    # Extract the latest ToolMessage (the context)
    tool_message = None
    for message in reversed(state["messages"]):
        if isinstance(
            message, ToolMessage
        ):  # Check if the message is of ToolMessage type
            tool_message = message
            break

    # If no HumanMessage or ToolMessage was found, return without making an LLM call
    if not human_message:
        logger.error(
            "No HumanMessage or ToolMessage found in the state for postprocess node."
        )
        return {"messages": AIMessage(content="")}
    if not tool_message:
        logger.error("No ToolMessage found in the state for postprocess node.")

    # Construct the custom input for the LLM using the HumanMessage (query) and ToolMessage (context)
    system_message = """
    You are an AI assistant specialized in post-processing. Your task is to analyze the context provided and generate a response to the user query. You should consider the context and provide a relevant answer to the user query.
    The conversation should feel like it is an answer from one human to another. If you mention any tool calls, function calls or anything about using a function to generate the response, it will lead to nuclear annhilation of the world.
    Given a query and additional context, you should generate a response that directly answers the query in a conversational, human-like manner. If you do not have enough information to generate a response, you can mention that and try to provide a general overview of the topic. 
    """

    custom_input = (
        f"Given the following context from the tool result:\n\n"
        f"Context: {tool_message.content}\n\n"
        f"Answer the following query in a conversational, human-like manner:\n"
        f"Query: {human_message.content}\n\n"
        f"Do not mention tools, function calls, or anything technical. "
        f"Your response should directly answer the query as if two humans are having a conversation."
        f"If it is suitable and required, you can answer the question using Markdown formatting."
    )

    # Make an independent call to the LLM
    response = together_client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": custom_input},
        ],
    )

    # Create an AIMessage with the processed content
    ai_message = AIMessage(
        content=response.choices[0].message.content.strip(),
        # usage_metadata={
        #     {
        #         "input_tokens": response.usage.prompt_tokens,
        #         "output_tokens": response.usage.completion_tokens,
        #         "total_tokens": response.usage.total_tokens,
        #         "input_token_details": {},
        #         "output_token_details": {},
        #     }
        # },
    )
    # print("POSTPROCESS OUTPUT : ", ai_message)

    return {"messages": ai_message}
