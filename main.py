import uuid
from typing import Optional
from openai import BadRequestError
from langgraph.errors import GraphRecursionError

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, status, WebSocketDisconnect
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.agent import Agent
from src.utils import verify_user_token
from manager import ConnectionManager
from src.prompts import system_prompt
from src.db.session import pool
from src.db.utils import (
    add_message_pair_to_db,
    check_thread_if_present,
    send_thread_name_to_db,
    updated_regenerated_answer_in_messages_table,
)
from src.loggers import logger
from src.query import query_expansion
from src.tools import get_sources, set_sources
from src.utils import answer_regeneration, chat_name_generator, query_followup


load_dotenv()


# Define the data model
class QueryRequest(BaseModel):
    query: str = Field(description="The user query to be processed.")
    thread_id: Optional[str] = Field(
        default=None, description="Thread id of the conversation"
    )
    user_id: str = Field(description="User_id of the conversation")


class RegenerationQueryRequest(BaseModel):
    query: str = Field(description="The user query to be processed.")
    answer: str = Field(description="The original answer to the query.")
    thread_id: Optional[str] = Field(
        default=None, description="Thread id of the conversation"
    )
    user_id: str = Field(description="User_id of the conversation")
    token: str = Field(description="User token for verification")
    message_id: str = Field(description="Message id of the conversation")


# Initialize FastAPI app
app = FastAPI()

manager = ConnectionManager()

# Creating and compiling the agent
lang_app = Agent(pool).create_agent()


@app.get("/")
async def health_check():
    return JSONResponse(
        status_code=status.HTTP_200_OK, content={"message": "All okay!"}
    )


@app.post("/regenerate_answer")
async def regenerate_answer(request: RegenerationQueryRequest):

    try:
        # regenerated_answer = answer_regeneration(request.query, request.answer)
        if not verify_user_token(request.token, request.user_id):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"message": "Invalid User Token. Please try again."},
            )
        regenerated_answer = answer_regeneration(request.query, request.answer)
        if (
            "reg_hybrid_search" in regenerated_answer
            or "sam_hybrid_search" in regenerated_answer
        ):
            regenerated_answer = "Dear user, it seems like I'm having some trouble answering your question. Would you please try rephrasing your question?"

        message_id = updated_regenerated_answer_in_messages_table(
            regenerated_answer, request.message_id
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "thread_id": request.thread_id,
                "query": request.query,
                "original_answer": request.answer,
                "regenerated_answer": regenerated_answer,
                "message_id": request.message_id,
            },
        )
    except Exception as e:
        logger.error(f"An unknown error occurred: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"An unknown error occurred: {e}. Please try again."},
        )


@app.websocket("/ws/atlas_chat")
async def websocket_process_query(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            token = websocket.query_params.get("token")
            data = eval(data)
            if "thread_id" not in data.keys():
                data["thread_id"] = uuid.uuid4()
                logger.error(
                    "Valid thread_id not sent. Setting random thread_id for the chat request which will be returned with the first response."
                )
            request = QueryRequest(
                query=data["query"],
                thread_id=data["thread_id"],
                user_id=data["user_id"],
            )
            if not verify_user_token(token, request.user_id):
                await websocket.close(
                    status.WS_1008_POLICY_VIOLATION,
                    reason="Invalid User Token. Please try again.",
                )
            thinking_response = {
                "tag": "thinking",
                "message": "Rephrasing your query to make it more detailed...",
                "thread_id": request.thread_id,
            }
            await manager.send_personal_message(thinking_response, websocket)
            expanded_query = query_expansion(request.query)

            if not check_thread_if_present(request.thread_id):
                thinking_response = {
                    "tag": "thinking",
                    "message": "While I think about your query, let's assign a name to this conversation as well so I remember it...",
                    "thread_id": request.thread_id,
                }
                await manager.send_personal_message(thinking_response, websocket)
                thread_name_list = chat_name_generator(request.query, expanded_query)

                send_thread_name_to_db(
                    request.user_id, request.thread_id, thread_name_list[0]
                )
                thinking_response = {
                    "tag": "thinking",
                    "message": f"Aha! I have assigned a name to this conversation. From now on it shall be called: {thread_name_list[0]} Let's continue...",
                    "thread_id": request.thread_id,
                }
                await manager.send_personal_message(thinking_response, websocket)
                logger.info(
                    f"Thread ID {request.thread_id} assigned the title: {thread_name_list[0]}"
                )

            thinking_response = {
                "tag": "thinking",
                "message": "Hmmm... Let me think about this for a moment and see how can I answer it better...",
                "thread_id": request.thread_id,
            }
            await manager.send_personal_message(thinking_response, websocket)

            messages = [
                # SystemMessage(
                #     content=f"You are Atlas, a helpful assistant designed by Binoloop Inc. which takes in a user query and first decides whether to call database searching tools or not for answering the given query, if yes then it answers the query based on the output of the search, if not then it does not call the tools and answers the query on its own. Decide and answer the query and remember to be kind and playful and make sure to provide the answers relevant to the question only. Today's date in YYYY-MM-DD format is: {datetime.now().strftime('%Y-%m-%d')}"
                # ),
                SystemMessage(content=system_prompt),
                HumanMessage(content=expanded_query),
            ]

            # Include the WebSocket in the initial state
            initial_state = {"messages": messages}

            set_sources({"sam_tenders": [], "far_acquisitions": []})

            # await pool.open()

            async for event in lang_app.astream(
                initial_state,
                config={"configurable": {"thread_id": request.thread_id}},
                stream_mode="values",
            ):
                if (
                    isinstance(event["messages"][-1], ToolMessage)
                    and event["messages"][-1].name == "reg_hybrid_search"
                ):
                    tool_response = {
                        "tag": "thinking",
                        "message": "Searching through the FAR regulations for getting more context on your answer...",
                        "thread_id": request.thread_id,
                    }
                    await manager.send_personal_message(tool_response, websocket)

                elif (
                    isinstance(event["messages"][-1], ToolMessage)
                    and event["messages"][-1].name == "sam_hybrid_search"
                ):
                    tool_response = {
                        "tag": "thinking",
                        "message": "Looking through the SAM database for getting more context on your answer...",
                        "thread_id": request.thread_id,
                    }
                    await manager.send_personal_message(tool_response, websocket)

                elif (
                    isinstance(event["messages"][-1], AIMessage)
                    and event["messages"][-1].content
                ):
                    fallback_message = False
                    if (
                        "reg_hybrid_search" in event["messages"][-1].content
                        or "sam_hybrid_search" in event["messages"][-1].content
                    ):
                        fallback_message = True
                        event["messages"][
                            -1
                        ].content = "Dear user, it seems like I'm having some trouble answering your question. Would you please try rephrasing your question?"

                    if fallback_message:
                        thinking_response = {
                            "tag": "thinking",
                            "message": "That's a tough one! Let me think a little bit more and see if I can come up with a better answer...",
                            "thread_id": request.thread_id,
                        }
                        await manager.send_personal_message(
                            thinking_response, websocket
                        )
                    else:
                        final_message_response = event["messages"][-1].content
                        sources_sent = get_sources()
                        thinking_response = {
                            "tag": "thinking",
                            "message": "Almost done... Generating some cool follow-up questions for you...",
                            "thread_id": request.thread_id,
                        }
                        await manager.send_personal_message(
                            thinking_response, websocket
                        )
                        follow_ups = query_followup(expanded_query)
                        other_attributes = {
                            "sources": sources_sent,
                            "follow-ups": follow_ups,
                        }
                        message_id = add_message_pair_to_db(
                            thread_id=request.thread_id,
                            user_id=request.user_id,
                            message_pair=(request.query, final_message_response),
                            other_attributes=other_attributes,
                        )
                        final_response = {
                            "tag": "answer",
                            "message": final_message_response,
                            "sources": sources_sent,
                            "status_code": status.HTTP_200_OK,
                            "follow_ups": follow_ups,
                            "thread_id": request.thread_id,
                            "message_id": message_id,
                        }
                        await manager.send_personal_message(final_response, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        # logger.info(f"Client disconnected with thread_id: {request.thread_id}")

    except GraphRecursionError:
        logger.error(f"Query could not be resolved.")
        manager.disconnect(websocket)
        await websocket.close(
            status.WS_1014_BAD_GATEWAY,
            reason=f"An error occurred in the graph recursion.",
        )
    except BadRequestError as be:
        logger.error(
            f"An error occurred on TogetherAI's request processing the request: {be}"
        )
        manager.disconnect(websocket)
        await websocket.close(
            status.WS_1014_BAD_GATEWAY,
            reason=f"An error occurred while processing the request: {be}. Please try again.",
        )
    except Exception as e:
        logger.error(f"An unknown error occurred: {e}", exc_info=True)
        manager.disconnect(websocket)
        await websocket.close(
            status.WS_1014_BAD_GATEWAY,
            reason=f"An unknown error occurred: {e}. Please try again.",
        )
