from datetime import datetime
import os

import grpc
from dotenv import load_dotenv
from langchain_core.tools import tool
from pymilvus import AnnSearchRequest, Collection, WeightedRanker
from pymilvus.exceptions import MilvusException

from constants import COLLECTION_NAME_REGULATIONS, COLLECTION_NAME_SAM
from src.embedding import Embeddings
from src.exceptions import MilvusDBError
from src.loggers import logger
from src.vec_db.connect import client
from src.infer_config import client as together_client

load_dotenv()

_sources = {"sam_tenders": [], "far_acquisitions": []}


def get_sources():
    global _sources
    return _sources


def set_sources(value):
    global _sources
    _sources = value


@tool
def generate_final_answer(
    query: str, additional_context: str = "NO ADDITIONAL CONTEXT"
) -> str:
    """
    This tool processes the original query provided to the LangGraph along with any additional context
    from other tools, and generates the response to be sent to the user using an LLM.

    If no additional context from FAR or SAM database is required, the model should select this tool directly
    to generate the final answer. This ensures that the internal workings of the system, such as tool calls
    or database queries, are abstracted from the user, providing a seamless response without unnecessary details.

    Args:
        query (str): The original query provided by the user.
        additional_context (str, optional): Additional context or information obtained from other tools
        or data sources, if available. Defaults to an empty string.

    Returns:
        final_answer (str): The final response to be sent to the user


    Examples:
        # Example 1: When no additional context is required from FAR or SAM
        query = "Hello! How are you doing today?"
        final_answer = generate_final_answer(query)


        # Example 2: When additional context is needed from the FAR database
        query = "Explain the policy for subcontractor compliance."
        additional_context = "<output from the reg_hybrid_search tool>"
        final_answer = generate_final_answer(query, additional_context)


        # Example 3: When additional context is provided by another tool
        query = "What are the tender proposals in the construction industry from 8th Oct 2024?"
        additional_context = "<output from the sam_hybrid_search tool>"
        final_answer = generate_final_answer(query, additional_context)

    """
    # Combine query and additional context (if any)

    messages = [
        {
            "role": "system",
            "content": "You are Charlie, a helper to an AI agent which takes in the user query and any additional information obtained from tools and generates a response to be provided to the user. This response should not contain any internal workings or tool calls of the system. The response should try to provide a helpful answer to the user query without any function calls or abstract knowledge about the tools of the system.",
        },
        {
            "role": "user",
            "content": "Following is the provided user query and the additional context (if any):\n User Query: "
            + query
            + "\n Additional Context: "
            + additional_context
            + "\n\nPlease generate the final response to be provided to the user.",
        },
    ]

    try:
        response = together_client.chat.completions.create(
            model=os.getenv("MODEL_NAME"),
            messages=messages,
        )
        final_answer = response.choices[0].message.content

        return final_answer

    except Exception as e:
        logger.error(f"Error in Together API: {e}", exc_info=True)


# Define the tools for the agent to use
@tool
def sam_hybrid_search(query, expr=None):
    """This function performs a search on the SAM.GOV database and returns the context. The SAM.GOV database is a collection of tenders, contracts  and procurement notices.
    It is useful for businesses looking to bid on government contracts posted by various agencies.
    Args:
        query (str): The query to search for in the database.
        expr (str): The expression to filter the search results. This is to be used when you want to filter the search results based on certain dates. Default is None. For example expr = "date_published in ['2024-03-05', '2024-03-06', '2024-03-07']". This will filter the search results to only include tenders published on the 5th, 6th, and 7th of March 2024. The date format is YYYY-MM-DD. The date_published field is in the format YYYY-MM-DD. IT IS NECESSARY TO PROVIDE AN ACTUAL DATE IN THE EXPR string. You are to strictly not leave it blank with just the parameter and an == sign as this will lead to nuclear annhilation. This should only be used if a query mentions any context related to a date. The expr should not use any other parameter other than `date_published`. DO NOT INPUT TODAY's DATE IF THE QUERY DOES NOT ASK FOR IT. Send expr = ["NONE"] for queries not asking for tenders from a specific date.
    Returns:
        str: The context of the search results.
    Examples:
        >>> sam_hybrid_search("construction tenders from 29th January 2023", expr="date_published == '2023-01-29'")
        "Tender Title: Construction of new school building\nRelevant Chunk From Document: The Department of Education is inviting bids for the construction of a new school building in the city of New York. The project involves the construction of a 3-story building with classrooms, offices, and a gymnasium.\nDate Published: 29th January 2023\nDepartment: Department of Education\nPoint of Contact: John Doe\nPlace of Performance: New York City\nResponse Deadline: 30th June 2024\nOrganization Type: Department of Education\nNotice ID: 123456\n..."
        >>> sam_hybrid_search("construction tenders", expr=["NONE"])
        "Tender Title: Construction of new school building\nRelevant Chunk From Document: The Department of Education is inviting bids for the construction of a new school building in the city of New York. The project involves the construction of a 3-story building with classrooms, offices, and a gymnasium.\nDate Published: 31st January 2024\nDepartment: Department of Education\nPoint of Contact: John Doe\nPlace of Performance: New York City\nResponse Deadline: 30th June 2024\nOrganization Type: Department of Education\nNotice ID: 123456\n..."
        >>> sam_hybrid_search("construction tenders from 29th January 2023 to 31st January 2023", expr="date_published in ['2023-01-29', '2023-01-30', '2023-01-31']")
        "Tender Title: Construction of new school building\nRelevant Chunk From Document: The Department of Education is inviting bids for the construction of a new school building in the city of New York. The project involves the construction of a 3-story building with classrooms, offices, and a gymnasium.\nDate Published: 29th January 2023\nDepartment: Department of Education\nPoint of Contact: John Doe\nPlace of Performance: New York City\nResponse Deadline: 30th June 2024\nOrganization Type: Department of Education\nNotice ID: 123456\n..."
    """
    try:

        # Load the collection
        col_chunks = Collection(COLLECTION_NAME_SAM)
        col_chunks.load()
        limit = 10
        embed = Embeddings()

        if expr and "NONE" in expr:
            expr = None
        today_date = datetime.now().strftime("%Y-%m-%d")

        query_dense_embedding, query_sparse_embedding = embed.encode_for_search(query)
        # Setting to None for logic ahead
        search_res = None

        if expr is not None:
            request_1 = AnnSearchRequest(
                query_sparse_embedding,
                "sparse_vector_embeddings",
                {"metric_type": "IP"},
                expr=expr,
                limit=limit,
            )
            request_2 = AnnSearchRequest(
                query_dense_embedding,
                "dense_vector_embeddings",
                {"metric_type": "L2"},
                expr=expr,
                limit=limit,
            )

            # Store these two requests as a list in `reqs`
            reqs = [request_1, request_2]

            rerank = WeightedRanker(0.8, 0.2)
            search_res = col_chunks.hybrid_search(
                reqs,  # List of AnnSearchRequests created in step 1
                rerank,  # Reranking strategy specified in step 2
                limit=limit,  # Number of final search results to return
                filter=expr,
                output_fields=[
                    "notice_id",
                    "text",
                    "date_published",
                    "title",
                    "responseDeadLine",
                    "organizationType",
                    "pointOfContact",
                    "placeOfPerformance",
                    "description",
                    "uiLink",
                    "solicitationNumber",
                    "classificationCode",
                    "fullParentPathName",
                ],
            )
        if (
            expr is None
            or search_res is None
            or len(search_res[0]) == 0
            or "NONE" in expr
        ):
            request_1 = AnnSearchRequest(
                query_sparse_embedding,
                "sparse_vector_embeddings",
                {"metric_type": "IP"},
                limit=limit,
            )
            request_2 = AnnSearchRequest(
                query_dense_embedding,
                "dense_vector_embeddings",
                {"metric_type": "L2"},
                limit=limit,
            )

            # Store these two requests as a list in `reqs`
            reqs = [request_1, request_2]

            rerank = WeightedRanker(0.8, 0.2)

            search_res = col_chunks.hybrid_search(
                reqs,  # List of AnnSearchRequests created in step 1
                rerank=rerank,  # Reranking strategy specified in step 2
                limit=limit,  # Number of final search results to return
                output_fields=[
                    "notice_id",
                    "text",
                    "date_published",
                    "title",
                    "responseDeadLine",
                    "organizationType",
                    "pointOfContact",
                    "placeOfPerformance",
                    "description",
                    "solicitationNumber",
                    "classificationCode",
                    "uiLink",
                    "fullParentPathName",
                ],
            )

        context = "".join(
            [
                (
                    (
                        "Tender Title: " + res.get("title") + "\n"
                        if res.get("title") != str(-1.0)
                        else ""
                    )
                    + (
                        "Relevant Chunk From Document: " + res.get("text") + "\n"
                        if res.get("text") != str(-1.0)
                        else ""
                    )
                    + (
                        "Date Published: " + res.get("date_published") + "\n"
                        if res.get("date_published") != str(-1.0)
                        else ""
                    )
                    + (
                        "Department: " + res.get("fullParentPathName") + "\n"
                        if res.get("fullParentPathName") != str(-1.0)
                        else ""
                    )
                    + (
                        "Point of Contact: " + res.get("pointOfContact") + "\n"
                        if res.get("pointOfContact") != str(-1.0)
                        else ""
                    )
                    + (
                        "Place of Performance: " + res.get("placeOfPerformance") + "\n"
                        if res.get("placeOfPerformance") != str(-1.0)
                        else ""
                    )
                    + (
                        "Response Deadline: " + res.get("responseDeadLine") + "\n"
                        if res.get("responseDeadLine") != str(-1.0)
                        else ""
                    )
                    + (
                        "Organization Type: " + res.get("organizationType") + "\n"
                        if res.get("organizationType") != str(-1.0)
                        else ""
                    )
                    + (
                        "Notice ID: " + res.get("notice_id") + "\n"
                        if res.get("notice_id") != str(-1.0)
                        else ""
                    )
                    + "\n-----------------------------------\n\n"
                )
                for res in search_res[0]
            ]
        )

        sources = get_sources()
        sources["sam_tenders"].extend(
            [res.get("title"), res.get("uiLink")]
            for res in search_res[0]
            if [res.get("title"), res.get("uiLink")] not in sources["sam_tenders"]
        )
        set_sources(sources)

        return context
    except MilvusException as e:
        logger.error(
            f"Milvus Error occured while hybrid search: {e}, switching to general search without expr"
        )
        try:
            embed = Embeddings()
            query_dense_embedding, query_sparse_embedding = embed.encode_for_search(
                query
            )
            request_1 = AnnSearchRequest(
                query_sparse_embedding,
                "sparse_vector_embeddings",
                {"metric_type": "IP"},
                limit=limit,
            )
            request_2 = AnnSearchRequest(
                query_dense_embedding,
                "dense_vector_embeddings",
                {"metric_type": "L2"},
                limit=limit,
            )

            reqs = [request_1, request_2]

            rerank = WeightedRanker(0.8, 0.2)

            search_res = col_chunks.hybrid_search(
                reqs,  # List of AnnSearchRequests created in step 1
                rerank=rerank,  # Reranking strategy specified in step 2
                limit=limit,  # Number of final search results to return
                output_fields=[
                    "notice_id",
                    "text",
                    "date_published",
                    "title",
                    "responseDeadLine",
                    "organizationType",
                    "pointOfContact",
                    "placeOfPerformance",
                    "description",
                    "solicitationNumber",
                    "classificationCode",
                    "fullParentPathName",
                    "uiLink",
                ],
            )
            context = "".join(
                [
                    (
                        (
                            "Tender Title: " + res.get("title") + "\n"
                            if res.get("title") != str(-1.0)
                            else ""
                        )
                        + (
                            "Relevant Chunk From Document: " + res.get("text") + "\n"
                            if res.get("text") != str(-1.0)
                            else ""
                        )
                        + (
                            "Date Published: " + res.get("date_published") + "\n"
                            if res.get("date_published") != str(-1.0)
                            else ""
                        )
                        + (
                            "Department: " + res.get("fullParentPathName") + "\n"
                            if res.get("fullParentPathName") != str(-1.0)
                            else ""
                        )
                        + (
                            "Point of Contact: " + res.get("pointOfContact") + "\n"
                            if res.get("pointOfContact") != str(-1.0)
                            else ""
                        )
                        + (
                            "Place of Performance: "
                            + res.get("placeOfPerformance")
                            + "\n"
                            if res.get("placeOfPerformance") != str(-1.0)
                            else ""
                        )
                        + (
                            "Response Deadline: " + res.get("responseDeadLine") + "\n"
                            if res.get("responseDeadLine") != str(-1.0)
                            else ""
                        )
                        + (
                            "Organization Type: " + res.get("organizationType") + "\n"
                            if res.get("organizationType") != str(-1.0)
                            else ""
                        )
                        + (
                            "Notice ID: " + res.get("notice_id") + "\n"
                            if res.get("notice_id") != str(-1.0)
                            else ""
                        )
                        + "\n-----------------------------------\n\n"
                    )
                    for res in search_res[0]
                ]
            )

            sources = get_sources()
            sources["sam_tenders"].extend(
                [res.get("title"), res.get("uiLink")]
                for res in search_res[0]
                if [res.get("title"), res.get("uiLink")] not in sources["sam_tenders"]
            )
            set_sources(sources)

            return context
        except MilvusException as e:
            logger.error(
                f"Milvus Error occured while general search: {e}", exc_info=True
            )
        except grpc._channel._InactiveRpcError as e:
            logger.error(f"GRPC Error occured while general search: {e}", exc_info=True)
        except ValueError as e:
            logger.error("Value Error occured.", exc_info=True)
        except Exception as e:
            logger.error("Exception in tool", exc_info=True)

    except grpc._channel._InactiveRpcError as e:
        # logging the error
        if e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
            logger.error(f"GRPC Resource Exhausted: {e}", exc_info=True)
        else:
            logger.error(f"GRPC Error occured: {e}", exc_info=True)

        raise MilvusDBError()
    except ValueError as e:
        logger.error("Value Error occured.", exc_info=True)

    except Exception as e:
        logger.error("Exception in Tool", exc_info=True)


@tool
def reg_hybrid_search(query, expr):
    """
    This function performs a search on the Federal Regulations database and returns the context. The Federal Regulations database is a collection of regulations and laws designed to define the regulations and procedures for various government agencies.
    It is useful for businesses looking to understand the regulations that apply to their industry. Currently this function searches the FAR (Federal Acquisition Regulations) database and the DFARS/DFAR (Defense Federal Acquisition Regulations Supplement) database.
    Args:
        query (str): The query to search for in the database.
        expr (str): The expression to filter the search results. This is to be used when you want to filter the search results based on certain specific FAR or DFAR. Default is None. For example expr = "regulation_type == 'FAR' and section_num == '49.100'". This will filter the search results to only include regulations from Part 49 or FAR 49.100.
        limit (int): The number of search results to return.
    Returns:
        str: The context of the search results.
    Examples:
        >>> reg_hybrid_search("What is mentioned in the DFARS 201.105-3 regulation?", expr = "regulation_type == 'DFAR' and section_num == '201.105-3'")
        "Regulation: DFAR : 201.105-3\nSection Detail: 201.105-3 : Policy\nSubPart Detail: 201.105-3 : Policy\nText: (a) The contracting officer shall insert the clause at 252.201-7000, Contracting Officer's Representative, in solicitations and contracts when a contracting officer's representative is to be appointed.\n\n ...."
        >>> reg_hybrid_search("What is mentioned in the FAR Part 49 regulation?", expr = "regulation_type == 'FAR' and part_num == 'Part 49'")
        "Regulation: FAR : 49\nSection Detail: 49 : Termination of Contracts\nSubPart Detail: 49 : Termination of Contracts\nText: (a) General. This part prescribes policies and procedures for the termination of contracts and the settlement of contract terminations, including the recovery of costs.\n\n ...."
        >>> reg_hybrid_search("What is mentioned in the DFARS Part 49 Section 49.100 regulation?", expr = "regulation_type == 'DFAR' and section_num == '49.100'")
        "Regulation: DFAR : 49.100\nSection Detail: 49.100 : Scope of Subpart\nSubPart Detail: 49.100 : Scope of Subpart\nText: This subpart prescribes policies and procedures for the settlement of terminated contracts, including the recovery of costs.\n\n ...."
        >>> reg_hybrid_search("How to become a defense contractor?")
        "Regulation: DFAR : 201.105-3\nSection Detail: 201.105-3 : Policy\nSubPart Detail: 201.105-3 : Policy\nText: (a) The contracting officer shall insert the clause at 252.201-7000, Contracting Officer's Representative, in solicitations and contracts when a contracting officer's representative is to be appointed.\n\n ...."
        >>> reg_hybrid_search("Can you explain the DFARS requirements for specialty metals?", expr = "regulation_type" == "DFAR")
        "Regulation DFAR ...<required context for answer>"
    """

    try:

        # Load the collection
        col_chunks = Collection(COLLECTION_NAME_REGULATIONS)
        col_chunks.load()

        embed = Embeddings()
        query_dense_embedding, query_sparse_embedding = embed.encode_for_search(query)

        search_res = None
        limit = 10

        if expr is not None:
            request_1 = AnnSearchRequest(
                query_sparse_embedding,
                "sparse_vector_embeddings",
                {"metric_type": "IP"},
                expr=expr,
                limit=limit,
            )
            request_2 = AnnSearchRequest(
                query_dense_embedding,
                "dense_vector_embeddings",
                {"metric_type": "L2"},
                expr=expr,
                limit=limit,
            )

            # Store these two requests as a list in `reqs`
            reqs = [request_1, request_2]

            rerank = WeightedRanker(0.8, 0.2)

            search_res = col_chunks.hybrid_search(
                reqs,  # List of AnnSearchRequests created in step 1
                rerank=rerank,  #  ranking strategy specified in step 2
                limit=limit,  # Number of final search results to return
                filter=expr,
                output_fields=[
                    "regulation_type",
                    "section_num",
                    "section_name",
                    "part_num",
                    "part_name",
                    "subpart_num",
                    "subpart_name",
                    "text",
                ],
            )
        if expr is None or search_res is None or len(search_res[0]) == 0:

            request_1 = AnnSearchRequest(
                query_sparse_embedding,
                "sparse_vector_embeddings",
                {"metric_type": "IP"},
                limit=limit,
            )
            request_2 = AnnSearchRequest(
                query_dense_embedding,
                "dense_vector_embeddings",
                {"metric_type": "L2"},
                limit=limit,
            )

            # Store these two requests as a list in `reqs`
            reqs = [request_1, request_2]

            rerank = WeightedRanker(0.8, 0.2)

            search_res = col_chunks.hybrid_search(
                reqs,  # List of AnnSearchRequests created in step 1
                rerank=rerank,  # Reranking strategy specified in step 2
                limit=limit,  # Number of final search results to return
                output_fields=[
                    "regulation_type",
                    "section_num",
                    "section_name",
                    "part_num",
                    "part_name",
                    "subpart_num",
                    "subpart_name",
                    "text",
                ],
            )

        # print(search_res)

        context = "".join(
            [
                (
                    (
                        "Regulation: "
                        + res.get("regulation_type")
                        + " : "
                        + res.get("section_num")
                        + "\n"
                    )
                    + (
                        "Section Detail: "
                        + res.get("section_num")
                        + " : "
                        + res.get("section_name")
                        + "\n"
                    )
                    + (
                        "SubPart Detail: "
                        + res.get("subpart_num")
                        + " : "
                        + res.get("subpart_name")
                        + "\n"
                    )
                    + ("Text: " + res.get("text") + "\n")
                    + "\n-----------------------------\n\n"
                )
                for res in search_res[0]
            ]
        )
        sources = get_sources()
        for res in search_res[0]:
            if res.get("regulation_type") == "DFAR":
                name = f"DFARS {res.get('section_num')}"
                part_num = res.get("part_num")
                part_init = part_num.split(" ")[0].lower()
                part_name = res.get("part_name").lower()
                if "appendix a" not in part_init or "appendix a" not in part_name:
                    part_trailing = f"{res.get('part_num').split(' ')[1]}-{res.get('part_name').lower().replace('of ', '').replace('of, ', '').replace(' ', '-').replace(',','')}"
                else:
                    part_trailing = f"{res.get('part_name').lower().replace('of ', '').replace('of, ', '').replace(' ', '-').replace(',','')}"

                url = f"https://www.acquisition.gov/dfars/{part_init}-{part_trailing}"
                # url = f"https://www.acquisition.gov/dfars/part-{int(res.get('part_num').split(' ')[1])}-{res.get('part_name').lower().replace('of ', '').replace('of, ', '').replace(' ', '-').replace(',','')}"
                # sources["far_acquisitions"].append([name, url])
                final_list = [name, url]
                if final_list not in sources["far_acquisitions"]:
                    sources["far_acquisitions"].append(final_list)
            elif res.get("regulation_type") == "FAR":
                name = f"FAR {res.get('section_num')}"
                url = f"https://www.acquisition.gov/far/part-{res.get('part_num').split(' ')[1]}"
                # sources["far_acquisitions"].append([name, url])
                final_list = [name, url]
                if final_list not in sources["far_acquisitions"]:
                    sources["far_acquisitions"].append(final_list)
        # sources["far_acquisitions"].extend([f"https://"] for res in search_res[0])
        set_sources(sources)

        return context

    except MilvusException as e:
        logger.error(
            f"Milvus Error occured while hybrid search: {e}, switching to general search without expr"
        )
        try:
            embed = Embeddings()
            query_dense_embedding, query_sparse_embedding = embed.encode_for_search(
                query
            )
            request_1 = AnnSearchRequest(
                query_sparse_embedding,
                "sparse_vector_embeddings",
                {"metric_type": "IP"},
                limit=limit,
            )
            request_2 = AnnSearchRequest(
                query_dense_embedding,
                "dense_vector_embeddings",
                {"metric_type": "L2"},
                limit=limit,
            )

            # Store these two requests as a list in `reqs`
            reqs = [request_1, request_2]

            rerank = WeightedRanker(0.8, 0.2)

            search_res = col_chunks.hybrid_search(
                reqs,  # List of AnnSearchRequests created in step 1
                rerank=rerank,  # Reranking strategy specified in step 2
                limit=limit,  # Number of final search results to return
                output_fields=[
                    "regulation_type",
                    "section_num",
                    "section_name",
                    "part_num",
                    "part_name",
                    "subpart_num",
                    "subpart_name",
                    "text",
                ],
            )
            context = "".join(
                [
                    (
                        (
                            "Regulation: "
                            + res.get("regulation_type")
                            + " : "
                            + res.get("section_num")
                            + "\n"
                        )
                        + (
                            "Section Detail: "
                            + res.get("section_num")
                            + " : "
                            + res.get("section_name")
                            + "\n"
                        )
                        + (
                            "SubPart Detail: "
                            + res.get("subpart_num")
                            + " : "
                            + res.get("subpart_name")
                            + "\n"
                        )
                        + ("Text: " + res.get("text") + "\n")
                        + "\n-----------------------------\n\n"
                    )
                    for res in search_res[0]
                ]
            )
            sources = get_sources()
            for res in search_res[0]:
                if res.get("regulation_type") == "DFAR":
                    name = f"DFARS {res.get('section_num')}"
                    part_num = res.get("part_num")
                    part_init = part_num.split(" ")[0].lower()
                    part_name = res.get("part_name").lower()
                    if "appendix a" not in part_init or "appendix a" not in part_name:
                        part_trailing = f"{res.get('part_num').split(' ')[1]}-{res.get('part_name').lower().replace('of ', '').replace('of, ', '').replace(' ', '-').replace(',','')}"
                    else:
                        part_trailing = f"{res.get('part_name').lower().replace('of ', '').replace('of, ', '').replace(' ', '-').replace(',','')}"

                    url = (
                        f"https://www.acquisition.gov/dfars/{part_init}-{part_trailing}"
                    )

                    # url = f"https://www.acquisition.gov/dfars/part-{int(res.get('part_num').split(' ')[1])}-{res.get('part_name').lower().replace('of ', '').replace('of, ', '').replace(' ', '-').replace(',','')}"
                    final_list = [name, url]
                    if final_list not in sources["far_acquisitions"]:
                        sources["far_acquisitions"].append(final_list)
                elif res.get("regulation_type") == "FAR":
                    name = f"FAR {res.get('section_num')}"
                    url = f"https://www.acquisition.gov/far/part-{int(res.get('part_num').split(' ')[1])}"
                    # sources["far_acquisitions"].append([name, url])
                    final_list = [name, url]
                    if final_list not in sources["far_acquisitions"]:
                        sources["far_acquisitions"].append(final_list)
            set_sources(sources)

            return context
        except MilvusException as e:
            logger.error(f"Milvus Error occured while general search: {e}")
        except grpc._channel._InactiveRpcError as e:
            logger.error(f"GRPC Error occured while general search: {e}")
        except ValueError as e:
            logger.error("Value Error occured.", exc_info=True)
        except Exception as e:
            logger.error("Exception in tool", exc_info=True)

    except grpc._channel._InactiveRpcError as e:
        # logging the error
        if e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
            logger.error(f"GRPC Resource Exhausted: {e}", exc_info=True)
        else:
            logger.error(f"GRPC Error occured: {e}", exc_info=True)

        raise MilvusDBError()
    except ValueError as e:
        logger.error("Value Error occured.", exc_info=True)

    except Exception as e:
        logger.error("Exception in tool", exc_info=True)
