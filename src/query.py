import os
import re

from dotenv import load_dotenv

from src.loggers import logger
from src.infer_config import client as together_client

load_dotenv()


def query_expansion(user_query):

    # Step 1: Query Expansion
    expansion_prompt = f"""
        You are an expert in expanding user queries related to searching government tenders and acquisitions regulations like FAR or DFARS. Your primary task is to identify whether the user's query is specifically related to government tenders or acquisitions. Only if the query is relevant to these areas, expand it into a more detailed and comprehensive query. If the query is general or unrelated, return the original query unchanged.
        You must also send the query unchanged if it is related to the company Binoloop or the product Atlas.

        Follow these steps:
        1. Carefully analyze the user query and determine if it's directly related to government tenders or acquisitions (e.g., government contracts, federal procurement, public projects, acquisition regulations).
        2. If it is related to government tenders or acquisitions:
        a. Expand the query to include more specific details, relevant context, and related aspects (e.g., contract details, agencies involved, project types, etc.).
        b. Ensure the expanded query is comprehensive and detailed, enhancing the search in a tenders or acquisitions database.
        3. If the query is not related to government tenders or acquisitions, return it unchanged without adding any new information.
        4. The final output should be a Python list containing either the expanded or original query in plain text, with no extra code formatting.

        Here are some examples to guide you:

        Example 1:
        User query: "Hello, how are you?"
        Output: 
        [
            "Hello, how are you?"
        ]

        Example 2:
        User query: "Recent government contracts for cybersecurity"
        Output: 
        [
            "What are the most recent government contracts awarded for cybersecurity services or products, including details on the awarding agency, contract value, duration, specific cybersecurity focus areas (e.g., network security, data protection, threat intelligence), and any notable requirements or innovations mentioned in the tender documents?"
        ]

        Example 3:
        User query: "Federal construction projects in California"
        Output: 
        [
            "Provide information on ongoing and upcoming federal construction projects in California, including project types (e.g., infrastructure, buildings, renovations), estimated budgets, timelines, specific locations within California, awarding agencies, and any sustainability or local economic impact requirements mentioned in the tender documents"
        ]

        Example 4:
        User query: "What's the weather like today?"
        Output:
        [
            "What's the weather like today?"
        ]

        Now, please analyze and expand (if appropriate) the following user query using this approach:
        User query: {user_query}
        Output: 
        [
            "[Your output here]"
        ]
    """
    try:
        response = together_client.chat.completions.create(
            model=os.getenv("MODEL_NAME"),
            messages=[{"role": "user", "content": expansion_prompt}],
        )

        expansion_response = response.choices[0].message.content.strip()

        inside_brackets = re.search(r"\[(.*?)\]", expansion_response, re.DOTALL).group(
            1
        )

        # Split the content into individual questions
        expansion_query_list = [
            q.strip().strip('"') for q in inside_brackets.split(",\n")
        ]

        if not expansion_query_list:
            raise Exception("")

        return expansion_query_list[0]

    except Exception as e:
        logger.error(
            "Could not generate expanded query. Returning original query.",
            exc_info=True,
        )
        return user_query


def query_decomposition(expanded_query):

    decomposition_prompt = f"""
        You are an AI assistant specialized in query decomposition. Your task is to take an expanded query and break it down into 4 to 5 smaller, more specific sub-questions. These sub-questions should cover different aspects of the main query and, when answered together, should provide a comprehensive response to the original query.

        Here's how you should approach this task:

        1. Analyze the expanded query to identify its main components and themes.
        2. Break down these components into distinct, focused sub-questions.
        3. Ensure that the sub-questions are clear, specific, and answerable.
        4. Make sure that the set of sub-questions covers all important aspects of the original query.

        Let's go through a few examples to illustrate this process:

        Example 1:
        Expanded query: What are the positive and negative effects of popular social media platforms like Facebook, Twitter, and Instagram on mental health, social relationships, productivity, and information dissemination in both personal and professional contexts?

        Thought process: This query covers multiple aspects of social media impact. We should break it down by effects (positive/negative), platforms, and areas of impact.

        Decomposed sub-questions (in Python list format):
        [
            "How do social media platforms like Facebook, Twitter, and Instagram impact users' mental health, both positively and negatively?",
            "In what ways do popular social media platforms affect social relationships in personal and professional settings?",
            "How does social media usage influence individual and workplace productivity?",
            "What role do major social media platforms play in information dissemination, and what are the implications for personal and professional contexts?",
            "How do the effects of social media differ across various platforms like Facebook, Twitter, and Instagram?"
        ]

        Example 2:
        Expanded query: What is the comprehensive environmental impact of electric cars compared to traditional gasoline vehicles, considering factors such as manufacturing processes, battery production and disposal, electricity sources for charging, and overall carbon footprint throughout the vehicle's lifecycle?

        Thought process: This query compares electric and gasoline vehicles across various environmental factors. We should break it down by stages of the vehicle lifecycle and specific environmental concerns.

        Decomposed sub-questions (in Python list format):
        [
            "How do the manufacturing processes of electric cars and traditional gasoline vehicles compare in terms of environmental impact?",
            "What are the environmental implications of battery production and disposal for electric vehicles?",
            "How does the source of electricity for charging electric cars affect their overall environmental impact?",
            "What is the comparative carbon footprint of electric and gasoline vehicles throughout their entire lifecycle?",
            "In what ways do electric cars offer environmental advantages over traditional gasoline vehicles, and what are their potential drawbacks?"
        ]

        Now, please decompose the following expanded query into sub-questions using this approach:

        Expanded query: {expanded_query}

        Thought process: [Your thought process here]

        Decomposed sub-questions (in Python list format):
        [
            "[First sub-question here]",
            "[Second sub-question here]",
            "[Third sub-question here]",
            "[Fourth sub-question here]",
            "[Fifth sub-question here]"
        ]
        """

    try:
        response = together_client.chat.completions.create(
            model=os.getenv("MODEL_NAME"),
            messages=[{"role": "user", "content": decomposition_prompt}],
        )

        decomposition_response = response.choices[0].message.content.strip()

        inside_brackets = re.search(
            r"\[(.*?)\]", decomposition_response, re.DOTALL
        ).group(1)

        # Split the content into individual questions
        decomposed_questions_list = [
            q.strip().strip('"') for q in inside_brackets.split(",\n")
        ]

        if not decomposed_questions_list:
            raise Exception("")

        return decomposed_questions_list

    except Exception as e:
        logger.error(
            "Could not generate decomposed query. Returning expanded query.",
            exc_info=True,
        )
        return expanded_query
