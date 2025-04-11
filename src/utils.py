import os
import re
import jwt

from together import Together

from src.loggers import logger
from src.infer_config import client as together_client


def query_followup(expanded_query):

    followup_questions_prompt = f"""
        You are an expert specializing in generating insightful follow-up questions. Your task is to analyze an expanded query and generate 3 different follow-up questions that explore various perspectives of the topic. These questions will help retrieve relevant documents from a vector database, overcoming some limitations of distance-based similarity search and adding contextual relevance.

        Here's how to approach this task:
        1. Carefully read and understand the expanded query.
        2. Identify key concepts, themes, or potential areas of interest within the query.
        3. Consider different angles or perspectives that could provide additional context.
        4. Formulate three distinct follow-up questions that explore these different aspects.

        Your follow-up questions should:
        - Be clear and concise
        - Explore different angles or perspectives of the topic
        - Encourage deeper analysis or consideration of the subject
        - Not simply restate information already covered in the expanded query

        Let's go through a few examples to illustrate this process:

        Example 1:
        Expanded query: What are the economic and environmental impacts of renewable energy sources such as solar, wind, and hydroelectric power compared to traditional fossil fuels?

        Thought process: We should consider various aspects of renewable energy, including economic factors, environmental benefits, and potential challenges. It's also important to think about the long-term implications and specific use cases.

        Follow-up questions (in Python list format):
        [
            "How do the installation and maintenance costs of renewable energy infrastructure compare to those of fossil fuel power plants over a 30-year period?",
            "What are the potential negative environmental consequences of large-scale renewable energy projects, such as habitat disruption or raw material extraction?",
            "How can developing countries balance the need for rapid economic growth with the transition to renewable energy sources?"
        ]

        Example 2 (in Python list format):
        Expanded query: What are the most effective strategies for improving urban public transportation systems to reduce traffic congestion, decrease carbon emissions, and enhance overall quality of life for city residents?

        Thought process: We should explore various aspects of urban public transportation, including infrastructure, technology, policy, and user experience. It's also important to consider the challenges and potential resistance to changes in transportation systems.

        Follow-up questions:
        [
            "How can cities integrate smart technologies and data analytics to optimize public transportation routes and schedules in real-time?",
            "What incentives or policies have been most successful in encouraging people to switch from private vehicles to public transportation in major cities worldwide?",
            "How do cultural attitudes and social norms influence the adoption and use of public transportation in different urban environments?"
        ]

        Now, please generate three follow-up questions based on the following expanded query:

        Expanded query: {expanded_query}

        Thought process: [Your thought process here]

        Follow-up questions (in Python list format):
        [
            "[First follow-up question here]",
            "[Second follow-up question here]",
            "[Third follow-up question here]",
        ]
        """

    try:
        response = together_client.chat.completions.create(
            model=os.getenv("MODEL_NAME"),
            messages=[{"role": "user", "content": followup_questions_prompt}],
        )

        followup_questions_response = response.choices[0].message.content.strip()

        inside_brackets = re.search(
            r"\[(.*?)\]", followup_questions_response, re.DOTALL
        ).group(1)

        # Split the content into individual questions
        followup_questions_list = [
            q.strip().strip('"') for q in inside_brackets.split(",\n")
        ]

        if not followup_questions_list:
            raise Exception("")

        return followup_questions_list

    except Exception as e:
        logger.error("Exception in generating follow-up questions", exc_info=True)
        return expanded_query


def chat_name_generator(user_query, expanded_query):

    client = Together(
        api_key=os.getenv("TOGETHER_API_KEY"),
        base_url=os.getenv("INFERENCE_LLM_BASE_URL"),
    )

    chat_name_gen_prompt = f"""
        You are an AI assistant specialized in creating concise and relevant chat names. Your task is to generate a chat name of no more than 5 words based on a user's original query and its expanded version. This chat name should capture the essence of the conversation and be easily recognizable for future reference.

        Follow these steps:
        1. Analyze both the original query and its expansion.
        2. Identify the core topic or main focus of the conversation.
        3. Think of a concise yet descriptive way to encapsulate this topic.
        4. Create a chat name using no more than 5 words.

        Here are some examples to guide you:

        Example 1:
        Original query: Effects of social media
        Expanded query: What are the positive and negative effects of popular social media platforms like Facebook, Twitter, and Instagram on mental health, social relationships, productivity, and information dissemination in both personal and professional contexts?
        Thought process: The query focuses on the impacts of social media across various aspects of life. A concise name should capture this broad impact.
        Chat name (in python list format): 
        [
            "Social Media's Multifaceted Impact"
        ]

        Example 2:
        Original query: Best diet for weight loss
        Expanded query: What are the most effective and sustainable dietary approaches for healthy weight loss, considering factors such as nutritional balance, individual metabolism, exercise integration, long-term adherence, and potential health impacts for different age groups and body types?
        Thought process: The query is about effective and sustainable weight loss diets. The chat name should reflect the focus on sustainability and effectiveness.
        Chat name (in python list format): 
        [
            "Sustainable Weight Loss Strategies"
        ]

        Example 3:
        Original query: Future of autonomous vehicles
        Expanded query: What are the technological advancements, regulatory challenges, ethical considerations, and potential societal impacts of widespread adoption of autonomous vehicles in urban and rural environments over the next decade?
        Thought process: The query covers various aspects of autonomous vehicles' future. The chat name should encompass this forward-looking, multi-faceted exploration.
        Chat name (in python list format): 
        [
            "Autonomous Vehicles: Future Landscape"
        ]

        Now, please generate a chat name for the following query:

        Original query: {user_query}
        Expanded query: {expanded_query}

        Thought process: [Your thought process here]
        Chat name (in python list format): 
        [
            "[Your chat name here]"
        ]
        """
    try:
        response = together_client.chat.completions.create(
            model=os.getenv("MODEL_NAME"),
            messages=[{"role": "user", "content": chat_name_gen_prompt}],
        )

        chat_name_gen_response = response.choices[0].message.content.strip()

        inside_brackets = re.search(
            r"\[(.*?)\]", chat_name_gen_response, re.DOTALL
        ).group(1)

        # Split the content into individual questions
        chat_name_gen_list = [
            q.strip().strip('"') for q in inside_brackets.split(",\n")
        ]

        if not chat_name_gen_list:
            raise Exception("")

        return chat_name_gen_list

    except Exception as e:
        logger.error("Exception in generating follow-up questions", exc_info=True)
        return ["New Chat"]


def answer_regeneration(original_question, original_answer):

    client = Together(
        api_key=os.getenv("TOGETHER_API_KEY"),
        base_url=os.getenv("INFERENCE_LLM_BASE_URL"),
    )

    ans_regeneration_prompt = f"""
        You are an AI assistant specialized in improving and refocusing answers to questions. Your task is to take an original question and its corresponding answer, then regenerate the answer in a more focused, well-structured, and error-free manner. The new answer should directly address the question asked, eliminate irrelevant information, and present the content in the most appropriate and clear structure.

        Follow these steps:
        1. Carefully read and understand the original question.
        2. Analyze the provided answer, identifying its main points and any errors or irrelevant information.
        3. Determine the most effective structure for the regenerated answer (e.g., paragraphs, occasional bullet points, or a mix).
        4. Restructure the content to directly address the question, focusing on the most relevant information.
        5. Improve clarity, coherence, and logical flow of the answer.
        6. Correct any errors or inaccuracies found in the original answer.
        7. Ensure the regenerated answer is comprehensive yet concise.
        8. Enclose the final regenerated answer in a Python list.

        Here are some examples to guide you:

        Example 1:
        Original question: What are the main causes of climate change?
        Original answer: Climate change is a complex issue that affects the entire planet. It's caused by a variety of factors, including deforestation, industrial processes, and the burning of fossil fuels. These activities release greenhouse gases into the atmosphere, primarily carbon dioxide. The greenhouse effect then traps heat in the Earth's atmosphere, leading to global warming. Some people argue that climate change is a natural process, but the vast majority of scientists agree that human activities are the primary driver of the current rapid warming trend. Climate change can lead to rising sea levels, more frequent extreme weather events, and changes in ecosystems. Polar bears are particularly affected by the melting of sea ice.

        Thought process: The original answer contains relevant information but includes tangential points and lacks a clear structure. It needs to be reorganized to focus directly on the causes of climate change, as asked in the question. A paragraph structure with a clear introduction and conclusion would be most appropriate here.

        Regenerated answer (in python list format): 
        [
            "Climate change is primarily driven by human activities that increase the concentration of greenhouse gases in the Earth's atmosphere. The main cause is the burning of fossil fuels such as coal, oil, and natural gas for energy production and transportation. This process releases large amounts of carbon dioxide, the most significant greenhouse gas contributing to global warming. Deforestation is another major factor, as it reduces the Earth's capacity to absorb carbon dioxide and often involves burning, which releases additional greenhouse gases. Industrial processes, including cement production and chemical manufacturing, also contribute significantly to emissions. Agricultural practices, particularly livestock farming and rice cultivation, produce methane, another potent greenhouse gas. Lastly, improper waste management, such as in landfills and wastewater treatment, releases methane and other greenhouse gases. While natural factors can influence climate, the scientific consensus is that these human activities are the dominant cause of the rapid warming observed in recent decades."
        ]

        Example 2:
        Original question: How does exercise affect mental health?
        Original answer: Exercise is really good for you. It makes your body stronger and helps you lose weight. When you exercise, your body releases endorphins, which are chemicals that make you feel good. This can help with depression and anxiety. Exercise also helps you sleep better at night. Some people like to run, while others prefer swimming or cycling. It's important to find an exercise you enjoy. Eating a healthy diet is also important for mental health. Exercise can improve cognitive function and reduce the risk of dementia in older adults.

        Thought process: The original answer contains some relevant information but is disorganized and includes off-topic points. The regenerated answer should focus specifically on the effects of exercise on mental health, organizing the information more logically. A mix of paragraphs and occasional bullet points might work well to present this information clearly.

        Regenerated answer (in python list format): 
        [
        "Exercise has a profound and multifaceted impact on mental health, offering benefits that extend far beyond physical fitness. Regular physical activity stimulates the release of endorphins, serotonin, and dopamine – neurotransmitters that enhance mood and promote feelings of happiness and well-being. This biochemical boost can significantly alleviate symptoms of depression and anxiety, making exercise an effective complementary treatment for these conditions.

        The mental health benefits of exercise include:
        - Stress reduction: Physical activity lowers stress hormones like cortisol and adrenaline, helping to reduce anxiety and tension.
        - Improved sleep: Regular exercise can regulate sleep patterns, leading to better quality rest, which is crucial for mental well-being.
        - Enhanced cognitive function: Increased blood flow to the brain during exercise can boost memory, concentration, and overall cognitive performance.
        - Increased self-esteem: Achieving fitness goals and improving physical health often leads to improved body image and self-confidence.

        Moreover, exercise provides opportunities for social interaction through group activities or sports, which can combat feelings of isolation and loneliness. It also builds resilience, increasing one's ability to cope with stress and adversity. For older adults, consistent physical activity has been shown to reduce the risk of cognitive decline and dementia.

        The type and intensity of exercise can vary based on individual preferences and capabilities, but consistency is key to reaping these mental health benefits. Whether it's a brisk walk, a yoga session, or intense cardio, finding enjoyable forms of exercise can lead to long-term improvements in mental well-being."
        ]

        Now, please regenerate the answer for the following question-answer pair:

        Original question: {original_question}
        Original answer: {original_answer}

        Thought process: [Your thought process here]
        Regenerated answer (in python list format): 
        [
            "[Your regenerated answer will be placed here, inside the Python list]"
        ]
"""
    try:
        response = together_client.chat.completions.create(
            model=os.getenv("MODEL_NAME"),
            messages=[{"role": "user", "content": ans_regeneration_prompt}],
        )

        ans_regeneration_response = response.choices[0].message.content.strip()
        if ans_regeneration_response is None:
            logger.error(f"Nonetype response received from the LLM: {response}")
            return original_answer
        inside_brackets = re.search(r"\[(.*?)\]", ans_regeneration_response, re.DOTALL)
        if inside_brackets is None or not inside_brackets:
            logger.error("The query is not extracted in the correct format")
            return original_answer
        else:
            inside_brackets = inside_brackets.group(1)
        # Split the content into individual questions
        ans_regeneration_list = [
            q.strip().strip('"') for q in inside_brackets.split(",\n")
        ]

        if not ans_regeneration_list:
            raise Exception("The query is not extracted in the correct format")

        return ans_regeneration_list[0]

    except Exception as e:
        logger.error(f"Exception in re-generating answer {e}", exc_info=True)
        return original_answer


def verify_user_token(token, user_id):
    # Decode the token
    try:
        decoded_token = jwt.decode(
            token, os.getenv("ATLAS_JWT_SECRET_KEY"), algorithms=["HS256"]
        )

        if decoded_token["userId"] == user_id:
            return True
        else:
            return False
    except jwt.ExpiredSignatureError:
        return False
    except jwt.InvalidTokenError:
        return False
    except Exception as e:
        logger.error("Exception in verifying user token", exc_info=True)
        return False
