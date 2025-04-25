from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate
from services.logger import logger

def get_main_prompt():
    prompt = """You are an expert assistant for question-answering tasks. Follow these rules:
    1. Use ONLY the provided context to answer questions
    2. If the context doesn't contain the answer, say "I don't have enough information to answer this question based on the provided context"
    3. Cite specific sources from the context when possible
    4. If the question is unclear, ask for clarification
    5. Provide concise but complete answers
    6. If multiple sources have conflicting information, acknowledge this and explain the differences
    7. Maintain a professional and helpful tone

    Context:
    {context}
    """
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{user_query}")
    ])
    return final_prompt


def get_query_refiner_prompt():
    contextualize_q_system_prompt = ("""
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as it is."
    """)

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("human","{query}"),
        ]
    )
    # print(final_prompt)
    return final_prompt