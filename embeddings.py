from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, OPENAI_API_KEY, LLM_UTILITY_MODEL

def get_embedding_model():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
        openai_api_key=OPENAI_API_KEY
    )

def get_utility_llm():
    return ChatOpenAI(
        model_name=LLM_UTILITY_MODEL,
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    ) 