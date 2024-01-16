from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from rag_chroma.prompt_templates import RAG_PROMPT_TEMPLATE

EMBEDDING_FUNCTION = OpenAIEmbeddings()
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "rag-chroma"

# Embed a single document as a test
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, collection_name=CHROMA_COLLECTION_NAME,
                     embedding_function=EMBEDDING_FUNCTION)
retriever = vectorstore.as_retriever()

# RAG prompt
template = RAG_PROMPT_TEMPLATE
prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI()

# RAG chain
chain = (
        RunnableParallel({"context": retriever, "user_input": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
