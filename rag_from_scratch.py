import os

os.environ["USER_AGENT"] = "rag-from-scratch/1.0"

import time
import warnings

from bs4.filter import SoupStrainer

from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langsmith.utils import LangSmithMissingAPIKeyWarning

def main():
    # Ignore LangSmith Warning
    warnings.filterwarnings(
        "ignore",
        category=LangSmithMissingAPIKeyWarning
    )

    #### INDEXING ####

    # Load Documents
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        )
    )
    docs = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Embed
    vectorstore = FAISS.from_documents(documents=splits, embedding=OllamaEmbeddings(model="nomic-embed-text"))

    #### RETRIEVAL and GENERATION ####

    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOllama(model="llama3.2:1b", temperature=0)

    # Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Question
    question = "What is Task Decomposition?"
    start_time = time.perf_counter()
    answer = rag_chain.invoke(question)
    end_time = time.perf_counter()
    generation_time = end_time - start_time

    print("Answer:", answer)
    print("Generation Time:", generation_time)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    main()
