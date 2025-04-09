import os

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from config import answer_examples

store = {}
load_dotenv()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_llm(model='llama3.2-ko'):
    llm = ChatOllama(
        model=model,
        base_url=os.getenv("LLM_BASE_URL")
    )

    return llm


def get_dictionary_chain():
    dictionary = ["expression representing a person -> 거주자"]
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        f"""
           Please look at the user's question and change the user's question by referring to our dictionary.
           If you decide there is no need to change, you do not need to change your question.
           Dictionary: {dictionary}
           
           Question: {{question}}
        """
    )

    dictionary_chain = prompt | llm | StrOutputParser()

    return dictionary_chain


def get_retriever():
    embedding = OllamaEmbeddings(model="llama3.2-ko", base_url=os.getenv("LLM_BASE_URL"))

    index_name = 'tax-markdown-index'

    database = ElasticsearchStore(
        es_url=os.getenv("ES_URL"),
        index_name=index_name,
        embedding=embedding,
        es_user=os.getenv("ES_USER"),
        es_password=os.getenv("ES_PASSWORD"),
        es_params={
            "verify_certs": False,
            "ssl_show_warn": False
        }
    )

    retriever = database.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}, k=3
    )
    return retriever

def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever

def get_rag_chain():
    llm = get_llm()
    history_aware_retriever = get_history_retriever()
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    system_prompt = (
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요."
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"
        "답변을 제공할 때는 소득세법 (XX)조에 따르면 이라고 시작하면서 답변해주시고"
        "2-3 문장정도의 짧은 내용의 답변을 원합니다."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick("answer")
    return conversational_rag_chain


def get_ai_response(query):
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()

    tax_chain = {"input": dictionary_chain} | rag_chain
    ai_response = tax_chain.stream(
        input={"question": query},
        config={
            "configurable": {"session_id": "abc123"}
        }
    )

    return ai_response
