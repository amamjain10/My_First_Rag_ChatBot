import pdfplumber
import streamlit as st
from click import prompt
from langchain_classic.chains import llm
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.header("My First Rag Chatbot ")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF File and Start asking Questions", type="pdf")

#extract content from the pdf and chunk it
if file is not None:
    #extract text
    with pdfplumber.open(file) as pdf:
        text= ""
        for page in pdf.pages:
            text += page.extract_text() +"\n"
    st.write(text)

    text_splitter= RecursiveCharacterTextSplitter(
        separators=["\n\n","\n","."," ",""],
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    st.write(chunks)

    #generating embeddings
    embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-small"
        openai_api_key = OPENAI_API_KEY
    )

    # store embeddings in vector db
    vector_store= FAISS.from_texts(chunks,embeddings)

    #get_user_question
    user_question = st.text_input("Enter your question here")

    #generate anwser
    #question -> embeddings -> similiarity search -> results to LLM -> response (CHAIN)
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    retriever = vectorstore.asretriever(
        search_type= "mmr",
        search_kwargs={"k":4}
    )

    llm= ChatopenAI(
        model="gpt-40-mini",
        temperature = 0.3,
        max_tokens = 1000,
        openai_api_key = OPENAI_API_KEY
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant answering questions about a PDF document.\n\n"
         "Guidelines:\n"
         "1. Provide complete, well-explained answers using the context below.\n"
         "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
         "3. If the context mentions related information, include it to give fuller picture.\n"
         "4. Only use information from the provided context - do not use outside knowledge.\n"
         "5. Summarize long information, ideally in bullets where needed\n"
         "6. If the information is not in the context, say so politely.\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    chain = (
        {"context" : retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if user_question:
        response = chain.invoke(user_question)
        st.write(response)

