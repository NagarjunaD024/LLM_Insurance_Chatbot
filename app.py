import streamlit as st
import os
import time
from pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.llm import LLMChain
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore


# Give your Pinecone Standard Account key
PINECONE_API_KEY = ""
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Give your  Open AI PI Account key
OPENAI_API_KEY = ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY



st.set_page_config(
    page_title="Insurance Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.header("Insurance Chatbot")
st.markdown("Ask your Insurance-related questions here.")



embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
model_name = "gpt-4o"
text_field = "text"

@st.cache_resource
def ret():
    # Initialize a Pinecone client with your API key
    pc = Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1")
    # load a Pinecone index
    index = pc.Index("insurance")
    time.sleep(5)
    db = PineconeVectorStore(index=index, embedding=embeddings, text_key=text_field)
    return db

@st.cache_resource
def init_memory():
    return ConversationBufferWindowMemory(
                                        k=2, 
                                        memory_key="chat_history", 
                                        return_messages=True,
                                        verbose=True)

memory = init_memory()

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a 
standalone question without changing the content in given question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
condense_question_prompt_template = PromptTemplate.from_template(_template)

prompt_template = """You are helpful information giving QA System and make sure you don't answer anything not related to following context except if it is mathematical calculations, geographical query, information about universities in Germany, and real-time weather updates for Germany.
You are always provide useful information & details available in the given context and provided exceptional cases. Use the following pieces of context to answer the question at the end and if question is from given exceptional situation then you can answer using your own data. 
Also check chat history if question can be answered from it or question asked about previous history. If you don't know the answer, just say that you don't know, don't try to make up an answer. Except if the question is asked from given exceptional topics that you can answer on your own.

{context}
Chat History: {chat_history}
Question: {question}
Answer:"""

qa_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "chat_history","question"]
)

db = ret()

#@st.cache_resource
def conversational_chat(query):
    llm = ChatOpenAI(model_name = model_name)
    question_generator = LLMChain(llm=llm, prompt=condense_question_prompt_template, memory=memory, verbose=True)
    doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt, verbose=True)
    agent = ConversationalRetrievalChain(
        retriever=db.as_retriever(search_kwargs={'k': 6}),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        memory=memory,
        verbose=True,
    )

    return agent

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input():
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content":prompt})
    # st.chat_message("user").write(prompt)
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        agent = conversational_chat(prompt)
        with st.spinner("Thinking..."):
            with get_openai_callback() as cb:
                response = agent({'question': prompt, 'chat_history': st.session_state.chat_history})#agent({"query": prompt})#conversational_chat(prompt)#
                st.session_state.chat_history.append((prompt, response["answer"]))
                message_placeholder.markdown(response["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            