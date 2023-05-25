#########################  Start Imports  ####################################

import streamlit as st
from streamlit_chat import message
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ChatMessageHistory, ConversationSummaryMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationChain
import pinecone
import openai 
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import re

######################### Secrets ###################################

OPEN_AI_KEY = st.secrets["OPEN_AI_KEY"]["OPEN_AI_KEY"]
PINECONE_KEY = st.secrets["PINECONE_KEY"]["PINECONE_KEY"]
PINECONE_API_ENV = st.secrets["PINECONE_KEY"]["PINECONE_API_ENV"]

openai.api_key = OPEN_AI_KEY


#########################  Start Functions  ####################################

def retrieve(query):
    query_embedding_request = openai.Embedding.create(
                          input=[query],
                          engine="text-embedding-ada-002"
                        )
    vector_embedding = query_embedding_request['data'][0]['embedding']
    similarity_result = index.query(vector_embedding, top_k=2, include_metadata=True, namespace=pdf_name)

    #Create prompt start and end
    prompt_start = (
            "Answer the question based on the context below.\n\n"+
            "Context:\n"
        )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:")

    #Context
    contexts = [
            x['metadata']['text'] for x in similarity_result['matches']
        ]

    #Full prompt
    limit = 3750
    for i in range(1, len(contexts)):
            if len("\n\n---\n\n".join(contexts[:i])) >= limit:
                prompt = (
                    prompt_start + "!!!" + f"\n\nit: {similarity_result.namespace}\n" +
                    "\n\n---\n\n".join(contexts[:i-1]) + "!!!" +
                    prompt_end
                )
                break
            elif i == len(contexts)-1:
                prompt = (
                    prompt_start + "!!!" + f"\n\nit: {similarity_result.namespace}\n" + 
                    "\n\n---\n\n".join(contexts) + "!!!" +
                    prompt_end
                )
    return prompt
def get_conversation():
    llm = ChatOpenAI(temperature=0, openai_api_key=OPEN_AI_KEY,model_name="gpt-3.5-turbo")
    template =""""
                System Prompt: You are a chatbot that will answer the user's questions. The user's question will be delimited by ***.
                The user will instruct you to answer a question based on some context they will provide you. The context will be delimeted by !!!.
                If the user's question does not provide enough
                context ask them to clarify their question. 
                Don't mention context.
                You may answer questions with other information outside the context, so long as you connect it back to the context.
                Always respond with the page number and paragraph of where you retrieved the answer to the user's questions. 
                
                Current conversation:
                {history}
                User: {input}
                Chatbot:
            """

    prompt = PromptTemplate(
        input_variables =["history", "input"], template=template
    )

    conversation = ConversationChain(llm=llm, verbose=True, memory=ConversationBufferWindowMemory(k=5), prompt=prompt)

    return conversation
def submit():
    user_input = st.session_state.user_input
    query = user_input
    st.session_state.user_input = ''
    if (len(user_input) > 1):
        full_prompt = retrieve("***"+user_input+"***")
        conversation = st.session_state[conversation_key]
        conversation.predict(input=full_prompt)   
    return query

##########################  Start Back End  ####################################


#Instantiate Pinecone
pinecone.init(
    api_key=PINECONE_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)

#Create PDF embeddings and upload to Pinecone
index_name = "pdfchat"
index = pinecone.Index(index_name)

##########################Start Chat Bot####################################
conversation_key = "conversation"
human_message_key = "human"

#Session State
if conversation_key not in st.session_state:
    st.session_state[conversation_key] = get_conversation()

conversation = st.session_state[conversation_key]

#Drop Down Button
with st.sidebar:
  options = [i for i in index.describe_index_stats()["namespaces"].keys()]
  pdf_name = st.selectbox('Which document would you like to chat with? ',options)

#User input field
response_container = st.container()
input_container = st.container()



           

st.title("PDF Chat")
st.markdown("****Select which document you would like to chat with in the side bar. After you have made your selection, type in your question and press enter. Responses may take 5-10 seconds to load.****")
user_input = st.text_input("What would you like to know?", key='user_input',on_change=submit)


for i, msg in enumerate(conversation.memory.chat_memory.messages):
            if msg.type == human_message_key:
                matches = re.findall(r'\*\*\*(.*?)\*\*\*', msg.content)
                message(matches[0], is_user=True, key=f"msg{i}", avatar_style="initials", seed="Q")
            else:
                message(msg.content, key=f"msg{i}", avatar_style="initials", seed="A")




