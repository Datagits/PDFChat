#########################  Start Imports  ####################################

import streamlit as st
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
#from langchain.chains.question_answering import load_qa_chain
import pinecone
import openai 

######################### Secrets ###################################

OPEN_AI_KEY = st.secrets["OPEN_AI_KEY"]["OPEN_AI_KEY"]
PINECONE_KEY = st.secrets["PINECONE_KEY"]["PINECONE_KEY"]
PINECONE_API_ENV = st.secrets["PINECONE_KEY"]["PINECONE_API_ENV"]

openai.api_key = OPEN_AI_KEY
#########################  Start Functions  ####################################

def retrieve(query):
    query_embedding_request = openai.Embedding.create(
                          input=[query],
                          engine=embed_model
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
                    prompt_start +
                    "\n\n---\n\n".join(contexts[:i-1]) +
                    prompt_end
                )
                break
            elif i == len(contexts)-1:
                prompt = (
                    prompt_start + f"\n\nit: {similarity_result.namespace}\n" +
                    "\n\n---\n\n".join(contexts) +
                    prompt_end
                )
    return prompt


def initiate_conversation(full_prompt):
    LLM_Response = openai.ChatCompletion.create(
                                            model="gpt-3.5-turbo",
                                            messages=[
                                                  {"role": "system", "content": """You are a chatbot that will answer the user's questions. 
                                                                                  The user will instruct you to answer a question based on 
                                                                                  some context they will provide you. They will then provide 
                                                                                  their question. You will complete the answer to their question 
                                                                                  based on the context. If the user's question does not provide enough
                                                                                  context ask them to clarify their question. Help them clarify their 
                                                                                  question by asking them for the specific information they need to answer
                                                                                  their question. Finally, if the user refers to 'it' they are likely referring
                                                                                  to the namespace provided in the context. Use the namespace to determine what 
                                                                                  the means by 'it' if they are not immediatly clear about what 'it' is. Go into detail 
                                                                                  about what 'it' is if they ask. Actually describe what the document is, don't just say it is a document. Don't mention context.
                                                                                  You may answer questions with other information outside the context, so long as you connect it back to the context"""},
                                                  {"role": "user", "content": full_prompt}
                                              ]
                                            )

    return LLM_Response['choices'][0]['message']['content']


#Clears the chat input field after each chat submission. 
def clear_text():
    st.session_state.init_input = st.session_state.widget
    st.session_state.widget = ''

 

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

#Session State
if "init_input" not in st.session_state:
    st.session_state.init_input = ""


#Drop Down Button
with st.sidebar:
  options = [i for i in index.describe_index_stats()["namespaces"].keys()]
  pdf_name = st.selectbox('Which document would you like to chat with? ',options)

#User input field
user_input_container = st.container()
user_input_container.text_input('What would you like to know?', key="widget", on_change=clear_text)

#Chat history
#chat_history_container = st.container()
#chat_history = chat_history_container.empty()

if st.session_state.init_input:
    # Update the chat history with the new user input and bot response
    #chat_history.write(f'User: {st.session_state.init_input}')
    
    #Create Query Embdeddings
    embed_model = "text-embedding-ada-002"
    query=st.session_state.init_input
    full_prompt = retrieve(query)

    #Initiate conversation with bot
    st.write(f'User: {query}\n', '\nPDFChat: ',initiate_conversation(full_prompt))
    #st.write(full_prompt)
    #chat_history.write(initiate_conversation(full_prompt))
    
