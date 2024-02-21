import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, AzureChatOpenAI,AzureOpenAIEmbeddings
from lib.gpt import retriever,chat_chain,chat_pipe_chain ,multiply ,convert_to_openai_tool
import json
from dotenv import load_dotenv
import re
import base64
import requests
from lib.index import AzureSearchIndex
from lib.Vectorization import Vision_Florence
from  io import BytesIO

load_dotenv()

def Image_search(description: str) :
    """Search for an image from a description
    Args:
        description: decsrizion of image
    """
    svi_f = Vision_Florence()
    azure = AzureSearchIndex('pvectortest')
    emb_text=svi_f.predict(description,application='text')
    result=azure.vector_hybrid_search('',emb_text)
    links='\n '.join([x['URL'] for x in result])
    ris=st.session_state['llm2'].invoke(f"fai questi link imagine in foramto markdown html ![Image](url) non inserire ``` e senza a capo : {links}")
    return ris.content

def checkurl(testo):
    pattern_url = re.compile(r'https?://\S+')
    urls_trovate = pattern_url.findall(testo)
    return urls_trovate

def bot(list_url):
    rtv=retriever(st.session_state['embeddings'])
    rtv.add_document(list_url)
    rtv_faiss=rtv.set_FAISS()
    return chat_pipe_chain(st.session_state['llm'],rtv_faiss)

def function_calling(answer):
     st.write(answer)
     name_fun=answer.additional_kwargs['tool_calls'][0]['function']['name']
     arguments=json.loads(answer.additional_kwargs['tool_calls'][0]['function']['arguments'])
     answer=globals()[name_fun](description=arguments['description'])
     return answer

if 'llm' not in st.session_state:
    #os.getenv('deployment_name')
    st.session_state['llm2'] = AzureChatOpenAI(model='gpt4')
    st.session_state['llm'] = st.session_state['llm2'].bind(tools=[convert_to_openai_tool(Image_search)])
    st.session_state['embeddings'] = AzureOpenAIEmbeddings(model='ada')

if 'list_url' not in st.session_state:
    st.session_state['list_url']=['https://www.50sfumaturedioutfit.com/come-vestirsi-bene/','https://www.elle.com/it/moda/tendenze/news/a1355277/come-vestirsi-bene-spendendo-poco/']

if 'bot' not in st.session_state:
    st.session_state['bot']=bot(st.session_state['list_url'])
# ------ load env variable ------# 

st.title("Echo Bot")

with st.sidebar:
    #------------------------Titole
    st.title("Image Retrieval")
    #------------------------Titole

    col =st.columns(2)
    with col[0]:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    # To read file as bytes:
    if uploaded_file is not None:
        
        with col[1]:  
          st.image(uploaded_file, caption=" ", use_column_width=True)
        imge_bytes = uploaded_file.getvalue()
        imge_bytes= BytesIO(imge_bytes).read()
    else:
        imge_bytes=0
        with col[1]:
           st.image('https://static.vecteezy.com/system/resources/previews/023/506/653/original/photo-upload-icon-editable-vector.jpg', caption=" ", use_column_width=True)
        #st.empty()
        st.info('You can enhance your search by uploading an image.')

    #--------------------------Send Button 
    button_check = st.button("Send")
# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hi I am your personal fashion assistant, tell me what you need and I will do my best to find the best for you. good! tell me what are you looking for? ")]

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    role='user' if message.dict()['type']=='human' else 'assistant'
    with st.chat_message(role):
         st.markdown(message.dict()["content"])

# React to user input

if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    answer =st.session_state['bot'].invoke({
            "chat_history": st.session_state.chat_history,
            "input": prompt
            })
    if answer.content=='' and len(answer.additional_kwargs)>0:
        answer=function_calling(answer)
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        pass
    else:
        answer=answer.content
        st.session_state.chat_history.append(HumanMessage(content=prompt))
    
    if isinstance(answer,list):
       response = '\n'.join(answer)
    else:
       response = str(answer)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.chat_history.append(AIMessage(content=response))
