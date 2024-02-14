import streamlit as st
from io import BytesIO
from Utility import  florence_single_embedding_bytes ,mobileNetV2_single_feature
from lib.index import LocalAnnoyIndex
import base64
from streamlit_selectable_image_gallery import image_gallery

# ------ load env variable ------# 
from dotenv import load_dotenv
load_dotenv()
# ------ load env variable ------# 

if 'index' not in st.session_state:
    st.session_state['Index'] = LocalAnnoyIndex(function_embedding=florence_single_embedding_bytes,vector_size=1024)
    st.session_state['Index'].load_index()#(path_index='output\index',json_file='output\index_mobileNetV2.json')

features=[]
similar=False
with st.sidebar:
    st.header('Image to Image')
    uploaded_file = st.file_uploader("Carica un'immagine", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Leggi i bytes del file caricato
        file_content = uploaded_file.getvalue()
        st.image(uploaded_file, caption="Immagine caricata", use_column_width=True)
        features = st.session_state['Index'].Search_AnnoyIndex(BytesIO(file_content),n=10)
        similar=True
        st.write(features)
 
if similar:
  st.header('Immagini simili:')
else:
  st.header('Archivio immagini:')

if len(features)>0:
    lsit_img=features
    if len(lsit_img)<1:
        height=250
    else:
        height=150

if features != []:
    images = []
    for file in  lsit_img :
        with open(file ,'rb') as f:
            img=f.read()
        encoded = base64.b64encode(img).decode()
        images.append(f"data:image/jpeg;base64,{encoded}")
    selected_index = image_gallery(images,height)



