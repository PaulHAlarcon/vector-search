import streamlit as st
from st_multimodal_chatinput import multimodal_chatinput

if 'user_inp' not in st.session_state:
    st.session_state['user_inp']=False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


 ##hack to make sure that chatinput is always at the bottom of the page
 ##will only work if multimodal_chatinput is called inside the first st.container of the page

 ##############################################################################
def reconfig_chatinput():
    st.markdown(
        """
    <style>
        div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"]:first-of-type {
            position: fixed;
            bottom: 0;
            width: 100%; /* Span the full width of the viewport */;
            background-color: #0E117;
            z-index: 1000;
            /* Other styles as needed */    
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    return

#reconfig_chatinput()       
##############################################################################

with st.container():
    st.session_state['user_inp'] = multimodal_chatinput()

if st.session_state['user_inp']:
    if st.session_state['user_inp']['text']!= '':
    #with st.chat_message("user"):
        #st.markdown(f'<img src={user_inp["images"][0]} alt="Girl in a jacket" width="100" height="100">',unsafe_allow_html=True)
    #st.markdown(f'<img src={user_inp["images"][0]} alt="Girl in a jacket" width="100" height="100">',unsafe_allow_html=True)
       st.session_state.chat_history.append({'type':'user','data':st.session_state['user_inp']})
       st.session_state.chat_history.append({'type':'bot' ,'data':{'text':'bene grazie'}})
for x in st.session_state.chat_history:
   with st.container(border=True):
        col=st.columns([0.5,6])
        with col[0]:
            if x['type']=='user':
               st.image('Data\shoes.jpg')
            else:
               st.image('Data\log_img.jpg')

        with col[1]:
            try:
              st.markdown(f'<img src={x["data"]["images"][0]} alt="Girl in a jacket" width="100" height="100">',unsafe_allow_html=True)
            except:
                pass
            st.write(x['data']['text'])
            