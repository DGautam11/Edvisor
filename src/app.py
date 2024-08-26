from utils import Utils
import streamlit as st
from engine import Engine
from dateutil import parser

# Set the page configuration
st.set_page_config(page_title='Edvisor', page_icon='🎓')
st.title('Edvisor 🤖')
st.write('Chatbot for Finland Study and Visa Services')

# Initialize the chatbot engine
@st.cache_resource
def initialize_engine():
    return Engine()

chatbot = initialize_engine()

#Sideba for chat history
st.sidebar.title("Edvisor")

# Initialize chat session if it doesn't exist
if "chat_id" not in st.session_state or st.session_state.chat_id not in chatbot.chat_manager.active_chats:
    st.session_state.chat_id = chatbot.chat_manager.create_new_chat()
    st.session_state.messages = []

#create a new chat button
if st.sidebar.button("New Chat"):
   st.session_state.chat_id = chatbot.chat_manager.create_new_chat()
   st.session_state.messages = []

previous_conversations= chatbot.chat_manager.get_all_conversations()

st.sidebar.title("Previous Conversations")
for chat in previous_conversations:
    col1,col2,col3 = st.sidebar.columns([1,3,1])
    relative_time = Utils.get_relative_time(chat["created_at"])
    with col1:
        if chat['id'] == st.session_state.chat_id:
            st.write("🟢") #green circle for active chats
        else:
            st.write(" ") #empty space for alignment
    with col2:
        if st.button(chat['title'], key=f"chat_{chat['id']}"):
            st.session_state.chat_id = chat['id']
            st.session_state.messages = chatbot.chat_manager.get_chat_history(chat['id'])
            st.caption(f"<small>{relative_time}</small>", unsafe_allow_html=True)
    with col3:
        if st.button("🗑️",key=f"delete_{chat['id']}"):
            chatbot.chat_manager.del_conversation(chat['id'])
            if st.session_state.chat_id == chat['id']:
                st.session_state.chat_id = chatbot.chat_manager.create_new_chat()
                st.session_state.messages = []
            st.rerun()

        

# Create a container for the chat messages
chat_container = st.container()

# User input handling
user_query = st.chat_input("Message Edvisor")

if user_query:
    if user_query.strip() == "":
        st.warning("Please enter a non-empty message.")
    else:
        # Add user message to the chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Generate bot response
        response = chatbot.generate_response(st.session_state.chat_id, user_query)
        
        # Add bot response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Update the chat history in the ChatManager
        chatbot.chat_manager.add_message(st.session_state.chat_id, "user", user_query)
        chatbot.chat_manager.add_message(st.session_state.chat_id, "assistant", response)

        # Force a rerun to update the UI
        st.rerun()

# Display chat history and new messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])