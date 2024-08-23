import streamlit as st
from engine import Engine
import datetime

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

#creatte a new chat button
if st.sidebar.button("New Chat"):
   st.session_state.chat_id = chatbot.chat_manager.create_new_chat()
   st.session_state.messages = []

previous_conversations= chatbot.chat_manager.get_all_conversations()

st.sidebar.title("Previous Conversations")
for chat in previous_conversations:
    col1,col2 = st.sidebar.columns([4,1])
    date_str = datetime.fromisoformat(chat["created_at"]).strftime("%Y-%m-%d")
    if col1.button(f"{date_str}-{chat['title']},key=chat['id']"):
        st.session_state.chat_id = chat['id']
        st.session_state.messages = chatbot.chat_manager.get_chat_history(chat['id'])
    if col2.button("🗑️",key=f"delete_{chat['id']}"):
        chatbot.chat_manager.del_conversation(chat['id'])
        if st.session_state.chat_id == chat['id']:
            st.session_state.chat_id = chatbot.chat_manager.create_new_chat()
            st.session_state.messages = []
        st.rerun()

        
# Initialize chat session if it doesn't exist
if "chat_id" not in st.session_state or st.session_state.chat_id not in chatbot.chat_manager.active_chats:
    st.session_state.chat_id = chatbot.chat_manager.create_new_chat()
    st.session_state.messages = []

# Create a container for the chat messages
chat_container = st.container()

# User input handling
user_query = st.chat_input("Message Edvisor")

if user_query:
    # Check if this is the first message in a new chat
    is_new_chat = len(st.session_state.messages) == 0
    
    # Add user message to the chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Generate bot response
    response = chatbot.generate_response(st.session_state.chat_id, user_query)
    
    # Add bot response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Update the chat history in the ChatManager
    chatbot.chat_manager.add_message(st.session_state.chat_id, "user", user_query)
    chatbot.chat_manager.add_message(st.session_state.chat_id, "assistant", response)

    # Force a rerun only if this was the first message in a new chat
    if is_new_chat:
        st.rerun()

# Display chat history and new messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])