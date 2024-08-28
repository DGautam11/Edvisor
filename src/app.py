import streamlit as st
from engine import Engine
from utils import Utils

# Set the page configuration
st.set_page_config(page_title='Edvisor', page_icon='🎓')
st.title('Edvisor 🤖')
st.write('Chatbot for Finland Study and Visa Services')

# Initialize the chatbot engine
@st.cache_resource
def initialize_engine():
    return Engine()

chatbot = initialize_engine()

# Sidebar for chat history
st.sidebar.title("Edvisor")

# Initialize session state variables
if "chat_id" not in st.session_state:
    st.session_state.chat_id = chatbot.chat_manager.create_new_chat()
    st.session_state.messages = []

# Create a new chat button
if st.sidebar.button("New Chat"):
    st.session_state.chat_id = chatbot.chat_manager.create_new_chat()
    st.session_state.messages = []
    st.rerun()

# Display previous conversations
previous_conversations = chatbot.chat_manager.get_all_conversations()

st.sidebar.title("Previous Conversations")
for chat in previous_conversations:
    col1, col2, col3 = st.sidebar.columns([1, 3, 1])
    relative_time = Utils.get_relative_time(chat["created_at"])
    with col1:
        if chat['id'] == st.session_state.chat_id:
            st.write("🟢")  # Green circle for active chats
        else:
            st.write(" ")  # Empty space for alignment
    with col2:
        # Use button for clickable text with embedded caption
        if st.button(f"{chat['title']}", key=f"chat_{chat['id']}", use_container_width=True):
            st.write(f"{relative_time}")
            st.session_state.chat_id = chat['id']
            st.session_state.messages = chatbot.chat_manager.get_chat_history(chat['id'])
            st.rerun()
    with col3:
        # Use button for delete action, styled as a red trash icon
        if st.button("🗑️", key=f"delete_{chat['id']}", help="Delete this conversation", use_container_width=True):
            chatbot.chat_manager.del_conversation(chat['id'])
            if st.session_state.chat_id == chat['id']:
                st.session_state.chat_id = chatbot.chat_manager.create_new_chat()
                st.session_state.messages = []
            st.rerun()

# Create a container for the chat messages
chat_container = st.container()

# Function to process user input
def process_user_input(user_query):
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Add user message to the chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Show a "thinking" message
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("Thinking...")
    
    # Generate bot response
    response = chatbot.generate_response(st.session_state.chat_id, user_query)
    
    # Remove the "thinking" message and display the bot's response
    thinking_placeholder.empty()
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add bot response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Update the chat history in the ChatManager
    chatbot.chat_manager.add_message(st.session_state.chat_id, "user", user_query)
    chatbot.chat_manager.add_message(st.session_state.chat_id, "assistant", response)

# Display all messages within the chat container
with chat_container:
    # Display existing chat history
    if not st.session_state.messages:
        st.info("No messages yet. Start a conversation!")
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# User input handling
user_query = st.chat_input("Message Edvisor")
if user_query:
    process_user_input(user_query)
    st.rerun()