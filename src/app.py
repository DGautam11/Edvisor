import streamlit as st
from engine import Engine
from utils import Utils

# Set the page configuration
st.set_page_config(page_title='Edvisor', page_icon='🎓')
st.title('Edvisor 🤖')
st.write('Chatbot for Finland Study and Visa Services')

# Custom CSS for styling
st.markdown("""
<style>
    .conversation-title {
        cursor: pointer;
        color: #4F8BF9;
        background: none;
        border: none;
        padding: 0;
        font: inherit;
        text-align: left;
    }
    .red-trash-button {
        color: white;
        background-color: red;
        border: none;
        border-radius: 5px;
        padding: 5px 10px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

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

if "widget_clicked" not in st.session_state:
    st.session_state.widget_clicked = None

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
        # Use markdown for clickable text
        st.markdown(f"<div class='conversation-title' onclick=\"handleClick('{chat['id']}')\">{chat['title']}</div>", unsafe_allow_html=True)
        st.caption(f"{relative_time}")
    with col3:
        # Use custom HTML for red trash button
        st.markdown(f"<button class='red-trash-button' onclick=\"handleDelete('{chat['id']}')\">🗑️</button>", unsafe_allow_html=True)

# JavaScript for handling clicks
st.markdown("""
<script>
function handleClick(chatId) {
    Streamlit.setComponentValue({type: 'select', chatId: chatId});
}
function handleDelete(chatId) {
    Streamlit.setComponentValue({type: 'delete', chatId: chatId});
}
</script>
""", unsafe_allow_html=True)

# Handle chat selection and deletion
if st.session_state.widget_clicked is not None:
    action = st.session_state.widget_clicked
    if action['type'] == 'select':
        st.session_state.chat_id = action['chatId']
        st.session_state.messages = chatbot.chat_manager.get_chat_history(action['chatId'])
        st.session_state.widget_clicked = None
        st.rerun()
    elif action['type'] == 'delete':
        chatbot.chat_manager.del_conversation(action['chatId'])
        if st.session_state.chat_id == action['chatId']:
            st.session_state.chat_id = chatbot.chat_manager.create_new_chat()
            st.session_state.messages = []
        st.session_state.widget_clicked = None
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

# Capture component value changes
if st.session_state.widget_clicked is None:
    component_value = st.empty()
    st.session_state.widget_clicked = component_value.get_component_value()
    if st.session_state.widget_clicked is not None:
        st.rerun()