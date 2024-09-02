import streamlit as st
from engine import Engine
from utils import Utils
from session_manager import SessionManager
import urllib.parse

# Set the page configuration
st.set_page_config(page_title='Edvisor', page_icon='🎓')
st.title('Edvisor 🤖')
st.write('Chatbot for Finland Study and Visa Services')

# Initialize the chatbot engine
@st.cache_resource
def initialize_engine():
    return Engine()

chatbot = initialize_engine()

# Check for OAuth callback
if "code" in st.query_params and "state" in st.query_params:
    try:
        # Properly construct the authorization response URL
        authorization_response = f"?{urllib.parse.urlencode({'code': st.query_params['code'], 'state': st.query_params['state']})}"

        # Check if the state is in session state
        if 'oauth_state' not in st.session_state:
            st.error("OAuth state not found. Please try logging in again.")
            st.stop()

        state = st.session_state.oauth_state

        # Fetch user info using the authorization code and state
        user_info = chatbot.get_user_info(authorization_response, state)

        if user_info and 'email' in user_info:
            # Save the user's email in the session and clear the OAuth state
            SessionManager.set_session(user_info['email'])
            del st.session_state.oauth_state  # Clear the state after use
            st.experimental_rerun()
        else:
            st.error("Failed to get user information. Please try again.")
            st.stop()

    except Exception as e:
        st.error(f"An error occurred during authentication: {str(e)}")
        st.stop()

# Check for existing session
user_email = SessionManager.get_session()

# Authentication check
if not user_email:
    st.markdown(
        """
        <style>
        .google-button {
            background: white;
            color: #444;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            border: thin solid #888;
            box-shadow: 1px 1px 1px grey;
            border-radius: 5px;
        }
        .google-button:hover {
            cursor: pointer;
        }

        span.icon {
            display: inline-block;
            vertical-align: middle;
            width: 42px;
            height: 42px;
        }
        span.buttonText {
            display: inline-block;
            vertical-align: middle;
            padding-left: 42px;
            padding-right: 42px;
            font-size: 14px;
            font-weight: bold;
            font-family: 'Roboto', sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.write("Please sign in to start chatting.")

    # Get authorization URL from the chatbot
    auth_url, state = chatbot.get_authorization_url()
    
    # Store the state in session state
    st.session_state.oauth_state = state
    
    # Display the sign-in button
    st.markdown(f'''
        <a href="{auth_url}" class="google-button">
        <span class="icon"><img src="./identity/g-normal.png"></span>
        <span class="buttonText">Sign in with Google</span>
        </a>
    ''', unsafe_allow_html=True)

    st.info("Please sign in with your Google account.")

else:
    # Sidebar for chat history
    st.sidebar.title("Edvisor")

    # Initialize session state variables
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = chatbot.chat_manager.create_new_chat(user_email)
        st.session_state.messages = []

    # Create a new chat button
    if st.sidebar.button("New Chat"):
        st.session_state.chat_id = chatbot.chat_manager.create_new_chat(user_email)
        st.session_state.messages = []
        st.rerun()

    st.subheader("Previous Conversations")

    # Display previous conversations
    previous_conversations = chatbot.get_user_chats(user_email)
    for chat in previous_conversations:
        col1, col2, col3, col4 = st.sidebar.columns([1, 2, 1, 1])
        relative_time = Utils.get_relative_time(chat["created_at"])
        with col1:
            if chat['id'] == st.session_state.chat_id:
                st.write("🟢")  # Green circle for active chats
            else:
                st.write(" ")  # Empty space for alignment
        with col2:
            if st.button(f"{chat['title']}", key=f"chat_{chat['id']}", use_container_width=True):
                st.session_state.chat_id = chat['id']
                st.session_state.messages = chatbot.get_chat_history(chat['id'], user_email)
                st.rerun()
        with col3:
            st.write(f"{relative_time}")
        with col4:
            if st.button("🗑️", key=f"delete_{chat['id']}", help="Delete this conversation", use_container_width=True):
                chatbot.delete_chat(chat['id'], user_email)
                if st.session_state.chat_id == chat['id']:
                    st.session_state.chat_id = chatbot.chat_manager.create_new_chat(user_email)
                    st.session_state.messages = []
                st.rerun()

    # Create a container for the chat messages
    chat_container = st.container()

    # Function to process user input
    def process_user_input(user_query, user_email):
        with st.chat_message("user"):
            st.markdown(user_query)
        
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("Thinking...")
        
            response = chatbot.generate_response(st.session_state.chat_id, user_query, user_email)
        
            thinking_placeholder.empty()

            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        chatbot.add_message(st.session_state.chat_id, "user", user_query, user_email)
        chatbot.add_message(st.session_state.chat_id, "assistant", response, user_email)

    # Display all messages within the chat container
    with chat_container:
        if not st.session_state.messages:
            st.info("No messages yet. Start a conversation!")
        else:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    # User input handling
    user_query = st.chat_input("Message Edvisor")
    if user_query:
        process_user_input(user_query, user_email)
        st.rerun()

    if st.sidebar.button("Logout"):
        SessionManager.clear_session()
        st.rerun()
