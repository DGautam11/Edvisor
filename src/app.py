import streamlit as st
from engine import Engine
from utils import Utils
from session_manager import SessionManager
from oauth_state_storage import OAuthStateStorage
import urllib.parse
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OAuth State Storage
oauth_state_storage = OAuthStateStorage()

# Set the page configuration
st.set_page_config(page_title='Edvisor', page_icon='🎓')
st.title('Edvisor 🤖')
st.write('Chatbot for Finland Study and Visa Services')

# Initialize the chatbot engine
@st.cache_resource
def initialize_engine():
    return Engine()

chatbot = initialize_engine()

# Check for existing session
user_email = SessionManager.get_session()

# Authentication check
if not user_email:
    st.write("Please sign in to start chatting.")

    # Get authorization URL from the chatbot
    auth_url, state = chatbot.get_authorization_url()
    
    # Store the state in both session state and file storage
    st.session_state.oauth_state = state
    oauth_state_storage.save_state(state)
    logger.debug(f"Generated new OAuth state: {state}")
    
    # Create a button for sign-in
    if st.button("Sign in with Google"):
        logger.debug(f"Sign-in button clicked. Redirecting to: {auth_url}")
        st.markdown(f'<meta http-equiv="refresh" content="0;url={auth_url}">', unsafe_allow_html=True)
        st.stop()

    # Check if we've been redirected back from Google
    params = st.query_params
    if "code" in params and "state" in params:
        try:
            # Extract the authorization code and state
            code = params.get("code")
            received_state = params.get("state")
            logger.debug(f"Received OAuth callback. State: {received_state}")

            # Verify the state using both session state and file storage
            session_state_valid = received_state == st.session_state.get('oauth_state')
            file_state_valid = oauth_state_storage.validate_state(received_state)
            logger.debug(f"Session state valid: {session_state_valid}, File state valid: {file_state_valid}")

            if not (session_state_valid or file_state_valid):
                logger.error("Invalid OAuth state")
                st.error("Invalid state. Please try logging in again.")
                st.stop()

            # Construct the authorization response
            authorization_response = f"?code={code}&state={received_state}"

            # Fetch user info
            user_info = chatbot.get_user_info(authorization_response, received_state)

            if user_info and 'email' in user_info:
                logger.info(f"Successfully authenticated user: {user_info['email']}")
                # Save the user's email in the session and clear the OAuth state
                SessionManager.set_session(user_info['email'])
                if 'oauth_state' in st.session_state:
                    del st.session_state.oauth_state
                st.query_params.clear()  # Clear the query parameters
                st.rerun()
            else:
                logger.error("Failed to get user information")
                st.error("Failed to get user information. Please try again.")
                st.stop()

        except Exception as e:
            logger.exception("Error during authentication")
            st.error(f"An error occurred during authentication: {str(e)}")
            st.stop()

else:
    logger.info(f"User already authenticated: {user_email}")
    # Sidebar for chat history
    st.sidebar.title("Edvisor")

    # Initialize session state variables
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = chatbot.create_new_chat(user_email)
        st.session_state.messages = []

    # Create a new chat button
    if st.sidebar.button("New Chat"):
        st.session_state.chat_id = chatbot.create_new_chat(user_email)
        st.session_state.messages = []
        st.rerun()

    st.sidebar.subheader("Previous Conversations")

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
                    st.session_state.chat_id = chatbot.create_new_chat(user_email)
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
        logger.info(f"User logged out: {user_email}")
        SessionManager.clear_session()
        st.query_params.clear()  # Clear any query parameters
        st.rerun()