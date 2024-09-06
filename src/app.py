import streamlit as st
from engine import Engine
from utils import Utils
from session_manager import SessionManager
from auth import OAuth
import logging
import uuid

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the page configuration
st.set_page_config(page_title='Edvisor', page_icon='🎓', layout="wide")
st.title('Edvisor 🤖')
st.write('Chatbot for Finland Study and Visa Services')

# Debug section
debug_container = st.container()

# Initialize the chatbot engine and OAuth
@st.cache_resource
def initialize_engine():
    return Engine()

@st.cache_resource
def initialize_oauth():
    return OAuth()

chatbot = initialize_engine()
oauth = initialize_oauth()

# Check for existing session
user_email = SessionManager.get_session()

# Authentication check
if not user_email:
    st.write("Please sign in to start chatting.")
    
    # Debug display
    with debug_container:
        st.subheader("Debug Information")
        st.write(f"Session State: {st.session_state}")
        st.write(f"Query Params: {dict(st.query_params)}")

    # Generate a new state for each auth attempt
    state = str(uuid.uuid4())
    auth_url = oauth.get_authorization_url(state)
    logger.debug(f"Generated new OAuth state: {state}")

    # Create a button for sign-in
    if st.button("Sign in with Google"):
        logger.debug(f"Sign-in button clicked. Redirecting to: {auth_url}")
        st.markdown(f'<meta http-equiv="refresh" content="0;url={auth_url}">', unsafe_allow_html=True)

    # Check if we've been redirected back from Google
    if "code" in st.query_params and "state" in st.query_params:
        try:
            # Extract the authorization code and state
            code = st.query_params["code"]
            received_state = st.query_params["state"]
            logger.debug(f"Received OAuth callback. Code: {code}, State: {received_state}")

            # Debug display
            with debug_container:
                st.write(f"Received Code: {code}")
                st.write(f"Received State: {received_state}")

            # Construct the authorization response
            authorization_response = f"?code={code}&state={received_state}"

            # Fetch user info
            user_info = oauth.get_user_info(authorization_response)

            # Debug display
            with debug_container:
                st.write(f"User Info: {user_info}")

            if user_info and 'email' in user_info:
                logger.info(f"Successfully authenticated user: {user_info['email']}")
                # Save the user's email in the session
                SessionManager.set_session(user_info['email'])
                st.query_params.clear()  # Clear the query parameters
                st.rerun()
            else:
                logger.error("Failed to get user information")
                st.error("Failed to get user information. Please try again.")
                st.stop()

        except Exception as e:
            logger.exception("Error during authentication")
            st.error(f"An error occurred during authentication: {str(e)}")
            with debug_container:
                st.write(f"Exception: {str(e)}")
            st.stop()

else:
    logger.info(f"User already authenticated: {user_email}")
    st.success(f"Logged in as: {user_email}")

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

    st.sidebar.subheader("Previous Conversations")

    # Display previous conversations
    previous_conversations = chatbot.chat_manager.get_all_conversations(user_email)
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
                st.session_state.messages = chatbot.chat_manager.get_chat_history(chat['id'], user_email)
                st.rerun()
        with col3:
            st.write(f"{relative_time}")
        with col4:
            if st.button("🗑️", key=f"delete_{chat['id']}", help="Delete this conversation", use_container_width=True):
                chatbot.chat_manager.del_conversation(chat['id'], user_email)
                if st.session_state.chat_id == chat['id']:
                    st.session_state.chat_id = chatbot.chat_manager.create_new_chat()
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
        
            # Use the LangChain-based Engine to generate a response
            response = chatbot.generate_response(st.session_state.chat_id, user_query, user_email)
        
            thinking_placeholder.empty()

            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

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

# Final debug display
with debug_container:
    st.subheader("Final Debug Information")
    st.write(f"Final Session State: {st.session_state}")
    st.write(f"Final Query Params: {dict(st.query_params)}")