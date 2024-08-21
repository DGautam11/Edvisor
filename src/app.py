import streamlit as st
from engine import Engine

# Set the page configuration
st.set_page_config(page_title='Edvisor', page_icon='🎓')
st.title('Edvisor 🤖')
st.write('Chatbot for Finland Study and Visa Services')

# Initialize the chatbot engine
chatbot = Engine()

# Placeholder for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# User input handling
user_query = st.text_input("Message EDVISOR")

if user_query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
      st.markdown(user_query)

    with st.chat_message("assistant"):
      # Generate response from chatbot
      response = chatbot.generate_response(st.session_state.messages)
      st.markdown(response)
      st.session_state.messages.append({"role": "assistant", "content": response})