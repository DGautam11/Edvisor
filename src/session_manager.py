import streamlit as st
from datetime import datetime, timedelta
from typing import Optional

class SessionManager:
    @staticmethod
    def set_session(user_email: str):
        st.session_state.user_email = user_email
        st.session_state.last_activity = datetime.now()

    @staticmethod
    def get_session() -> Optional[str]:
        if 'user_email' in st.session_state and 'last_activity' in st.session_state:
            if datetime.now() - st.session_state.last_activity <= timedelta(weeks=1):
                st.session_state.last_activity = datetime.now()
                return st.session_state.user_email
            else:
                SessionManager.clear_session()
        return None

    @staticmethod
    def clear_session():
        if 'user_email' in st.session_state:
            del st.session_state.user_email
        if 'last_activity' in st.session_state:
            del st.session_state.last_activity