from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from config import Config
import os

class OAuth:
    def __init__(self):
        os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'  # Only for development, not recommended for production.
        self.config = Config()
        self.flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": self.config.client_id,
                    "auth_uri": self.config.auth_uri,
                    "token_uri": self.config.token_uri,
                    "client_secret": self.config.client_secret,
                    "redirect_uris": self.config.redirect_uris,
                }
            },
            scopes=self.config.scopes,
        )
        self.state = None  # Initialize the state

    def get_authorization_url(self):
        self.flow.redirect_uri = self.config.redirect_uris[0]  # Use the first redirect URI
        authorization_url, self.state = self.flow.authorization_url(prompt='consent')
        return authorization_url

    def get_user_info(self, authorization_response, state):
        if state != self.state:
            raise Exception("Invalid state")
        self.flow.fetch_token(authorization_response=authorization_response)
        credentials = self.flow.credentials
        service = build('oauth2', 'v2', credentials=credentials)
        user_info = service.userinfo().get().execute()
        return user_info
