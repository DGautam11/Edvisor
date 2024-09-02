import json
import os
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class OAuthStateStorage:
    def __init__(self, file_path='oauth_states.json'):
        self.file_path = file_path

    def save_state(self, state):
        states = self.load_states()
        states[state] = (datetime.now() + timedelta(minutes=10)).isoformat()
        with open(self.file_path, 'w') as f:
            json.dump(states, f)
        logger.debug(f"Saved state: {state}")

    def validate_state(self, state):
        states = self.load_states()
        if state in states:
            expiration = datetime.fromisoformat(states[state])
            if datetime.now() < expiration:
                del states[state]
                with open(self.file_path, 'w') as f:
                    json.dump(states, f)
                logger.debug(f"Validated state: {state}")
                return True
        logger.debug(f"Failed to validate state: {state}")
        return False

    def load_states(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                return json.load(f)
        return {}