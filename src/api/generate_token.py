import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.middleware import create_jwt_token

# This script generates a JWT token for a given username.
def generate_token(username: str) -> str:
    """
    Generate a JWT token for a given username.
    """
    return create_jwt_token(username)


if __name__ == "__main__":
    print(generate_token("test_user"))