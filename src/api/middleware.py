import os
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import jwt
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

load_dotenv()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"


# Crée un token JWT avec un identifiant utilisateur (sub) et une durée de validité.
# La durée de validité par défaut est de 60 minutes.
# Le token est signé avec la clé secrète définie dans les variables d'environnement.
# Le token contient la date d'expiration (exp) et l'identifiant utilisateur (sub).
# Si la clé secrète n'est pas définie, une exception est levée.
def create_jwt_token(user_id: str, expires_delta: int = 60):
    """Crée un token JWT avec un identifiant utilisateur (sub) et une durée de validité."""
    payload = {
        "sub": user_id,
        "exp": datetime.now() + timedelta(minutes=expires_delta)
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token

# This middleware checks for a valid JWT token in the Authorization header of incoming requests.
# If the token is missing, expired, or invalid, it returns a 401 Unauthorized response
class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path.startswith("/predict"):
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                return JSONResponse(status_code=401, content={"detail": "Missing Authorization header"})
            try:
                token = auth_header.split(" ")[1]
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
                request.state.user = payload["sub"]
            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Token expired"})
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})
        return await call_next(request)