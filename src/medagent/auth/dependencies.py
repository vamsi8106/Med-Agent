"""FastAPI auth dependencies: Bearer JWT for HTTP routes, query-token for WebSocket."""

from fastapi import Depends, HTTPException, Request, WebSocket
from fastapi.security import OAuth2PasswordBearer

from medagent.auth.security import decode_access_token
from medagent.core.exceptions import AuthError
from medagent.core.models import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


async def _resolve_user(app_state: object, token: str) -> User:
    try:
        username = decode_access_token(
            token,
            app_state.settings.jwt_secret_key,
            app_state.settings.jwt_algorithm,  # type: ignore[attr-defined]
        )
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    record = await app_state.user_store.get_by_username(username)  # type: ignore[attr-defined]
    if record is None:
        raise HTTPException(status_code=401, detail="User not found")
    user, _hashed_password = record
    return user


async def get_current_user(request: Request, token: str = Depends(oauth2_scheme)) -> User:
    return await _resolve_user(request.app.state.medagent, token)


async def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user


async def get_current_user_ws(websocket: WebSocket) -> User | None:
    token = websocket.query_params.get("token")
    if token is None:
        return None
    try:
        return await _resolve_user(websocket.app.state.medagent, token)
    except HTTPException:
        return None
