"""FastAPI auth dependencies: Bearer JWT for HTTP routes; a token check for the
WebSocket handshake (the token arrives in its first message, see api/routes/ws.py)."""

from fastapi import Depends, HTTPException, Request, WebSocket
from fastapi.security import OAuth2PasswordBearer

from medagent.auth.security import decode_token_claims
from medagent.core.exceptions import AuthError
from medagent.core.models import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


async def _resolve_user(app_state: object, token: str) -> User:
    # app_state is really app.py's AppState, typed as `object` here to avoid
    # auth/ importing upward from the top-level app.py glue module.
    try:
        username, version = decode_token_claims(
            token,
            app_state.settings.jwt_secret_key,  # type: ignore[attr-defined]
            app_state.settings.jwt_algorithm,  # type: ignore[attr-defined]
        )
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    record = await app_state.user_store.get_by_username(username)  # type: ignore[attr-defined]
    # Unknown, disabled, or revoked (password changed / reset / revoke-tokens):
    # one answer for all, so the response doesn't say which.
    if record is None or record[0].disabled or record[0].token_version != version:
        raise HTTPException(status_code=401, detail="Invalid access token")
    user, _hashed_password = record
    return user


async def get_current_user(request: Request, token: str = Depends(oauth2_scheme)) -> User:
    return await _resolve_user(request.app.state.medagent, token)


async def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user


async def user_for_ws_token(websocket: WebSocket, token: str) -> User | None:
    try:
        return await _resolve_user(websocket.app.state.medagent, token)
    except HTTPException:
        return None
