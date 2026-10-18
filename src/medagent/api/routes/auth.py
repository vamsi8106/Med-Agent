"""Accounts and login."""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordRequestForm

from medagent.api.schemas import CreateUserRequest, Token
from medagent.api.state import AppState, enforce_limit, get_state
from medagent.auth.dependencies import require_admin
from medagent.auth.security import create_access_token, hash_password, verify_password
from medagent.core.exceptions import AuthError
from medagent.core.models import User

router = APIRouter()

# Checked when the username does not exist, so a login for an unknown user takes
# as long as one with a wrong password and timing does not reveal which
# usernames exist.
_DUMMY_PASSWORD_HASH = hash_password("timing-equaliser-not-a-real-password")
_LOGIN_LIMITED = "Too many login attempts; try again later"


@router.post("/admin/users", response_model=User)
async def create_user(
    body: CreateUserRequest,
    _admin: User = Depends(require_admin),
    state: AppState = Depends(get_state),
) -> User:
    # No public registration endpoint: every account (after the one admin
    # seeded from ADMIN_BOOTSTRAP_USERNAME/PASSWORD at startup) is created
    # by an existing admin, so patient data is never open to self-signup.
    try:
        return await state.user_store.create_user(
            body.username, hash_password(body.password), body.role
        )
    except AuthError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/auth/token", response_model=Token)
async def login(
    request: Request,
    form: OAuth2PasswordRequestForm = Depends(),
    state: AppState = Depends(get_state),
) -> Token:
    # Limited before the password is checked, and counted even when it is right,
    # so guessing is capped per account and per source. Behind a reverse proxy,
    # run uvicorn with --proxy-headers so client.host is the real client.
    client_ip = request.client.host if request.client else "unknown"
    enforce_limit(state.login_by_ip, client_ip, limit="login_ip", message=_LOGIN_LIMITED)
    enforce_limit(
        state.login_by_username,
        form.username.lower(),
        limit="login_username",
        message=_LOGIN_LIMITED,
    )

    record = await state.user_store.get_by_username(form.username)
    password_ok = verify_password(form.password, record[1] if record else _DUMMY_PASSWORD_HASH)
    if record is None or not password_ok:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token(
        form.username,
        state.settings.jwt_secret_key,
        state.settings.jwt_algorithm,
        state.settings.jwt_expire_minutes,
    )
    return Token(access_token=token)
