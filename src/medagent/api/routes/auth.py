"""Accounts and login."""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordRequestForm

from medagent.api.schemas import (
    ChangePasswordRequest,
    CreateUserRequest,
    ResetPasswordRequest,
    Token,
)
from medagent.api.state import AppState, enforce_limit, get_state
from medagent.auth.dependencies import get_current_user, require_admin
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
    # A disabled account gets the same answer as a wrong password.
    if record is None or not password_ok or record[0].disabled:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token(
        form.username,
        state.settings.jwt_secret_key,
        state.settings.jwt_algorithm,
        state.settings.jwt_expire_minutes,
        version=record[0].token_version,
    )
    return Token(access_token=token)


# --- user management. Every action is audited (patient_id null; details hold
# only the target username), and every change that should end existing sessions
# bumps the user's token version.


def _require_found(found: bool) -> None:
    if not found:
        raise HTTPException(status_code=404, detail="Unknown user")


@router.get("/admin/users", response_model=list[User])
async def list_users(
    _admin: User = Depends(require_admin), state: AppState = Depends(get_state)
) -> list[User]:
    return await state.user_store.list_users()


@router.post("/admin/users/{username}/disable", status_code=204)
async def disable_user(
    username: str, admin: User = Depends(require_admin), state: AppState = Depends(get_state)
) -> None:
    if username == admin.username:
        # Would lock the acting admin out mid-session; use another admin account.
        raise HTTPException(status_code=409, detail="You cannot disable your own account")
    _require_found(await state.user_store.set_disabled(username, True))
    await state.audit_log.record(admin, None, "user_disabled", {"target": username})


@router.post("/admin/users/{username}/enable", status_code=204)
async def enable_user(
    username: str, admin: User = Depends(require_admin), state: AppState = Depends(get_state)
) -> None:
    _require_found(await state.user_store.set_disabled(username, False))
    await state.audit_log.record(admin, None, "user_enabled", {"target": username})


@router.post("/admin/users/{username}/reset-password", status_code=204)
async def reset_password(
    username: str,
    body: ResetPasswordRequest,
    admin: User = Depends(require_admin),
    state: AppState = Depends(get_state),
) -> None:
    _require_found(await state.user_store.set_password(username, hash_password(body.new_password)))
    await state.audit_log.record(admin, None, "user_password_reset", {"target": username})


@router.post("/admin/users/{username}/revoke-tokens", status_code=204)
async def revoke_tokens(
    username: str, admin: User = Depends(require_admin), state: AppState = Depends(get_state)
) -> None:
    _require_found(await state.user_store.revoke_tokens(username))
    await state.audit_log.record(admin, None, "user_tokens_revoked", {"target": username})


@router.post("/auth/change-password", status_code=204)
async def change_password(
    body: ChangePasswordRequest,
    user: User = Depends(get_current_user),
    state: AppState = Depends(get_state),
) -> None:
    """Ends every session, including this one: log in again with the new password."""
    # Checking the current password is a guess like a login, so it is limited.
    enforce_limit(
        state.login_by_username,
        user.username.lower(),
        limit="login_username",
        message=_LOGIN_LIMITED,
    )
    record = await state.user_store.get_by_username(user.username)
    if record is None or not verify_password(body.current_password, record[1]):
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    await state.user_store.set_password(user.username, hash_password(body.new_password))
    await state.audit_log.record(user, None, "password_changed")
