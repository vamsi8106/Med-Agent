"""Password hashing and JWT access tokens.

Password hashing uses stdlib PBKDF2-HMAC-SHA256 (no extra native dependency
like bcrypt/argon2) with a random salt per password, stored as "salt$hash".
"""

import hashlib
import hmac
import os
from datetime import UTC, datetime, timedelta

import jwt

from medagent.core.exceptions import AuthError

_PBKDF2_ITERATIONS = 600_000


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERATIONS)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    try:
        salt_hex, digest_hex = hashed.split("$", 1)
    except ValueError:
        return False
    salt = bytes.fromhex(salt_hex)
    expected = bytes.fromhex(digest_hex)
    actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERATIONS)
    return hmac.compare_digest(actual, expected)


def create_access_token(subject: str, secret_key: str, algorithm: str, expire_minutes: int) -> str:
    now = datetime.now(UTC)
    payload = {"sub": subject, "iat": now, "exp": now + timedelta(minutes=expire_minutes)}
    return jwt.encode(payload, secret_key, algorithm=algorithm)


def decode_access_token(token: str, secret_key: str, algorithm: str) -> str:
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Access token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError(f"Invalid access token: {exc}") from exc

    subject = payload.get("sub")
    if not subject:
        raise AuthError("Access token missing subject")
    return subject
