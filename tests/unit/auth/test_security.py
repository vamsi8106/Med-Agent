import pytest

from medagent.auth.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from medagent.core.exceptions import AuthError


def test_hash_and_verify_password_round_trip() -> None:
    hashed = hash_password("s3cret!")
    assert verify_password("s3cret!", hashed) is True
    assert verify_password("wrong", hashed) is False


def test_hash_password_uses_random_salt() -> None:
    assert hash_password("s3cret!") != hash_password("s3cret!")


def test_create_and_decode_access_token_round_trip() -> None:
    token = create_access_token("dr.alpha", "secret", "HS256", expire_minutes=5)
    subject = decode_access_token(token, "secret", "HS256")
    assert subject == "dr.alpha"


def test_decode_access_token_rejects_wrong_secret() -> None:
    token = create_access_token("dr.alpha", "secret", "HS256", expire_minutes=5)
    with pytest.raises(AuthError):
        decode_access_token(token, "wrong-secret", "HS256")


def test_decode_access_token_rejects_expired_token() -> None:
    token = create_access_token("dr.alpha", "secret", "HS256", expire_minutes=-1)
    with pytest.raises(AuthError):
        decode_access_token(token, "secret", "HS256")
