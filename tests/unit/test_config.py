import pytest
from pydantic import ValidationError

from medagent.core.config import INSECURE_JWT_SECRET, Settings, get_settings


def test_default_settings() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "groq"
    assert settings.log_level == "INFO"


def test_get_settings_singleton() -> None:
    assert get_settings() is get_settings()


def test_development_accepts_the_placeholder_secret() -> None:
    settings = Settings(_env_file=None, app_env="development", jwt_secret_key=INSECURE_JWT_SECRET)
    assert settings.jwt_secret_key == INSECURE_JWT_SECRET


@pytest.mark.parametrize("secret", [INSECURE_JWT_SECRET, "short-secret"])
def test_production_refuses_a_weak_jwt_secret(secret: str) -> None:
    with pytest.raises(ValidationError, match="JWT_SECRET_KEY"):
        Settings(_env_file=None, app_env="production", jwt_secret_key=secret)


def test_production_refuses_a_short_bootstrap_password() -> None:
    with pytest.raises(ValidationError, match="ADMIN_BOOTSTRAP_PASSWORD"):
        Settings(
            _env_file=None,
            app_env="production",
            jwt_secret_key="x" * 40,
            admin_bootstrap_password="short",
        )


def test_production_accepts_strong_settings() -> None:
    settings = Settings(
        _env_file=None,
        app_env="production",
        jwt_secret_key="x" * 40,
        admin_bootstrap_password="a-long-enough-password",
    )
    assert settings.app_env == "production"
