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
        llm_provider="openai_compatible",
        llm_base_url="http://ollama:11434/v1",
        llm_model="qwen2.5:7b",
    )
    assert settings.app_env == "production"


_STRONG = {"_env_file": None, "app_env": "production", "jwt_secret_key": "x" * 40}


def test_production_refuses_groq_without_a_data_agreement() -> None:
    with pytest.raises(ValidationError, match="ALLOW_EXTERNAL_LLM_WITH_PHI"):
        Settings(**_STRONG, llm_provider="groq")


def test_production_allows_groq_when_a_data_agreement_is_asserted() -> None:
    settings = Settings(**_STRONG, llm_provider="groq", allow_external_llm_with_phi=True)
    assert settings.llm_provider == "groq"


@pytest.mark.parametrize(
    "base_url",
    [
        "http://ollama:11434/v1",
        "http://localhost:11434/v1",
        "http://10.0.0.5:8000/v1",
        "http://host.docker.internal:11434/v1",
        "http://gpu-box.local:8000/v1",
    ],
)
def test_production_allows_a_self_hosted_model(base_url: str) -> None:
    settings = Settings(
        **_STRONG, llm_provider="openai_compatible", llm_base_url=base_url, llm_model="m"
    )
    assert settings.llm_base_url == base_url


def test_production_refuses_an_openai_compatible_endpoint_on_a_public_host() -> None:
    with pytest.raises(ValidationError, match="ALLOW_EXTERNAL_LLM_WITH_PHI"):
        Settings(
            **_STRONG,
            llm_provider="openai_compatible",
            llm_base_url="https://api.example-llm.com/v1",
            llm_model="m",
        )
