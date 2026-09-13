from medagent.core.config import Settings, get_settings


def test_default_settings() -> None:
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "groq"
    assert settings.log_level == "INFO"


def test_get_settings_singleton() -> None:
    assert get_settings() is get_settings()
