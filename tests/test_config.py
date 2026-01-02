from app.config import Settings


def test_settings_defaults() -> None:
    cfg = Settings()
    assert cfg.lmstudio_base_url.startswith("http")
    assert cfg.session_max_messages > 0
    assert cfg.context_window > 0

