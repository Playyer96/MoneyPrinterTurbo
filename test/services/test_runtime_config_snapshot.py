from app.config import config


def test_runtime_config_snapshot_isolated_from_later_changes():
    key = "runtime_snapshot_test_key"
    original = config.app.get(key)
    try:
        config.app[key] = "submitted"
        snapshot = config.capture_runtime_config()
        config.app[key] = "later"

        with config.use_runtime_config_snapshot(snapshot):
            assert config.app[key] == "submitted"
            config.app[key] = "worker-only"

        assert config.app[key] == "later"
    finally:
        if original is None:
            config.app.pop(key, None)
        else:
            config.app[key] = original
