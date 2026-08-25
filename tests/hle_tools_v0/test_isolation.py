from pathlib import Path

import yaml


COMPOSE = Path("solvers/hle-tools-v0/sandbox/compose.yaml")


def test_generated_code_sandbox_is_host_isolated():
    compose = yaml.safe_load(COMPOSE.read_text())
    service = compose["services"]["default"]
    assert service["network_mode"] == "none"
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in service["security_opt"]
    assert service["user"] != "0"
    assert "privileged" not in service
    assert "volumes" not in service
    rendered = COMPOSE.read_text()
    assert "/var/run/docker.sock" not in rendered
    assert "network_mode: host" not in rendered
