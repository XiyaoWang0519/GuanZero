"""Provider contract and spend/ownership boundaries, without real credentials."""
from __future__ import annotations

import io
import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from infra import runpod


class Response(io.BytesIO):
    def __init__(self, value):
        super().__init__(json.dumps(value).encode() if value is not None else b"")


class QueueOpener:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request.get_method(), request.full_url,
                           json.loads(request.data) if request.data else None,
                           dict(request.header_items()), timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            response = response(self.calls[-1])
        return Response(response)


def account():
    return {"data": {"myself": {"id": "user_test", "clientBalance": 18,
                                "currentSpendPerHr": 0, "spendLimit": 80,
                                "isAutoPayEnabled": False, "apiKey": "must-not-print"}}}


def catalog(price=0.99, availability="LOW"):
    return {"gpus": [{"id": "NVIDIA GeForce RTX 5090", "name": "RTX 5090",
                      "memory": 32, "availability": availability,
                      "price": {"secure": price, "community": 0.69},
                      "dataCenters": [{"id": "EU-RO-1", "availability": availability}],
                      "env": {"SECRET": "must-not-print"}}]}


def pod_response(call):
    payload = call[2]
    return {"id": "podnew123", "name": payload["name"], "consumerUserId": "user_test",
            "costPerHr": 0.99, "gpuCount": 1, "desiredStatus": "RUNNING",
            "imageName": payload["imageName"], "publicIp": "8.8.8.8",
            "portMappings": {"22": 23456}, "apiKey": "must-not-print",
            "env": {"RUNPOD_API_KEY": "must-not-print"}}


def create_args(tmp_path):
    return dict(path=tmp_path / "pod.json", gpu_id="NVIDIA GeForce RTX 5090",
                public_key="ssh-ed25519 AAAATEST synthetic-test-key", max_hourly_usd=1.25,
                budget_usd=5, max_hours=2, clock=lambda: 1000.0)


def create_fixture(tmp_path):
    opener = QueueOpener(account(), [], catalog(), pod_response)
    client = runpod.Client("fake-api-secret", opener=opener)
    result = runpod.create(client, **create_args(tmp_path))
    return client, opener, result


def test_http_key_is_header_only_and_errors_hide_bodies():
    opener = QueueOpener(HTTPError(runpod.REST + "/pods", 401, "fake-api-secret", {},
                                  io.BytesIO(b"fake-api-secret")))
    client = runpod.Client("fake-api-secret", opener=opener)
    with pytest.raises(runpod.ProviderError, match="RunPod HTTP 401") as error:
        client.pods()
    call = opener.calls[0]
    assert call[3]["Authorization"] == "Bearer fake-api-secret"
    assert "fake-api-secret" not in call[1]
    assert "fake-api-secret" not in str(error.value)
    assert call[4] <= 30


@pytest.mark.parametrize("url", ["https://attacker.invalid/pods", "https://rest.runpod.io.evil/v1/pods",
                                  runpod.REST + "/pods?api_key=secret"])
def test_rejects_unapproved_destinations_and_query_keys(url):
    opener = QueueOpener()
    with pytest.raises(runpod.RunPodError):
        runpod.Client("secret", opener=opener).request("GET", url)
    assert not opener.calls


def test_redirects_never_forward_authorization():
    with pytest.raises(runpod.ProviderError):
        runpod.NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.invalid")


def test_explicit_env_file_overrides_inherited_key_without_shell_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "old-key")
    path = tmp_path / ".env"
    path.write_text("UNRELATED_SECRET=nope\nexport RUNPOD_API_KEY='selected-key' # comment\n")
    assert runpod.load_key(path) == "selected-key"
    assert runpod.load_key() == "old-key"
    assert "UNRELATED_SECRET" not in __import__("os").environ
    path.write_text("RUNPOD_API_KEY=one\nRUNPOD_API_KEY=two\n")
    with pytest.raises(runpod.RunPodError, match="one RUNPOD_API_KEY"):
        runpod.load_key(path)


def test_create_sends_v1_contract_and_persists_secret_free_identity(tmp_path):
    client, opener, result = create_fixture(tmp_path)
    manifest = runpod.read_manifest(tmp_path / "pod.json")
    assert manifest["deadline_epoch"] == 8200
    assert manifest["actual_gpu_hourly_usd"] == 0.99
    assert manifest["estimated_total_hourly_usd"] < 1.25
    assert manifest["account_id"] == "user_test"
    assert manifest["pod_id"] == "podnew123"
    assert manifest["name"].startswith("guanzero-test-")
    payload = opener.calls[-1][2]
    assert opener.calls[-1][:2] == ("POST", runpod.REST + "/pods")
    assert payload["gpuCount"] == 1
    assert payload["interruptible"] is False
    assert payload["ports"] == ["22/tcp"]
    assert payload["minVCPUPerGPU"] == 4
    assert payload["minRAMPerGPU"] == 16
    assert payload["allowedCudaVersions"] == ["12.8", "12.9", "13.0"]
    assert payload["env"]["PUBLIC_KEY"] == payload["env"]["SSH_PUBLIC_KEY"]
    assert result["pod"]["ssh"] == {"host": "8.8.8.8", "port": 23456, "user": "root"}
    assert "must-not-print" not in json.dumps(result)
    assert "fake-api-secret" not in (tmp_path / "pod.json").read_text()


@pytest.mark.parametrize("price,available", [(1.30, "LOW"), (0.99, "NONE"), (None, "LOW")])
def test_quote_or_availability_rejection_never_creates_pod(tmp_path, price, available):
    opener = QueueOpener(account(), [], catalog(price, available))
    with pytest.raises(runpod.RunPodError):
        runpod.create(runpod.Client("fake-key", opener=opener), **create_args(tmp_path))
    assert not any(method == "POST" and url == runpod.REST + "/pods"
                   for method, url, *_ in opener.calls)
    assert not (tmp_path / "pod.json").exists()


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_spend_caps_fail_before_provider_calls(tmp_path, value):
    opener = QueueOpener()
    args = create_args(tmp_path)
    args["budget_usd"] = value
    with pytest.raises(runpod.RunPodError):
        runpod.create(runpod.Client("fake-key", opener=opener), **args)
    assert not opener.calls


def test_ambiguous_create_is_not_retried_and_blocks_duplicate_create(tmp_path):
    opener = QueueOpener(account(), [], catalog(), URLError("fake-api-secret"))
    client = runpod.Client("fake-api-secret", opener=opener)
    with pytest.raises(runpod.RunPodError, match="creation not confirmed") as error:
        runpod.create(client, **create_args(tmp_path))
    assert "fake-api-secret" not in str(error.value)
    manifest = json.loads((tmp_path / "pod.json").read_text())
    assert manifest["state"] == "creation_uncertain"
    assert manifest["name"].startswith("guanzero-test-")
    with pytest.raises(runpod.RunPodError, match="already exists"):
        runpod.create(client, **create_args(tmp_path))
    assert len(opener.calls) == 4


def test_allocated_price_above_cap_is_deleted_after_id_saved(tmp_path, monkeypatch):
    def expensive(call):
        pod = pod_response(call)
        pod["costPerHr"] = 1.5
        return pod
    observed = []
    def terminate(client, path):
        observed.append(runpod.read_manifest(path)["pod_id"])
        return {"confirmed_gone": True}
    monkeypatch.setattr(runpod, "teardown", terminate)
    opener = QueueOpener(account(), [], catalog(), expensive)
    with pytest.raises(runpod.RunPodError, match="termination confirmed"):
        runpod.create(runpod.Client("fake-key", opener=opener), **create_args(tmp_path))
    assert observed == ["podnew123"]


@pytest.mark.parametrize("field,value", [("id", "otherpod123"), ("name", "unrelated"),
                                        ("consumerUserId", "another-user")])
def test_delete_refuses_nonmatching_live_ownership(tmp_path, field, value):
    client, opener, _ = create_fixture(tmp_path)
    pod = pod_response(opener.calls[-1])
    pod[field] = value
    opener.responses.append(pod)
    with pytest.raises(runpod.RunPodError, match="identity does not match"):
        runpod.teardown(client, tmp_path / "pod.json")
    assert not any(call[0] == "DELETE" for call in opener.calls)


class Clock:
    def __init__(self, now=1000.0):
        self.now = now
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.now += duration


def not_found():
    return HTTPError(runpod.REST + "/pods/podnew123", 404, "missing", {}, None)


def test_delete_retries_transient_provider_failure_and_confirms_gone(tmp_path):
    client, opener, _ = create_fixture(tmp_path)
    pod = pod_response(opener.calls[-1])
    opener.responses.extend([pod, HTTPError("", 503, "unavailable", {}, None),
                             pod, None, not_found()])
    clock = Clock()
    result = runpod.teardown(client, tmp_path / "pod.json", clock=clock, sleep=clock.sleep)
    assert result["confirmed_gone"]
    assert sum(call[0] == "DELETE" for call in opener.calls) == 2
    assert runpod.read_manifest(tmp_path / "pod.json")["state"] == "terminated"


def test_guard_reserves_provider_shutdown_time_before_deadline(tmp_path):
    client, opener, _ = create_fixture(tmp_path)
    pod = pod_response(opener.calls[-1])
    opener.responses.extend([pod, pod, None, not_found()])
    clock = Clock(now=8065)
    result = runpod.guard(client, tmp_path / "pod.json", poll_seconds=500,
                          clock=clock, sleep=clock.sleep)
    assert result["confirmed_gone"]
    assert 8080 <= clock.now < 8200
    assert max(clock.sleeps) <= 30
    assert sum(call[0] == "DELETE" for call in opener.calls) == 1


def test_guard_exits_after_another_process_confirms_deletion(tmp_path):
    client, opener, _ = create_fixture(tmp_path)
    path = tmp_path / "pod.json"
    pod = pod_response(opener.calls[-1])
    opener.responses.extend([pod, not_found()])
    clock = Clock()
    def sleep_and_finish(duration):
        clock.sleep(duration)
        manifest = runpod.read_manifest(path)
        manifest["state"] = "terminated"
        runpod.write_manifest(path, manifest)
    result = runpod.guard(client, path, clock=clock, sleep=sleep_and_finish)
    assert result["confirmed_gone"]
    assert clock.now == 1015
    assert not any(call[0] == "DELETE" for call in opener.calls)


def test_existing_id_and_tampered_manifest_cannot_authorize_delete(tmp_path):
    _, _, _ = create_fixture(tmp_path)
    path = tmp_path / "pod.json"
    manifest = json.loads(path.read_text())
    manifest["preexisting_pod_ids"] = [manifest["pod_id"]]
    runpod.write_manifest(path, manifest)
    with pytest.raises(runpod.RunPodError, match="preexisting"):
        runpod.read_manifest(path)


def test_cli_configuration_errors_never_dump_env_contents(tmp_path, capsys):
    path = tmp_path / ".env"
    path.write_text("RUNPOD_API_KEY='unterminated-secret\n")
    assert runpod.main(["--env-file", str(path), "inspect"]) == 1
    output = capsys.readouterr()
    assert "unterminated-secret" not in output.err
    assert "invalid RUNPOD_API_KEY assignment" in output.err
