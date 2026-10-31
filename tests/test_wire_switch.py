"""AFM_LITHO_WIRE_VERSION: default 2, "3" selects v3, anything else is refused, the keeper passes it on."""

import importlib

import pytest

from afm_litho import wire_version
from afm_litho.keeper import cli
from afm_litho.keeper.process import PopenHandle


def test_default_is_2():
    assert wire_version.from_env({}) == 2
    assert wire_version.from_env({wire_version.ENV: ""}) == 2


def test_3_selects_v3():
    assert wire_version.from_env({wire_version.ENV: "3"}) == 3
    assert wire_version.from_env({wire_version.ENV: "2"}) == 2


@pytest.mark.parametrize("bad", ["4", "v3", "three", "0", "3.0"])
def test_bad_value_is_refused(bad):
    with pytest.raises(ValueError, match="AFM_LITHO_WIRE_VERSION"):
        wire_version.from_env({wire_version.ENV: bad})


def test_wire_module_reads_the_env_once_at_import(monkeypatch):
    from afm_litho.hardware import wire
    try:
        monkeypatch.setenv(wire_version.ENV, "3")
        assert importlib.reload(wire).WIRE_VERSION == 3
        monkeypatch.setenv(wire_version.ENV, "9")
        with pytest.raises(ValueError):
            importlib.reload(wire)
    finally:
        monkeypatch.delenv(wire_version.ENV, raising=False)
        importlib.reload(wire)


def test_keeper_passes_it_to_the_service_child_and_its_restarts(monkeypatch):
    monkeypatch.setenv(wire_version.ENV, "3")
    specs = {s.name: s for s in cli.child_specs(7461, "127.0.0.1", recorder_port=1)}
    assert specs["service"].env[wire_version.ENV] == "3"
    seen = []

    class FakeProc:
        pid = 1

    def fake_popen(argv, **kw):
        seen.append(kw["env"])
        return FakeProc()

    monkeypatch.setattr("afm_litho.keeper.process.subprocess.Popen", fake_popen)
    spec = specs["service"]
    PopenHandle(spec)
    PopenHandle(spec, spec.restart_args)     # the restart uses the same spec
    assert [e[wire_version.ENV] for e in seen] == ["3", "3"]


def test_keeper_refuses_a_bad_value(monkeypatch, capsys):
    monkeypatch.setenv(wire_version.ENV, "7")
    assert cli.main(["--no-browser"]) == 1
    assert "AFM_LITHO_WIRE_VERSION" in capsys.readouterr().err
