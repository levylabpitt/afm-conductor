"""panel.example.org as an alias for the loopback, and the guards on it.

The chosen plan: a public DNS A record for panel.example.org pointing
at 127.0.0.1, so every machine runs its own copy at a stable URL with nothing
reachable from the network. TLS is required because Google Identity Services
allows plain http origins only for the literal name `localhost`.

The guard that matters is `resolves_to_loopback`. If the record is missing,
or somebody later repoints it at a real machine, the app must fall back to
127.0.0.1 rather than sending the browser to a different computer's
instrument. `every address must be loopback`, not `any`, because a name that
resolves to both would otherwise pass the check here and then connect to the
routable address in the browser.
"""

from __future__ import annotations

import socket

import pytest

from afm_litho import site


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for k in ("AFM_LITHO_TLS_CERT", "AFM_LITHO_TLS_KEY"):
        monkeypatch.delenv(k, raising=False)
    # There is no default name, so the tests that exercise one configure it.
    monkeypatch.setenv("AFM_LITHO_HOSTNAME", "panel.example.org")
    # "No certificate" now means the default location is empty as well as the
    # environment being unset: tls_paths falls back to where make_local_cert
    # writes, so clearing the variables alone stopped meaning anything and
    # these tests started finding this machine's real certificate.
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "empty"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "empty"))
    yield


def _dns(monkeypatch, addrs):
    """Pin what getaddrinfo returns, so the test does not depend on whether
    the record has actually been created yet."""
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda host, *a, **k: [(None, None, None, "", (a_, 0)) for a_ in addrs])


def _certs(monkeypatch, tmp_path):
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    cert.write_text("cert"); key.write_text("key")
    monkeypatch.setenv("AFM_LITHO_TLS_CERT", str(cert))
    monkeypatch.setenv("AFM_LITHO_TLS_KEY", str(key))


# ------------------------------------------------------------------- dns --

def test_a_loopback_record_is_accepted(monkeypatch):
    _dns(monkeypatch, ["127.0.0.1"])
    assert site.resolves_to_loopback("panel.example.org")


def test_ipv6_loopback_counts(monkeypatch):
    _dns(monkeypatch, ["::1"])
    assert site.resolves_to_loopback("panel.example.org")


def test_a_routable_address_is_refused(monkeypatch):
    """Someone repointed the record at a real machine."""
    _dns(monkeypatch, ["10.0.4.22"])
    assert not site.resolves_to_loopback("panel.example.org")


def test_a_mixed_record_is_refused(monkeypatch):
    """The reason the check is `every` and not `any`: the browser would pick
    the routable one."""
    _dns(monkeypatch, ["127.0.0.1", "10.0.4.22"])
    assert not site.resolves_to_loopback("panel.example.org")


def test_a_missing_record_is_refused(monkeypatch):
    def boom(*a, **k):
        raise socket.gaierror("NXDOMAIN")
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    assert not site.resolves_to_loopback("panel.example.org")


def test_an_empty_hostname_is_refused():
    assert not site.resolves_to_loopback("")


# ------------------------------------------------------------- resolution --

def test_the_hostname_is_used_when_dns_and_tls_are_both_right(monkeypatch, tmp_path):
    _dns(monkeypatch, ["127.0.0.1"])
    _certs(monkeypatch, tmp_path)
    s = site.resolve_site(7461)
    assert s["url"] == "https://panel.example.org:7461/"
    assert s["origin"] == "https://panel.example.org:7461"
    assert s["tls"] and not s["reasons"]


def test_without_a_certificate_it_stays_on_the_loopback(monkeypatch):
    """A browser wants https for any name but localhost, so a hostname with
    no certificate is not usable for sign-in and is not worth the risk."""
    _dns(monkeypatch, ["127.0.0.1"])
    s = site.resolve_site(7461)
    assert s["url"] == "http://127.0.0.1:7461/"
    assert not s["tls"]
    assert any("certificate" in r for r in s["reasons"])


def test_bad_dns_falls_back_and_says_why(monkeypatch, tmp_path):
    _dns(monkeypatch, ["10.0.4.22"])
    _certs(monkeypatch, tmp_path)
    s = site.resolve_site(7461)
    assert s["url"] == "http://127.0.0.1:7461/"
    assert any("does not resolve" in r for r in s["reasons"])


def test_half_configured_tls_does_not_stop_it_starting(monkeypatch, tmp_path):
    """A cert path with no key, or a path to a file that is not there, drops
    back to http on loopback rather than refusing to run."""
    _dns(monkeypatch, ["127.0.0.1"])
    monkeypatch.setenv("AFM_LITHO_TLS_CERT", str(tmp_path / "missing.pem"))
    monkeypatch.setenv("AFM_LITHO_TLS_KEY", str(tmp_path / "missing.key"))
    s = site.resolve_site(7461)
    assert s["tls"] is False
    assert s["url"].startswith("http://127.0.0.1")


def test_the_hostname_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("AFM_LITHO_HOSTNAME", "")
    s = site.resolve_site(7461)
    assert s["url"] == "http://127.0.0.1:7461/"
    assert s["reasons"] == []               # unconfigured is not a failure


def test_there_is_no_default_hostname(monkeypatch):
    """Shipped blank: the panel stays on plain loopback, makes no DNS lookup
    and prints no reason, until AFM_LITHO_HOSTNAME is set."""
    monkeypatch.delenv("AFM_LITHO_HOSTNAME", raising=False)
    monkeypatch.setattr(site.socket, "getaddrinfo", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("no lookup without a hostname")))
    assert site.hostname() == "" and site.tls_possible() is False
    s = site.resolve_site(7461)
    assert s["url"] == "http://127.0.0.1:7461/" and s["tls"] is False
    assert s["reasons"] == []


def test_the_fallback_is_always_loopback_never_the_bare_hostname(monkeypatch):
    """The failure that would matter: falling back to http://panel.example.org
    would be both unusable for sign-in and, if the record ever moved, a
    connection to someone else's instrument."""
    for addrs in (["10.0.4.22"], ["8.8.8.8"], ["192.168.1.5"]):
        _dns(monkeypatch, addrs)
        s = site.resolve_site(7461)
        assert "panel.example.org" not in s["url"], addrs
        assert s["hostname"] == "127.0.0.1"


# ------------------------------------------------- the port in the URL -----
# User request: "do I need to write :7461 to access? why?"
# Because a browser assumes 443 for https, so https://panel.example.org means
# :443 and nothing was listening there. Serving on 443 is what removes it.

def test_serving_on_443_drops_the_port_from_the_url(monkeypatch, tmp_path):
    _dns(monkeypatch, ["127.0.0.1"])
    _certs(monkeypatch, tmp_path)
    s = site.resolve_site(443)
    assert s["url"] == "https://panel.example.org/"
    assert s["origin"] == "https://panel.example.org", (
        "the origin Google matches must not carry an implied port either")


def test_any_other_port_is_still_named(monkeypatch, tmp_path):
    _dns(monkeypatch, ["127.0.0.1"])
    _certs(monkeypatch, tmp_path)
    assert site.resolve_site(7461)["origin"] == "https://panel.example.org:7461"


def test_443_is_only_preferred_when_it_will_actually_be_used(monkeypatch):
    """Binding 443 for a plain-http loopback panel would take a port off the
    machine for no benefit: the URL carries a port either way."""
    monkeypatch.setenv("AFM_LITHO_PORT", "443")
    assert site.choose_port(tls=False, preferred=7461) >= 7461
    assert site.choose_port(tls=False, preferred=7461) != 443


def test_443_is_opt_in_so_nobody_has_to_edit_a_console(monkeypatch):
    """Moving to 443 changes the ORIGIN, and Google matches origins exactly,
    so it invalidates the registered one and creates a chore. The default is
    the port that already works."""
    monkeypatch.delenv("AFM_LITHO_PORT", raising=False)
    assert site.choose_port(tls=True, preferred=7461) != 443


def test_a_taken_443_falls_back_rather_than_refusing_to_start(monkeypatch):
    import socket
    monkeypatch.setenv("AFM_LITHO_PORT", "443")
    holder = socket.socket()
    try:
        holder.bind(("127.0.0.1", 443))
        holder.listen(1)
    except OSError:
        pytest.skip("cannot bind 443 here to simulate the conflict")
    try:
        p = site.choose_port(tls=True, preferred=7461)
        assert p != 443 and p >= 7461, "should have stepped aside, not failed"
    finally:
        holder.close()


def test_tls_possible_answers_without_being_told_a_port(monkeypatch, tmp_path):
    """The port choice depends on this, so it must not depend on the port."""
    _dns(monkeypatch, ["127.0.0.1"])
    _certs(monkeypatch, tmp_path)
    assert site.tls_possible() is True
    _dns(monkeypatch, ["10.0.4.22"])
    assert site.tls_possible() is False
