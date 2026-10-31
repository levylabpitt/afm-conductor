"""Choosing the instrument plugin: one setting, "asylum" or "twin" (optionally "twin:mfp3d")."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from ..run.model import Platform, Part
from .interface import Instrument

PluginName = Literal["asylum", "twin"]
PLUGINS: tuple[PluginName, ...] = ("asylum", "twin")


@dataclass(frozen=True)
class PluginChoice(Part):
    """Which plugin runs the instrument, and for the twin which platform it plays.

    The twin plays the Cypher unless told otherwise (2026-10-26). The Asylum plugin
    takes the platform from the instrument it connects to, so a platform may not be chosen.
    """

    plugin: PluginName
    platform: Platform | None = None

    def __post_init__(self) -> None:
        if self.plugin == "twin" and self.platform is None:
            object.__setattr__(self, "platform", "cypher")
        super().__post_init__()

    def check(self) -> None:
        if self.plugin == "asylum" and self.platform is not None:
            raise ValueError("PluginChoice: the Asylum plugin reads the platform from the "
                             "instrument; do not choose one")

    @classmethod
    def from_setting(cls, setting: str) -> PluginChoice:
        """Read the setting text: "asylum", "twin", "twin:cypher" or "twin:mfp3d"."""
        plugin, _, platform = setting.strip().partition(":")
        return cls(plugin, platform or None)


PluginFactory = Callable[[PluginChoice], Instrument]


class PluginRegistry:
    """The plugins this program can run, by name. Opening one starts it."""

    def __init__(self) -> None:
        self._factories: dict[str, PluginFactory] = {}

    def register(self, name: PluginName, factory: PluginFactory) -> None:
        if name not in PLUGINS:
            raise ValueError(f"{name!r} is not a plugin name; expected one of {PLUGINS}")
        if name in self._factories:
            raise ValueError(f"plugin {name!r} is already registered")
        self._factories[name] = factory

    def open(self, choice: PluginChoice) -> Instrument:
        if choice.plugin not in self._factories:
            raise ValueError(f"plugin {choice.plugin!r} is not registered")
        return self._factories[choice.plugin](choice)
