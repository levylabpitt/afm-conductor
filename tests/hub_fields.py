"""Where a hub field lives: on `hub.host` (the instrument's state) or on the hub itself."""

from afm_litho.service.host_state import HOST_FIELD_NAMES


def owner(hub, name: str):
    """The object that holds `name`, for loops over mixed field names.

    `fault` stays on the hub: its property announces each change to the pages."""
    return hub.host if name in HOST_FIELD_NAMES and name != "fault" else hub
