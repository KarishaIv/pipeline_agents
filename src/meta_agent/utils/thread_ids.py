"""Utilities for generating and managing thread IDs."""

import uuid


def generate_thread_id() -> str:
    """Generate a new thread ID using UUID v7.

    UUID v7 is time-based and sortable, making it suitable for
    generating unique thread identifiers.

    Returns:
        A new UUID v7 string.
    """
    return str(uuid.uuid7())
