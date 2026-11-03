"""Declared causes for conformance tests a suite host switches off."""

from __future__ import annotations

from enum import Enum


class Cause(Enum):
    """Why a host redefines, blanks or skips an inherited contract test."""

    NOT_IMPLEMENTED = "not_implemented"
    BACKEND = "backend"
    ENVIRONMENT = "environment"
    OTHER = "other"
