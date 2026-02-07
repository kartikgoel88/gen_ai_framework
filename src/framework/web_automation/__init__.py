"""Web automation: navigate to URL, fill fields, upload files, click."""

from .client import WebAutomationClient
from .types import (
    FillStep,
    UploadStep,
    ClickStep,
    WaitStep,
    AutomationStep,
    RunResult,
)

__all__ = [
    "WebAutomationClient",
    "FillStep",
    "UploadStep",
    "ClickStep",
    "WaitStep",
    "AutomationStep",
    "RunResult",
]
