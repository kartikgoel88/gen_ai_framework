"""Schemas for web automation steps and results."""

from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field


class FillStep(BaseModel):
    """Fill a form field (input, textarea) by selector."""

    action: Literal["fill"] = "fill"
    selector: str = Field(..., description="CSS selector or Playwright locator (e.g. input#email)")
    value: str = Field(..., description="Value to fill")


class UploadStep(BaseModel):
    """Set file(s) on a file input. Use file_index to reference multipart file (0-based)."""

    action: Literal["upload"] = "upload"
    selector: str = Field(..., description="CSS selector for the file input (e.g. input[type=file])")
    file_index: int = Field(0, description="Index of the uploaded file in the request (0-based)")
    file_path: str | None = Field(
        None,
        description="Path on server (injected when processing multipart; do not send in request)",
    )


class ClickStep(BaseModel):
    """Click an element (e.g. submit button)."""

    action: Literal["click"] = "click"
    selector: str = Field(..., description="CSS selector for the element to click")


class WaitStep(BaseModel):
    """Wait for a selector to appear or for a duration (ms)."""

    action: Literal["wait"] = "wait"
    selector: str | None = Field(None, description="Wait until this selector is visible")
    timeout_ms: int | None = Field(None, description="Or wait this many milliseconds")


AutomationStep = Annotated[
    Union[FillStep, UploadStep, ClickStep, WaitStep],
    Field(discriminator="action"),
]


class RunResult(BaseModel):
    """Result of running a web automation session."""

    success: bool = True
    final_url: str | None = None
    error: str | None = None
    steps_completed: int = 0
