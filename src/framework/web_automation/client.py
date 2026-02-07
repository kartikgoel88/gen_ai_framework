"""Playwright-based web automation client: navigate, fill, upload, click."""

import asyncio
from pathlib import Path
from typing import Sequence

from playwright.async_api import async_playwright, Browser, Page, TimeoutError as PlaywrightTimeout

from .types import (
    AutomationStep,
    FillStep,
    UploadStep,
    ClickStep,
    WaitStep,
    RunResult,
)


class WebAutomationClient:
    """Headless browser automation via Playwright. Supports fill, file upload, click, wait."""

    def __init__(self, headless: bool = True, timeout_ms: int = 30_000):
        self.headless = headless
        self.timeout_ms = timeout_ms

    async def run(
        self,
        url: str,
        steps: Sequence[AutomationStep],
    ) -> RunResult:
        """
        Navigate to url and run the given steps in order.
        Steps can be fill, upload (requires file_path set on step), click, or wait.
        """
        async with async_playwright() as p:
            browser: Browser = await p.chromium.launch(headless=self.headless)
            try:
                page = await browser.new_page()
                page.set_default_timeout(self.timeout_ms)
                await page.goto(url, wait_until="domcontentloaded")
                final_url = page.url
                completed = 0
                for i, step in enumerate(steps):
                    try:
                        if step.action == "fill":
                            await self._do_fill(page, step)
                        elif step.action == "upload":
                            await self._do_upload(page, step)
                        elif step.action == "click":
                            await self._do_click(page, step)
                        elif step.action == "wait":
                            await self._do_wait(page, step)
                        else:
                            raise ValueError(f"Unknown step action: {getattr(step, 'action', step)}")
                        completed += 1
                    except Exception as e:
                        await browser.close()
                        return RunResult(
                            success=False,
                            final_url=final_url,
                            error=f"Step {i + 1} ({getattr(step, 'action', '?')}): {e!s}",
                            steps_completed=completed,
                        )
                final_url = page.url
                return RunResult(
                    success=True,
                    final_url=final_url,
                    steps_completed=completed,
                )
            finally:
                await browser.close()

    async def _do_fill(self, page: Page, step: FillStep) -> None:
        await page.locator(step.selector).fill(step.value)

    async def _do_upload(self, page: Page, step: UploadStep) -> None:
        if not step.file_path or not Path(step.file_path).exists():
            raise FileNotFoundError(f"Upload step requires a valid file_path: {getattr(step, 'file_path', None)}")
        await page.locator(step.selector).set_input_files(step.file_path)

    async def _do_click(self, page: Page, step: ClickStep) -> None:
        await page.locator(step.selector).click()

    async def _do_wait(self, page: Page, step: WaitStep) -> None:
        if step.selector:
            await page.wait_for_selector(step.selector, state="visible", timeout=self.timeout_ms)
        elif step.timeout_ms is not None:
            await asyncio.sleep(step.timeout_ms / 1000.0)
        else:
            raise ValueError("Wait step must have selector or timeout_ms")
