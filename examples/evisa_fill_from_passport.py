"""
Fill Vietnam e-visa form (https://evisa.gov.vn/e-visa/foreigners) using passport details.

Uses framework:
- Passport parsing: PassportEye (+ optional LLM) to get passport_number, name, date_of_birth, date_of_issue, date_of_expiry.
- Browser: Playwright with multiple locator strategies (placeholder, label, CSS) so filling works even when the site changes.

Usage (from project root):
  python -m examples.evisa_fill_from_passport --passport-image data/images/Passport-page-001.jpg
  python -m examples.evisa_fill_from_passport --passport-image path/to/passport.jpg --no-headless
  python -m examples.evisa_fill_from_passport --passport-image path/to/passport.jpg --extract   # use LLM to fill gaps

Note: Passport image should show the MRZ (bottom lines) for best results.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.framework.documents.types import ExtractResult


def _date_to_ddmmyyyy(s: str) -> str:
    """Convert YYYY-MM-DD to DD/MM/YYYY for forms that expect it."""
    if not s or len(s) != 10 or s[4] != "-" or s[7] != "-":
        return str(s)
    y, m, d = s[:4], s[5:7], s[8:10]
    return f"{d}/{m}/{y}"


# Per-field: list of (strategy, value). strategy = "placeholder" | "label" | "css"; value = placeholder text, label text, or CSS selector.
# We try each until one locator finds an element and fill succeeds.
EVISA_FIELDS = {
    "passport_number": (str, [
        ("placeholder", "Passport number"),
        ("placeholder", "passport"),
        ("label", "Passport number"),
        ("label", "Passport No"),
        ("css", "input[name='passportNumber']"),
        ("css", "input[id*='passport']"),
        ("css", "input[placeholder*='assport']"),
    ]),
    "name": (str, [
        ("placeholder", "Full name"),
        ("placeholder", "Full Name"),
        ("label", "Full name"),
        ("label", "Name"),
        ("css", "input[name='fullName']"),
        ("css", "input[id*='name']"),
        ("css", "input[placeholder*='ame']"),
    ]),
    "date_of_birth": (_date_to_ddmmyyyy, [
        ("placeholder", "Date of birth"),
        ("placeholder", "DOB"),
        ("label", "Date of birth"),
        ("label", "Birth"),
        ("css", "input[name='dateOfBirth']"),
        ("css", "input[id*='birth']"),
        ("css", "input[placeholder*='irth']"),
    ]),
    "date_of_issue": (_date_to_ddmmyyyy, [
        ("placeholder", "Date of issue"),
        ("placeholder", "Issue date"),
        ("label", "Issue date"),
        ("css", "input[name='issueDate']"),
        ("css", "input[id*='issue']"),
        ("css", "input[placeholder*='ssue']"),
    ]),
    "date_of_expiry": (_date_to_ddmmyyyy, [
        ("placeholder", "Date of expiry"),
        ("placeholder", "Expiry"),
        ("label", "Expiry date"),
        ("label", "Valid until"),
        ("css", "input[name='expiryDate']"),
        ("css", "input[id*='expir']"),
        ("css", "input[placeholder*='xpir']"),
    ]),
}

EVISA_URL = "https://evisa.gov.vn/e-visa/foreigners"


def _get_entities_from_passport_image(
    image_path: Path,
    use_llm: bool = False,
) -> dict:
    """Get passport entities from one image using PassportEye; optionally use LLM to fill gaps."""
    from src.clients.documents.parse_passport import (
        PASSPORT_ENTITIES,
        _entities_from_mrz,
        extract_passport_entities,
        _processor_for_method,
    )

    processor = _processor_for_method("passporteye")
    result: ExtractResult = processor.extract(image_path)
    if result.error:
        return {k: None for k in PASSPORT_ENTITIES}

    mrz = (result.metadata or {}).get("mrz")
    if mrz and isinstance(mrz, dict):
        entities = _entities_from_mrz(result.metadata)
        if use_llm and result.text and any(entities.get(k) is None for k in PASSPORT_ENTITIES):
            from src.framework.config import get_settings
            from src.framework.api.deps_llm import get_llm
            llm = get_llm(get_settings())
            llm_entities = extract_passport_entities(result.text, llm)
            for k in PASSPORT_ENTITIES:
                if entities.get(k) is None and llm_entities.get(k) is not None:
                    entities[k] = llm_entities[k]
        return entities

    if use_llm and result.text:
        from src.framework.config import get_settings
        from src.framework.api.deps_llm import get_llm
        return extract_passport_entities(result.text, get_llm(get_settings()))

    return {k: None for k in PASSPORT_ENTITIES}


async def _fill_with_playwright(entities: dict, headless: bool, timeout_ms: int = 15000) -> None:
    """Open e-visa URL and fill fields using Playwright, trying multiple locators per field."""
    from playwright.async_api import async_playwright

    filled = 0
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        try:
            page = await browser.new_page()
            page.set_default_timeout(timeout_ms)
            await page.goto(EVISA_URL, wait_until="domcontentloaded")
            try:
                await page.wait_for_selector("input, [role='textbox']", timeout=timeout_ms)
            except Exception:
                pass

            for key, (formatter, strategies) in EVISA_FIELDS.items():
                value = entities.get(key)
                if value is None or value == "":
                    continue
                value_str = formatter(value) if formatter else str(value)
                for strategy, locator_value in strategies:
                    try:
                        if strategy == "placeholder":
                            loc = page.get_by_placeholder(locator_value, exact=False)
                        elif strategy == "label":
                            loc = page.get_by_label(locator_value, exact=False)
                        else:
                            loc = page.locator(locator_value)
                        await loc.first.fill(value_str)
                        filled += 1
                        print(f"  Filled {key} (via {strategy})", file=sys.stderr)
                        break
                    except Exception:
                        continue
                else:
                    print(f"  Skipped {key}: no matching field found", file=sys.stderr)

            print(f"Done. Filled {filled} field(s).", file=sys.stderr)
        finally:
            await browser.close()


async def run_evisa_fill(
    passport_image: Path,
    headless: bool = True,
    use_llm: bool = False,
) -> None:
    """Parse passport image, then open e-visa URL and fill form with extracted details."""
    print("Parsing passport image...", file=sys.stderr)
    entities = _get_entities_from_passport_image(passport_image, use_llm=use_llm)
    print("Extracted:", {k: v for k, v in entities.items() if v}, file=sys.stderr)

    if not any(entities.get(k) for k in EVISA_FIELDS):
        print("No fields to fill. Check passport image or use --extract for LLM.", file=sys.stderr)
        return

    print(f"Opening {EVISA_URL} and filling form...", file=sys.stderr)
    await _fill_with_playwright(entities, headless=headless)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill Vietnam e-visa form from passport image.")
    parser.add_argument("--passport-image", type=Path, required=True, help="Path to passport image (front/MRZ).")
    parser.add_argument("--no-headless", action="store_true", help="Show browser window.")
    parser.add_argument("--extract", action="store_true", help="Use LLM to fill missing fields from OCR text.")
    args = parser.parse_args()

    if not args.passport_image.exists():
        print(f"Error: file not found: {args.passport_image}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run_evisa_fill(
        args.passport_image,
        headless=not args.no_headless,
        use_llm=args.extract,
    ))


if __name__ == "__main__":
    main()
