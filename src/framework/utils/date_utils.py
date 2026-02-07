"""Shared date parsing and formatting (e.g. passport, forms)."""

import re
from typing import Any

# Month name to number for parsing "Jan 15 1990" etc.
MONTH_NAMES = {
    "jan": "01", "january": "01", "feb": "02", "february": "02", "mar": "03", "march": "03",
    "apr": "04", "april": "04", "may": "05", "jun": "06", "june": "06", "jul": "07", "july": "07",
    "aug": "08", "august": "08", "sep": "09", "sept": "09", "september": "09", "oct": "10", "october": "10",
    "nov": "11", "november": "11", "dec": "12", "december": "12",
}


def normalize_date(value: Any) -> str | None:
    """
    Normalize any date string to YYYY-MM-DD; return None if unparseable.

    Handles: YYYY-MM-DD, DD/MM/YYYY, MRZ YYMMDD, month names (Jan 15 1990), 8-digit forms.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value).strip()
    else:
        value = value.strip()
    s = re.sub(r"\s+", " ", value).strip()
    if not s or s.lower() in ("null", "none", "n/a"):
        return None
    s = s.replace(",", " ").replace(".", "-")
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    m = re.match(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", s)
    if m:
        d, mo, y = m.group(1).zfill(2), m.group(2).zfill(2), m.group(3)
        return f"{y}-{mo}-{d}"
    m = re.match(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})$", s)
    if m:
        y, mo, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
        return f"{y}-{mo}-{d}"
    digits_only = re.sub(r"\D", "", s)
    if len(digits_only) == 6:
        m = re.match(r"^(\d{2})(\d{2})(\d{2})$", digits_only)
        if m:
            yy, mo, d = m.group(1), m.group(2), m.group(3)
            y = "19" + yy if int(yy) > 30 else "20" + yy
            return f"{y}-{mo}-{d}"
    m = re.match(r"(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})$", s)
    if m:
        d, mon, y = m.group(1).zfill(2), m.group(2).lower()[:3], m.group(3)
        mo = next((v for k, v in MONTH_NAMES.items() if k.startswith(mon) or mon in k), None)
        if mo:
            return f"{y}-{mo}-{d}"
    m = re.match(r"([a-zA-Z]+)\s+(\d{1,2})\s+(\d{4})$", s)
    if m:
        mon, d, y = m.group(1).lower()[:3], m.group(2).zfill(2), m.group(3)
        mo = next((v for k, v in MONTH_NAMES.items() if k.startswith(mon) or mon in k), None)
        if mo:
            return f"{y}-{mo}-{d}"
    m = re.match(r"(\d{4})\s+([a-zA-Z]+)\s+(\d{1,2})$", s)
    if m:
        y, mon, d = m.group(1), m.group(2).lower()[:3], m.group(3).zfill(2)
        mo = next((v for k, v in MONTH_NAMES.items() if k.startswith(mon) or mon in k), None)
        if mo:
            return f"{y}-{mo}-{d}"
    if len(digits_only) == 8:
        y1, mo1, d1 = digits_only[:4], digits_only[4:6], digits_only[6:8]
        if 1900 <= int(y1) <= 2100 and 1 <= int(mo1) <= 12 and 1 <= int(d1) <= 31:
            return f"{y1}-{mo1}-{d1}"
        d2, mo2, y2 = digits_only[:2], digits_only[2:4], digits_only[4:8]
        if 1900 <= int(y2) <= 2100 and 1 <= int(mo2) <= 12 and 1 <= int(d2) <= 31:
            return f"{y2}-{mo2}-{d2}"
    return None


def date_to_ddmmyyyy(s: str) -> str:
    """Convert YYYY-MM-DD to DD/MM/YYYY for forms that expect it."""
    if not s or len(s) != 10 or s[4] != "-" or s[7] != "-":
        return str(s)
    y, m, d = s[:4], s[5:7], s[8:10]
    return f"{d}/{m}/{y}"
