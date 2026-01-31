"""Rule-based validations for batch bills: month match, name fuzzy match, address match (cab)."""

import uuid
from datetime import datetime
from typing import Any, Optional

from rapidfuzz import fuzz

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

MANUAL_PREFIX = "MANUAL"
NAME_MATCH_THRESHOLD = 75
ADDRESS_MATCH_THRESHOLD = 40

# Supported date formats to try for month extraction
DATE_FORMATS = ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"]


def _parse_date_month(date_str: Optional[str]) -> Optional[int]:
    """Return month (1-12) from date string, or None if unparseable."""
    if not date_str or not str(date_str).strip():
        return None
    s = str(date_str).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).month
        except (ValueError, TypeError):
            continue
    return None


def _ensure_bill_id(bill: dict[str, Any], filename_key: str = "file_name") -> None:
    """Set bill id to MANUAL-{filename}-{uuid} if missing. Mutates bill."""
    if bill.get("id") is None or (isinstance(bill.get("id"), str) and not bill["id"].strip()):
        fname = bill.get(filename_key) or bill.get("filename") or "unknown"
        bill["id"] = f"{MANUAL_PREFIX}-{fname}-{uuid.uuid4()}"


def validate_ride(
    ride: dict[str, Any],
    client_addresses: dict[str, list[str]],
    emp_name: Optional[str] = None,
    emp_month: Optional[str] = None,
    name_threshold: int = NAME_MATCH_THRESHOLD,
    address_threshold: int = ADDRESS_MATCH_THRESHOLD,
) -> dict[str, Any]:
    """
    Validate a cab/ride bill: month match, name match (fuzzy), address match (fuzzy).
    Returns validation dict with month_match, name_match, name_match_score,
    address_match, address_match_score, is_valid.
    """
    validations: dict[str, Any] = {}

    _ensure_bill_id(ride, filename_key="filename" if "filename" in ride else "file_name")

    # 1. Month validation
    ride_month = _parse_date_month(ride.get("date"))
    expected_month = MONTH_MAP.get((emp_month or "").lower()) if emp_month else None
    validations["month_match"] = (
        ride_month is not None and expected_month is not None and ride_month == expected_month
    )
    if expected_month is None:
        validations["month_match"] = True  # No employee month → skip check

    # 2. Name validation
    rider = (ride.get("rider_name") or ride.get("vendor") or "").lower()
    emp = (emp_name or "").lower()
    name_score = fuzz.partial_ratio(rider, emp) if rider or emp else 100
    validations["name_match_score"] = name_score
    validations["name_match"] = name_score >= name_threshold
    if not emp:
        validations["name_match"] = True  # No employee name → skip check

    # 3. Address validation (cab only)
    pickup = (ride.get("pickup_address") or "").lower()
    drop = (ride.get("drop_address") or "").lower()
    client = (ride.get("client") or "").upper()
    addresses = client_addresses.get(client) or []
    best_address_score = 0
    for addr in addresses:
        a = (addr or "").lower()
        best_address_score = max(
            best_address_score,
            fuzz.partial_ratio(pickup, a),
            fuzz.partial_ratio(drop, a),
        )
    validations["address_match_score"] = best_address_score
    validations["address_match"] = (
        best_address_score >= address_threshold if addresses else True
    )

    validations["is_valid"] = (
        validations["month_match"]
        and validations["name_match"]
        and validations["address_match"]
    )
    return validations


def validate_meal(
    meal: dict[str, Any],
    emp_name: Optional[str] = None,
    emp_month: Optional[str] = None,
    name_threshold: int = NAME_MATCH_THRESHOLD,
) -> dict[str, Any]:
    """
    Validate a meal bill: month match, name match (fuzzy).
    Returns validation dict with month_match, name_match, name_match_score, is_valid.
    """
    validations: dict[str, Any] = {}

    _ensure_bill_id(meal, filename_key="filename" if "filename" in meal else "file_name")

    # 1. Month validation
    meal_month = _parse_date_month(meal.get("date"))
    expected_month = MONTH_MAP.get((emp_month or "").lower()) if emp_month else None
    validations["month_match"] = (
        meal_month is not None and expected_month is not None and meal_month == expected_month
    )
    if expected_month is None:
        validations["month_match"] = True

    # 2. Name validation
    buyer = (meal.get("buyer_name") or meal.get("vendor") or "").lower()
    emp = (emp_name or "").lower()
    name_score = fuzz.partial_ratio(buyer, emp) if buyer or emp else 100
    validations["name_match_score"] = name_score
    validations["name_match"] = name_score >= name_threshold
    if not emp:
        validations["name_match"] = True

    validations["address_match"] = True  # Not applicable for meals
    validations["is_valid"] = validations["month_match"] and validations["name_match"]
    return validations
