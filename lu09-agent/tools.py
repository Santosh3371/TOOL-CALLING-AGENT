"""Read-only assignment lookup, workload calculation, and study-plan drafting tools."""

import csv
import math
from datetime import date
from pathlib import Path

from langchain_core.tools import tool

ASSIGNMENTS_FILE = Path(__file__).parent / "data" / "assignments.csv"
MAX_RESULT_CHARS = 700


class AssignmentDataError(RuntimeError):
    """Raised when the local assignment CSV cannot be read safely."""


def clip(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    """Keep tool output short enough for the model and the visible trace."""
    text = str(text)
    return text if len(text) <= limit else text[:limit] + " ...(cut)"


def read_assignments() -> list[dict[str, str]]:
    """Read assignment rows without changing the CSV."""
    try:
        with ASSIGNMENTS_FILE.open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))
    except (OSError, UnicodeError, csv.Error) as error:
        raise AssignmentDataError(
            "The local assignment data could not be read."
        ) from error


def _find_assignment(name: str) -> dict[str, str] | None:
    target = name.strip().casefold()
    if not target:
        return None
    for row in read_assignments():
        if (row.get("assignment_name") or "").strip().casefold() == target:
            return row
    return None


def _assignment_summary(row: dict[str, str]) -> str:
    return (
        f"{row.get('assignment_name', 'Unnamed assignment')} "
        f"({row.get('subject', 'Unknown subject')}) — due "
        f"{row.get('due_date', 'date unavailable')}; "
        f"{row.get('estimated_hours', 'hours unavailable')} hours; "
        f"status: {row.get('status', 'unknown')}."
    )


@tool
def find_assignments(subject: str) -> str:
    """Find up to five assignments for a subject, or use 'due soon', 'overdue', or 'all' for deadline views."""
    query = subject.strip().casefold()
    if not query:
        return "Enter a subject, or use 'due soon', 'overdue', or 'all'."

    rows = read_assignments()
    if query in {"all", "everything"}:
        matches = rows
    elif query in {"due soon", "upcoming", "due this week", "deadlines"}:
        today = date.today()
        matches = [
            row
            for row in rows
            if _valid_due_date(row) is not None
            and 0 <= (_valid_due_date(row) - today).days <= 7
        ]
        matches.sort(key=lambda row: row["due_date"])
    elif query == "overdue":
        today = date.today()
        matches = [
            row
            for row in rows
            if _valid_due_date(row) is not None
            and (_valid_due_date(row) - today).days < 0
        ]
        matches.sort(key=lambda row: row["due_date"])
    else:
        matches = [
            row
            for row in rows
            if query in row.get("subject", "").casefold()
        ]

    if not matches:
        return f"No assignments found for '{subject.strip()}'."
    shown = matches[:5]
    lines = [_assignment_summary(row) for row in shown]
    if len(matches) > len(shown):
        lines.append(f"...and {len(matches) - len(shown)} more.")
    return clip("\n".join(lines))


def _valid_due_date(row: dict[str, str]) -> date | None:
    try:
        return date.fromisoformat(str(row.get("due_date") or ""))
    except ValueError:
        return None


def _valid_hours(row: dict[str, str]) -> float | None:
    try:
        hours = float(row.get("estimated_hours") or "")
    except (TypeError, ValueError):
        return None
    return hours if math.isfinite(hours) and hours >= 0 else None


@tool
def calculate_study_workload(assignment_name: str) -> str:
    """Calculate the days remaining and estimated study hours for one assignment."""
    if not assignment_name.strip():
        return "Enter an assignment name to calculate its workload."
    row = _find_assignment(assignment_name)
    if row is None:
        return f"No assignment found named '{assignment_name.strip()}'."

    due_date = _valid_due_date(row)
    hours = _valid_hours(row)
    if due_date is None or hours is None:
        return "This assignment has invalid deadline or estimated-hours data."

    days_remaining = (due_date - date.today()).days
    if days_remaining < 0:
        deadline = f"{abs(days_remaining)} day(s) overdue"
    elif days_remaining == 0:
        deadline = "due today"
    else:
        deadline = f"{days_remaining} day(s) remaining"
    return clip(
        f"{row['assignment_name']}: {deadline}; "
        f"estimated study time is {hours:g} hour(s)."
    )


@tool
def draft_study_plan(assignment_name: str) -> str:
    """Draft a short study plan for an assignment without scheduling or sending anything."""
    if not assignment_name.strip():
        return "Enter an assignment name to draft its study plan."
    row = _find_assignment(assignment_name)
    if row is None:
        return f"No assignment found named '{assignment_name.strip()}'."

    due_date = _valid_due_date(row)
    hours = _valid_hours(row)
    if due_date is None or hours is None or hours <= 0:
        return "This assignment has invalid deadline or estimated-hours data."

    days_remaining = max(1, (due_date - date.today()).days)
    sessions = min(max(1, days_remaining), max(1, round(hours / 0.75)))
    hours_per_session = hours / sessions
    plan = (
        f"Draft plan for {row['assignment_name']} ({row.get('subject', 'subject')}): "
        f"complete about {hours_per_session:.1f} study hour(s) in each of "
        f"{sessions} session(s) before {due_date.isoformat()}. Start with the "
        "most challenging part, then review your work and leave time for a final check."
    )
    return clip(plan)


TOOLS = [find_assignments, calculate_study_workload, draft_study_plan]
