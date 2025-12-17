"""Shared CLI utilities for AXIOM-CORE.

This module provides shared utilities, constants, and decorators
used across all CLI command modules.
"""


# Shared constants
TENANT_ID = "axiom-colony"


def print_header(title: str):
    """Print a consistent header for CLI output.

    Args:
        title: The header title
    """
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_section(title: str):
    """Print a section separator.

    Args:
        title: The section title
    """
    print(f"\n{title}:")


def print_receipt_note(receipt_name: str):
    """Print receipt emission note.

    Args:
        receipt_name: Name of the receipt emitted
    """
    print(f"\n[{receipt_name} receipt emitted above]")


def format_table_row(values: list, widths: list, separator: str = " | ") -> str:
    """Format a row for table output.

    Args:
        values: List of values to format
        widths: List of column widths
        separator: Column separator

    Returns:
        Formatted table row string
    """
    parts = []
    for v, w in zip(values, widths):
        if isinstance(v, float):
            parts.append(f"{v:>{w}.4f}")
        else:
            parts.append(f"{v:>{w}}")
    return separator.join(parts)


def print_result(label: str, value):
    """Print a labeled result value.

    Args:
        label: Result label
        value: Result value
    """
    print(f"  {label}: {value}")


def print_slo_check(check_name: str, passed: bool, value=None):
    """Print SLO check result.

    Args:
        check_name: Name of the SLO check
        passed: Whether the check passed
        value: Optional value to display
    """
    status = "PASS" if passed else "FAIL"
    if value is not None:
        print(f"  [{status}] {check_name} (got: {value})")
    else:
        print(f"  [{status}] {check_name}")
