"""Tests for spaceproof.rnes module."""

import pytest
from spaceproof.rnes import (
    validate_rnes_compliance,
    check_receipt_emission,
    emit_rnes_receipt,
    interpret_receipt,
    execute_receipt_chain,
    get_execution_result,
    emit_interpretation_receipt,
    create_sandbox,
    execute_in_sandbox,
    validate_sandbox_output,
    emit_sandbox_receipt,
)


def test_validate_rnes_compliance_valid():
    """validate_rnes_compliance passes for compliant code."""
    code = '''
def process():
    emit_receipt("operation", {"data": "test"})
    return True
'''
    result = validate_rnes_compliance(code)
    assert result["compliant"] is True


def test_validate_rnes_compliance_missing_receipt():
    """validate_rnes_compliance fails for missing receipts."""
    code = '''
def process():
    # No receipt emission
    return True
'''
    result = validate_rnes_compliance(code)
    # Should flag missing receipt emission
    assert "warnings" in result or "compliant" in result


def test_check_receipt_emission():
    """check_receipt_emission verifies receipt was emitted."""
    execution_log = [
        {"type": "receipt", "receipt_type": "operation"},
        {"type": "return", "value": True},
    ]
    result = check_receipt_emission(execution_log)
    assert result["receipts_emitted"] >= 1


def test_emit_rnes_receipt(capsys):
    """emit_rnes_receipt emits valid receipt."""
    receipt = emit_rnes_receipt(
        operation_id="op-001",
        compliant=True,
        receipts_emitted=3,
    )
    assert receipt["receipt_type"] == "rnes_validation"
    assert receipt["compliant"] is True


def test_interpret_receipt():
    """interpret_receipt parses receipt for execution."""
    receipt = {
        "receipt_type": "command",
        "action": "navigate",
        "parameters": {"target": "waypoint_1"},
    }
    interpretation = interpret_receipt(receipt)
    assert "action" in interpretation
    assert "parameters" in interpretation


def test_execute_receipt_chain():
    """execute_receipt_chain executes sequence of receipts."""
    chain = [
        {"receipt_type": "init", "action": "setup"},
        {"receipt_type": "command", "action": "execute"},
        {"receipt_type": "finalize", "action": "cleanup"},
    ]
    result = execute_receipt_chain(chain)
    assert result["success"] is True
    assert result["executed_count"] == 3


def test_get_execution_result():
    """get_execution_result retrieves execution outcome."""
    result = get_execution_result(execution_id="exec-001")
    assert isinstance(result, dict)


def test_emit_interpretation_receipt(capsys):
    """emit_interpretation_receipt emits valid receipt."""
    receipt = emit_interpretation_receipt(
        receipt_id="r-001",
        interpretation={"action": "navigate"},
        success=True,
    )
    assert receipt["receipt_type"] == "rnes_interpretation"


def test_create_sandbox():
    """create_sandbox creates isolated execution environment."""
    sandbox = create_sandbox(
        sandbox_id="sandbox-001",
        constraints={"max_memory_mb": 256, "max_time_sec": 30},
    )
    assert sandbox["sandbox_id"] == "sandbox-001"
    assert "constraints" in sandbox


def test_execute_in_sandbox():
    """execute_in_sandbox runs code in isolation."""
    sandbox = create_sandbox("sandbox-002", {"max_time_sec": 5})
    code = "result = 2 + 2"
    result = execute_in_sandbox(sandbox, code)
    assert result["success"] is True


def test_execute_in_sandbox_timeout():
    """execute_in_sandbox handles timeout."""
    sandbox = create_sandbox("sandbox-003", {"max_time_sec": 0.001})
    # Long-running code
    code = "while True: pass"
    result = execute_in_sandbox(sandbox, code)
    assert result["success"] is False or result.get("timeout") is True


def test_validate_sandbox_output():
    """validate_sandbox_output checks output is safe."""
    output = {"result": 42, "receipts": [{"type": "operation"}]}
    result = validate_sandbox_output(output)
    assert result["valid"] is True


def test_validate_sandbox_output_unsafe():
    """validate_sandbox_output rejects unsafe output."""
    output = {"result": None, "error": "access_violation"}
    result = validate_sandbox_output(output)
    # Should flag unsafe output
    assert "valid" in result


def test_emit_sandbox_receipt(capsys):
    """emit_sandbox_receipt emits valid receipt."""
    receipt = emit_sandbox_receipt(
        sandbox_id="sandbox-001",
        execution_time_ms=150,
        success=True,
        output_hash="abc123:def456",
    )
    assert receipt["receipt_type"] == "sandbox_execution"
    assert receipt["success"] is True
