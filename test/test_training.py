"""Tests for spaceproof.training module."""

import pytest
from spaceproof.training import (
    extract_training_example,
    emit_extraction_receipt,
    apply_labels,
    get_label_schema,
    emit_labeling_receipt,
    export_to_jsonl,
    export_to_huggingface,
    emit_export_receipt,
    score_example_quality,
    filter_by_quality,
    emit_quality_receipt,
    deduplicate_examples,
    compute_similarity,
    emit_dedup_receipt,
    add_to_retraining_queue,
    get_retraining_queue,
    prioritize_queue,
    emit_feedback_receipt,
)


def test_extract_training_example():
    """extract_training_example transforms intervention to example."""
    intervention = {
        "intervention_id": "int-001",
        "target_decision_id": "dec-001",
        "reason_code": "RE001_FACTUAL_ERROR",
        "original_action": {"type": "wrong"},
        "corrected_action": {"type": "correct"},
        "justification": "Test justification",
    }
    example = extract_training_example(intervention)
    assert example["source_intervention_id"] == "int-001"
    assert "example_id" in example
    assert "timestamp" in example


def test_emit_extraction_receipt(capsys):
    """emit_extraction_receipt emits valid receipt."""
    receipt = emit_extraction_receipt(
        example_id="ex-001",
        intervention_id="int-001",
    )
    assert receipt["receipt_type"] == "training_extraction"
    assert receipt["example_id"] == "ex-001"


def test_apply_labels():
    """apply_labels adds structured labels to example."""
    example = {"example_id": "ex-001", "reason_code": "RE001_FACTUAL_ERROR"}
    labeled = apply_labels(example)
    assert "labels" in labeled
    assert isinstance(labeled["labels"], dict)


def test_get_label_schema():
    """get_label_schema returns valid schema dict."""
    schema = get_label_schema()
    assert isinstance(schema, dict)
    assert "reason_codes" in schema or len(schema) > 0


def test_emit_labeling_receipt(capsys):
    """emit_labeling_receipt emits valid receipt."""
    receipt = emit_labeling_receipt(
        example_id="ex-001",
        labels={"category": "error", "severity": "high"},
    )
    assert receipt["receipt_type"] == "training_labeling"


def test_export_to_jsonl():
    """export_to_jsonl exports examples to JSONL format."""
    examples = [
        {"example_id": "ex-001", "data": "test1"},
        {"example_id": "ex-002", "data": "test2"},
    ]
    result = export_to_jsonl(examples)
    assert result["success"] is True
    assert result["format"] == "jsonl"
    assert result["count"] == 2


def test_export_to_huggingface():
    """export_to_huggingface prepares HuggingFace format."""
    examples = [{"example_id": "ex-001", "data": "test"}]
    result = export_to_huggingface(examples, dataset_name="test-dataset")
    assert result["success"] is True
    assert result["format"] == "huggingface"


def test_emit_export_receipt(capsys):
    """emit_export_receipt emits valid receipt."""
    receipt = emit_export_receipt(
        format="jsonl",
        count=10,
        destination="local",
    )
    assert receipt["receipt_type"] == "training_export"
    assert receipt["count"] == 10


def test_score_example_quality():
    """score_example_quality returns score between 0 and 1."""
    example = {
        "example_id": "ex-001",
        "justification": "Clear explanation",
        "original_action": {"type": "wrong"},
        "corrected_action": {"type": "correct"},
    }
    score = score_example_quality(example)
    assert 0 <= score <= 1


def test_filter_by_quality():
    """filter_by_quality removes low-quality examples."""
    examples = [
        {"example_id": "ex-001", "quality_score": 0.9},
        {"example_id": "ex-002", "quality_score": 0.3},
        {"example_id": "ex-003", "quality_score": 0.85},
    ]
    filtered = filter_by_quality(examples, threshold=0.8)
    assert len(filtered) == 2
    assert all(e["quality_score"] >= 0.8 for e in filtered)


def test_emit_quality_receipt(capsys):
    """emit_quality_receipt emits valid receipt."""
    receipt = emit_quality_receipt(
        example_id="ex-001",
        quality_score=0.92,
    )
    assert receipt["receipt_type"] == "training_quality"
    assert receipt["quality_score"] == 0.92


def test_deduplicate_examples():
    """deduplicate_examples removes duplicates."""
    examples = [
        {"example_id": "ex-001", "source_intervention_id": "int-001"},
        {"example_id": "ex-002", "source_intervention_id": "int-001"},  # Duplicate
        {"example_id": "ex-003", "source_intervention_id": "int-002"},
    ]
    unique = deduplicate_examples(examples)
    assert len(unique) == 2


def test_compute_similarity():
    """compute_similarity returns similarity score."""
    ex1 = {"data": "hello world"}
    ex2 = {"data": "hello world"}
    ex3 = {"data": "completely different"}

    sim_same = compute_similarity(ex1, ex2)
    sim_diff = compute_similarity(ex1, ex3)

    assert sim_same >= sim_diff


def test_emit_dedup_receipt(capsys):
    """emit_dedup_receipt emits valid receipt."""
    receipt = emit_dedup_receipt(
        original_count=10,
        deduplicated_count=8,
        removed_count=2,
    )
    assert receipt["receipt_type"] == "training_dedup"
    assert receipt["removed_count"] == 2


def test_add_to_retraining_queue():
    """add_to_retraining_queue adds example to queue."""
    example = {"example_id": "ex-001", "severity": "CRITICAL"}
    result = add_to_retraining_queue(example, priority="IMMEDIATE")
    assert result["queued"] is True
    assert result["priority"] == "IMMEDIATE"


def test_get_retraining_queue():
    """get_retraining_queue returns queue items."""
    queue = get_retraining_queue()
    assert isinstance(queue, list)


def test_prioritize_queue():
    """prioritize_queue orders by priority."""
    queue = [
        {"example_id": "ex-001", "priority": "MEDIUM"},
        {"example_id": "ex-002", "priority": "IMMEDIATE"},
        {"example_id": "ex-003", "priority": "LOW"},
    ]
    prioritized = prioritize_queue(queue)
    # IMMEDIATE should be first
    assert prioritized[0]["priority"] == "IMMEDIATE"


def test_emit_feedback_receipt(capsys):
    """emit_feedback_receipt emits valid receipt."""
    receipt = emit_feedback_receipt(
        queue_size=5,
        immediate_count=2,
    )
    assert receipt["receipt_type"] == "training_feedback"
