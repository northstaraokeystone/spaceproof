"""Training pipeline - Transform interventions to training data.

Every human correction is training data, not just audit event.
Extracts, labels, scores quality, deduplicates, and exports examples.
"""

from .extractor import (
    extract_training_example,
    gather_input_context,
    format_bad_output,
    format_good_output,
    emit_training_example_receipt,
    TrainingExample,
)

from .labeler import (
    apply_label,
    compute_label_distribution,
    get_labels_for_reason_code,
    LabeledExample,
)

from .exporter import (
    export_to_jsonl,
    export_to_huggingface,
    get_export_stats,
    ExportResult,
)

from .quality import (
    score_example_quality,
    filter_by_quality,
    compute_quality_distribution,
    QualityScore,
)

from .dedup import (
    deduplicate_examples,
    compute_similarity,
    find_duplicates,
    DedupResult,
)

from .feedback_loop import (
    add_to_retraining_queue,
    get_retraining_queue,
    process_retraining_batch,
    validate_learning,
    FeedbackLoopState,
)

__all__ = [
    # Extractor
    "extract_training_example",
    "gather_input_context",
    "format_bad_output",
    "format_good_output",
    "emit_training_example_receipt",
    "TrainingExample",
    # Labeler
    "apply_label",
    "compute_label_distribution",
    "get_labels_for_reason_code",
    "LabeledExample",
    # Exporter
    "export_to_jsonl",
    "export_to_huggingface",
    "get_export_stats",
    "ExportResult",
    # Quality
    "score_example_quality",
    "filter_by_quality",
    "compute_quality_distribution",
    "QualityScore",
    # Dedup
    "deduplicate_examples",
    "compute_similarity",
    "find_duplicates",
    "DedupResult",
    # Feedback loop
    "add_to_retraining_queue",
    "get_retraining_queue",
    "process_retraining_batch",
    "validate_learning",
    "FeedbackLoopState",
]
