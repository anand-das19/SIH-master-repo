"""
AgriNode AI - Phase 6 Ground-Truth Benchmarking Tests
tests/test_phase6.py

Verifies:
  1. Monotonicity across drought, flood, and heat engines
  2. Full benchmark execution via run_benchmark()
  3. SLA compliance:
     - Disaster Recall >= 85%
     - Control Specificity >= 85%
     - False Alarm Rate <= 15%
     - Monotonicity = 100%
     - Mean Inference Latency < 50 ms
  4. Markdown report generation and non-empty content
  5. Terminal report formatter validity
"""

import sys, os
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.benchmark import (
    run_benchmark,
    evaluate_monotonicity,
    format_benchmark_terminal,
    export_markdown_report,
    OUTPUT_REPORT_MD
)


def test_monotonicity():
    """Score monotonicity must pass for all three hazard engines."""
    mono = evaluate_monotonicity()
    assert mono["drought"], "Drought score monotonicity failed"
    assert mono["flood"], "Flood score monotonicity failed"
    assert mono["heat"], "Heat score monotonicity failed"
    assert mono["all_passed"], "Overall monotonicity test failed"


def test_benchmark_sla_compliance():
    """Evaluate benchmark suite and assert all SLA thresholds are strictly met."""
    bench = run_benchmark()

    assert bench["total_cases"] == 20
    assert bench["disaster_total"] == 16
    assert bench["control_total"] == 4

    sla = bench["sla_results"]

    # 1. Disaster Recall >= 85%
    assert sla["disaster_recall"]["passed"], f"Disaster recall {sla['disaster_recall']['value']}% < 85%"
    assert sla["disaster_recall"]["value"] >= 85.0

    # 2. Control Specificity >= 85%
    assert sla["control_specificity"]["passed"], f"Control specificity {sla['control_specificity']['value']}% < 85%"
    assert sla["control_specificity"]["value"] >= 85.0

    # 3. False Alarm Rate <= 15%
    assert sla["false_alarm_rate"]["passed"], f"False alarm rate {sla['false_alarm_rate']['value']}% > 15%"
    assert sla["false_alarm_rate"]["value"] <= 15.0

    # 4. Score Monotonicity 100%
    assert sla["score_monotonicity"]["passed"]

    # 5. Average Pipeline Latency < 50ms
    assert sla["avg_latency_ms"]["passed"], f"Avg latency {sla['avg_latency_ms']['value']} ms >= 50ms"

    # Overall SLA status
    assert bench["all_sla_passed"], "Not all SLAs passed"
    assert bench["overall_accuracy"] >= 95.0


def test_terminal_report_rendering():
    """Terminal report must render valid ASCII string without errors."""
    bench = run_benchmark()
    text = format_benchmark_terminal(bench)
    assert isinstance(text, str)
    assert len(text) > 500
    assert "FINAL VERDICT: ALL SLA TARGETS EXCEEDED" in text
    text.encode("ascii")


def test_markdown_report_export(tmp_path):
    """Markdown benchmark report must export properly with tables and headers."""
    bench = run_benchmark()
    test_file = tmp_path / "test_benchmark.md"
    export_markdown_report(bench, output_file=test_file)

    assert test_file.exists()
    content = test_file.read_text(encoding="utf-8")
    assert "# AgriNode AI — Phase 6 Ground-Truth Benchmark Report" in content
    assert "Disaster Recall" in content
    assert "Control Specificity" in content
    assert "CASE_DR_01" in content
    assert "CASE_CTRL_01" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
