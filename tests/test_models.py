from format_guard.models import GuardMetrics, ValidationResult


def test_validation_result_defaults() -> None:
    result = ValidationResult(success=True)

    assert result.success is True
    assert result.value is None
    assert result.attempts == 0
    assert result.repaired is False
    assert result.error is None
    assert result.raw_output is None
    assert result.final_output is None


def test_validation_result_success() -> None:
    result = ValidationResult(
        success=True,
        value={"name": "Mainak"},
        attempts=1,
        repaired=False,
        raw_output='{"name": "Mainak"}',
        final_output='{"name": "Mainak"}',
    )

    assert result.success is True
    assert result.value == {"name": "Mainak"}
    assert result.attempts == 1
    assert result.repaired is False


def test_validation_result_failure() -> None:
    result = ValidationResult(
        success=False,
        attempts=4,
        repaired=True,
        error="Field required",
        raw_output='{"name": "Mainak"}',
        final_output='{"name": "Mainak"}',
    )

    assert result.success is False
    assert result.attempts == 4
    assert result.repaired is True
    assert result.error == "Field required"


def test_guard_metrics_defaults() -> None:
    metrics = GuardMetrics()

    assert metrics.attempts == 0
    assert metrics.repairs == 0
    assert metrics.validation_failures == 0
    assert metrics.successful is False
    assert metrics.repair_rate == 0.0


def test_guard_metrics_repair_rate() -> None:
    metrics = GuardMetrics(
        attempts=4,
        repairs=2,
        validation_failures=2,
        successful=True,
    )

    assert metrics.repair_rate == 0.5


def test_guard_metrics_zero_attempts() -> None:
    metrics = GuardMetrics(
        attempts=0,
        repairs=0,
    )

    assert metrics.repair_rate == 0.0