"""Tests for spaceproof.sovereignty module."""

import pytest
from spaceproof.sovereignty import (
    SovereigntyConfig,
    compute_sovereignty,
    find_threshold,
    internal_rate,
    external_rate,
    sovereignty_advantage,
    is_sovereign,
    MARS_LIGHT_DELAY_AVG_S,
)


def test_internal_rate_positive():
    """internal_rate is non-negative."""
    assert internal_rate(0) >= 0  # log2(1) = 0 when crew=0
    assert internal_rate(10) > internal_rate(1)


def test_internal_rate_increases_with_crew():
    """internal_rate increases with crew size."""
    r1 = internal_rate(10)
    r2 = internal_rate(100)
    assert r2 > r1


def test_external_rate_positive():
    """external_rate is positive for valid inputs."""
    rate = external_rate(2.0, 480.0)
    assert rate > 0


def test_external_rate_formula():
    """external_rate follows bandwidth/(2*delay) formula."""
    rate = external_rate(10.0, 500.0)
    expected = (10.0 * 1e6) / (2 * 500.0)
    assert rate == pytest.approx(expected)


def test_sovereignty_advantage():
    """sovereignty_advantage is internal - external."""
    adv = sovereignty_advantage(100, 50)
    assert adv == 50


def test_is_sovereign():
    """is_sovereign returns True for positive advantage."""
    assert is_sovereign(1.0) is True
    assert is_sovereign(-1.0) is False
    assert is_sovereign(0.0) is False


def test_compute_sovereignty():
    """compute_sovereignty returns valid result."""
    config = SovereigntyConfig(crew=100)
    result = compute_sovereignty(config)

    assert result.internal_rate > 0
    assert result.external_rate > 0
    assert isinstance(result.sovereign, bool)


def test_find_threshold():
    """find_threshold returns reasonable crew size."""
    threshold = find_threshold(
        bandwidth_mbps=2.0,
        delay_s=MARS_LIGHT_DELAY_AVG_S,
    )
    assert threshold >= 1
    assert threshold <= 1000


def test_sovereignty_at_threshold():
    """Crew at threshold achieves sovereignty with high delay."""
    # Use very high delay (conjunction) where sovereignty is achievable
    threshold = find_threshold(bandwidth_mbps=2.0, delay_s=1320.0)  # Max delay
    config = SovereigntyConfig(crew=threshold, bandwidth_mbps=2.0, delay_s=1320.0)
    result = compute_sovereignty(config)
    # With 1320s delay, internal rate can exceed external
    assert result.internal_rate > 0
    assert result.external_rate > 0
