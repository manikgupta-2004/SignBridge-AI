"""
Unit tests for SignBridge AI Prediction Smoother.
"""

from src.prediction_smoother import PredictionSmoother


def test_smoother_initial_uncertainty():
    smoother = PredictionSmoother(window_size=5, confidence_threshold=0.8, stability_ratio=0.6)
    # Fewer than 4 frames should not be considered stable yet
    label, conf, is_stable = smoother.update("A", 0.95)
    assert not is_stable
    assert label == PredictionSmoother.UNCERTAIN_LABEL


def test_smoother_reaches_stability_with_high_confidence():
    smoother = PredictionSmoother(window_size=5, confidence_threshold=0.75, stability_ratio=0.6)
    for _ in range(5):
        label, conf, is_stable = smoother.update("B", 0.90)

    assert is_stable
    assert label == "B"
    assert conf >= 0.75


def test_smoother_rejects_low_confidence():
    smoother = PredictionSmoother(window_size=5, confidence_threshold=0.80, stability_ratio=0.6)
    for _ in range(5):
        label, conf, is_stable = smoother.update("C", 0.45)  # Under threshold

    assert not is_stable
    assert label == PredictionSmoother.UNCERTAIN_LABEL


def test_smoother_reset():
    smoother = PredictionSmoother(window_size=5, confidence_threshold=0.70)
    for _ in range(5):
        smoother.update("D", 0.9)
    assert smoother.current_agreement > 0.0

    smoother.reset()
    assert smoother.current_agreement == 0.0
    assert len(smoother.label_window) == 0
