"""
SignBridge AI - Temporal Prediction Smoother
Prevents flickering and false positives using rolling window aggregation,
confidence thresholding, and stability scoring.
"""

from collections import Counter, deque
from typing import Deque, List, Optional, Tuple


class PredictionSmoother:
    """
    Stabilizes noisy frame-by-frame neural network predictions across time.
    Uses majority voting and weighted confidence accumulation.
    """

    UNCERTAIN_LABEL = "Uncertain"

    def __init__(
        self,
        window_size: int = 9,
        confidence_threshold: float = 0.78,
        stability_ratio: float = 0.65
    ):
        """
        Args:
            window_size: Number of past frames to retain in rolling buffer.
            confidence_threshold: Minimum confidence required (0.0 to 1.0).
            stability_ratio: Fraction of window frames that must agree (e.g., 0.65 = 65%).
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.stability_ratio = stability_ratio

        self.label_window: Deque[str] = deque(maxlen=window_size)
        self.confidence_window: Deque[float] = deque(maxlen=window_size)

    def update(self, raw_label: str, raw_confidence: float) -> Tuple[str, float, bool]:
        """
        Ingests a new raw prediction and returns the stabilized consensus.

        Args:
            raw_label: The raw category predicted by the CNN for current frame.
            raw_confidence: Confidence score between 0.0 and 1.0.

        Returns:
            Tuple of (stabilized_label, average_confidence, is_stable).
            If confidence is below threshold or consensus is weak, stabilized_label is 'Uncertain'.
        """
        # If single-frame confidence is below threshold, register as Uncertain
        if raw_confidence < self.confidence_threshold:
            self.label_window.append(self.UNCERTAIN_LABEL)
            self.confidence_window.append(raw_confidence)
        else:
            self.label_window.append(raw_label)
            self.confidence_window.append(raw_confidence)

        # Count frequencies in the window
        counts = Counter(self.label_window)
        most_common_label, count = counts.most_common(1)[0]

        # Calculate agreement ratio in the window
        agreement = count / len(self.label_window)
        is_stable = (
            agreement >= self.stability_ratio and
            most_common_label != self.UNCERTAIN_LABEL and
            len(self.label_window) >= min(4, self.window_size)
        )

        # Calculate average confidence for the consensus label
        relevant_confs = [
            conf for lbl, conf in zip(self.label_window, self.confidence_window)
            if lbl == most_common_label
        ]
        avg_confidence = float(sum(relevant_confs) / len(relevant_confs)) if relevant_confs else 0.0

        if not is_stable:
            return self.UNCERTAIN_LABEL, avg_confidence, False

        return most_common_label, avg_confidence, True

    def reset(self) -> None:
        """Clears the temporal history buffer."""
        self.label_window.clear()
        self.confidence_window.clear()

    @property
    def current_agreement(self) -> float:
        """Returns the agreement fraction (0.0 to 1.0) of the most frequent label in window."""
        if not self.label_window:
            return 0.0
        counts = Counter(self.label_window)
        return counts.most_common(1)[0][1] / len(self.label_window)

    def get_history(self) -> List[Tuple[str, float]]:
        """Returns snapshot of current buffer as list of (label, confidence)."""
        return list(zip(self.label_window, self.confidence_window))
