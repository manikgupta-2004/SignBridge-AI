"""
SignBridge AI - Sentence Construction Engine
Debounces gestural inputs, enforces hold stability, prevents repeat sign spamming,
handles special tokens (SPACE, DEL, NOTHING), and maintains an editable text buffer.
"""

import time
from typing import List, Optional, Tuple

from src.predictive_text import PredictiveTextEngine


class SentenceBuilder:
    """
    Manages text accumulation from gestural streams with re-arm debouncing.
    
    Important Linguistic Distinction:
    This module performs orthographic finger-spelling accumulation (letter-by-letter).
    It does NOT translate continuous American Sign Language (ASL) grammar,
    which relies on non-manual markers, spatial syntax, and facial expressions.
    """

    def __init__(
        self,
        hold_duration_frames: int = 12,
        cooldown_frames: int = 15,
        space_token: str = "space",
        delete_token: str = "del",
        nothing_token: str = "nothing",
        auto_accept_on_space: bool = False
    ):
        """
        Args:
            hold_duration_frames: Number of continuous stable frames before committing character.
            cooldown_frames: Minimum frames to wait after commit before repeating or changing.
            space_token: Category label for adding a space.
            delete_token: Category label for backspace.
            nothing_token: Category label for neutral/rest pose.
            auto_accept_on_space: If True, signing space gesture auto-accepts top suggestion.
        """
        self.hold_duration_frames = hold_duration_frames
        self.cooldown_frames = cooldown_frames
        self.space_token = space_token.lower()
        self.delete_token = delete_token.lower()
        self.nothing_token = nothing_token.lower()
        self.auto_accept_on_space = auto_accept_on_space

        # Predictive smart text engine
        self.predictive_engine = PredictiveTextEngine(max_suggestions=5)

        # Text buffers
        self.current_text: str = ""
        self.history: List[str] = []         # For undo support
        self.recent_tokens: List[str] = []   # Visual log of committed signs

        # State tracking
        self.current_candidate: Optional[str] = None
        self.candidate_hold_count: int = 0
        self.last_committed_token: Optional[str] = None
        self.cooldown_counter: int = 0
        self.is_armed: bool = True           # Ready to accept next gesture

    def update(self, stable_label: str, is_stable: bool) -> Optional[str]:
        """
        Processes a single stabilized frame and commits characters when thresholds are satisfied.

        Args:
            stable_label: Output from PredictionSmoother (or 'Uncertain').
            is_stable: Boolean indicator if temporal consensus was reached.

        Returns:
            The committed token string if a change was enacted, else None.
        """
        # Decrement cooldown counter if active
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        clean_label = stable_label.strip()
        lower_label = clean_label.lower()

        # Handle neutral / nothing pose or unstable frame
        if not is_stable or lower_label == self.nothing_token:
            # Entering neutral re-arms the builder for immediate next character
            self.is_armed = True
            self.candidate_hold_count = 0
            self.current_candidate = None
            return None

        # Check if gesture matches ongoing candidate
        if clean_label == self.current_candidate:
            self.candidate_hold_count += 1
        else:
            self.current_candidate = clean_label
            self.candidate_hold_count = 1

        # Check if held long enough to trigger commitment
        if self.candidate_hold_count >= self.hold_duration_frames:
            # Check re-arm condition:
            # If it's the exact same sign as last committed, must have cooled down or re-armed
            can_commit = (
                self.is_armed or
                (clean_label != self.last_committed_token and self.cooldown_counter == 0)
            )

            if can_commit:
                committed = self._commit_token(clean_label)
                # Reset counters and enforce cooldown
                self.candidate_hold_count = 0
                self.cooldown_counter = self.cooldown_frames
                self.is_armed = False  # Must transition or cool down before repeating
                self.last_committed_token = clean_label
                return committed

        return None

    def get_current_word(self) -> str:
        """Returns the unfinished word actively being spelled (after last space)."""
        if not self.current_text or self.current_text.endswith(" "):
            return ""
        tokens = self.current_text.split()
        return tokens[-1] if tokens else ""

    def get_suggestions(self) -> List[str]:
        """Returns smart word completions / next-word suggestions for current buffer."""
        return self.predictive_engine.get_suggestions(self.current_text)

    def accept_suggestion(self, suggestion: str) -> str:
        """
        Replaces active unfinished word with suggestion (or appends next word)
        and adds trailing space ready for the next word.
        """
        clean_sugg = suggestion.strip().upper()
        if not clean_sugg:
            return self.current_text

        # Save previous state for Undo
        self.history.append(self.current_text)

        if not self.current_text or self.current_text.endswith(" "):
            # Appending a next-word suggestion
            self.current_text += clean_sugg + " "
        else:
            # Completing the active prefix
            tokens = self.current_text.split()
            tokens[-1] = clean_sugg
            self.current_text = " ".join(tokens) + " "

        action = f"[{clean_sugg}]"
        self.recent_tokens.append(action)
        if len(self.recent_tokens) > 20:
            self.recent_tokens.pop(0)

        # Re-arm builder
        self.candidate_hold_count = 0
        self.is_armed = True
        self.last_committed_token = None
        return self.current_text

    def _commit_token(self, token: str) -> str:
        """Executes text mutation based on the token category."""
        # Save previous state for Undo
        self.history.append(self.current_text)

        lower_token = token.lower()

        if lower_token == self.space_token:
            if self.auto_accept_on_space:
                curr_word = self.get_current_word()
                suggestions = self.get_suggestions()
                if curr_word and suggestions:
                    top_sugg = suggestions[0]
                    tokens = self.current_text.split()
                    tokens[-1] = top_sugg
                    self.current_text = " ".join(tokens) + " "
                    action = f"[{top_sugg}]"
                else:
                    if self.current_text and not self.current_text.endswith(" "):
                        self.current_text += " "
                    action = "[SPACE]"
            else:
                # Add space only if not already trailing space
                if self.current_text and not self.current_text.endswith(" "):
                    self.current_text += " "
                action = "[SPACE]"
        elif lower_token == self.delete_token:
            # Backspace previous character
            if self.current_text:
                self.current_text = self.current_text[:-1]
            action = "[DEL]"
        elif lower_token == self.nothing_token:
            # NEVER insert 'nothing'
            action = "[NOTHING]"
        else:
            # Alphabet character (A-Z) or Whole-Word Phrase Gesture
            char_to_add = token.upper()
            if len(char_to_add) > 1:
                # Whole-word / phrase gesture like "I LOVE YOU", "YES", "NO", "STOP", "OKAY"
                if self.current_text and not self.current_text.endswith(" "):
                    self.current_text += " "
                self.current_text += char_to_add + " "
                action = f"[{char_to_add}]"
            else:
                self.current_text += char_to_add
                action = char_to_add

        self.recent_tokens.append(action)
        if len(self.recent_tokens) > 20:
            self.recent_tokens.pop(0)

        return action

    def undo(self) -> bool:
        """Restores previous text state."""
        if self.history:
            self.current_text = self.history.pop()
            return True
        return False

    def clear(self) -> None:
        """Wipes the text buffer."""
        if self.current_text:
            self.history.append(self.current_text)
        self.current_text = ""
        self.recent_tokens.clear()
        self.candidate_hold_count = 0
        self.is_armed = True
        self.last_committed_token = None

    def get_progress_ratio(self) -> float:
        """Returns the hold progress (0.0 to 1.0) towards the next character commit."""
        if self.hold_duration_frames <= 0:
            return 0.0
        return min(1.0, self.candidate_hold_count / self.hold_duration_frames)

    def get_text(self) -> str:
        """Returns the accumulated string."""
        return self.current_text
