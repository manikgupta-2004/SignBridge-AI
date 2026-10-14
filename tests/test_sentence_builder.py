"""
Unit tests for SignBridge AI Sentence Builder.
"""

from src.sentence_builder import SentenceBuilder


def test_sentence_builder_commits_letter_after_hold():
    builder = SentenceBuilder(hold_duration_frames=3, cooldown_frames=2)
    
    # Send 3 frames of stable 'H'
    r1 = builder.update("H", is_stable=True)
    r2 = builder.update("H", is_stable=True)
    r3 = builder.update("H", is_stable=True)

    assert r1 is None
    assert r2 is None
    assert r3 == "H"
    assert builder.get_text() == "H"


def test_sentence_builder_debouncing_prevents_repeat_spam():
    builder = SentenceBuilder(hold_duration_frames=3, cooldown_frames=5)
    
    # Send 'A' continuously for 12 frames
    committed = []
    for _ in range(12):
        res = builder.update("A", is_stable=True)
        if res:
            committed.append(res)

    # Should only commit once without a neutral break or re-arm
    assert len(committed) == 1
    assert builder.get_text() == "A"


def test_sentence_builder_rearms_after_neutral():
    builder = SentenceBuilder(hold_duration_frames=3, cooldown_frames=2)
    
    # First 'A'
    for _ in range(3):
        builder.update("A", is_stable=True)
    assert builder.get_text() == "A"

    # Neutral pose 'nothing'
    builder.update("nothing", is_stable=True)

    # Second 'A'
    for _ in range(3):
        builder.update("A", is_stable=True)
    assert builder.get_text() == "AA"


def test_sentence_builder_space_and_delete():
    builder = SentenceBuilder(hold_duration_frames=2, cooldown_frames=2)
    
    # Commit 'H', 'I'
    for _ in range(2): builder.update("H", is_stable=True)
    builder.update("nothing", is_stable=True)
    for _ in range(2): builder.update("I", is_stable=True)
    builder.update("nothing", is_stable=True)

    assert builder.get_text() == "HI"

    # Commit space
    for _ in range(2): builder.update("space", is_stable=True)
    assert builder.get_text() == "HI "

    # Commit delete
    builder.update("nothing", is_stable=True)
    for _ in range(2): builder.update("del", is_stable=True)
    assert builder.get_text() == "HI"


def test_sentence_builder_clear_and_undo():
    builder = SentenceBuilder(hold_duration_frames=2, cooldown_frames=2)
    for _ in range(2): builder.update("A", is_stable=True)
    builder.update("nothing", is_stable=True)
    for _ in range(2): builder.update("B", is_stable=True)

    assert builder.get_text() == "AB"

    # Undo
    builder.undo()
    assert builder.get_text() == "A"

    # Clear
    builder.clear()
    assert builder.get_text() == ""


def test_sentence_builder_get_current_word():
    builder = SentenceBuilder()
    assert builder.get_current_word() == ""
    
    builder.current_text = "HELLO HO"
    assert builder.get_current_word() == "HO"
    
    builder.current_text = "HELLO "
    assert builder.get_current_word() == ""


def test_sentence_builder_suggestions_and_manual_accept():
    builder = SentenceBuilder()
    builder.current_text = "HE"
    
    suggestions = builder.get_suggestions()
    assert "HELLO" in suggestions
    
    # Accept HELLO
    builder.accept_suggestion("HELLO")
    assert builder.get_text() == "HELLO "
    
    # After accepting HELLO, suggestions should show next-word predictions like HOW, MY, etc.
    next_suggs = builder.get_suggestions()
    assert len(next_suggs) > 0
    assert "HOW" in next_suggs or "MY" in next_suggs or "I" in next_suggs
    
    # Accept next word HOW
    builder.accept_suggestion("HOW")
    assert builder.get_text() == "HELLO HOW "
    
    # Undo reverts to HELLO 
    builder.undo()
    assert builder.get_text() == "HELLO "


def test_sentence_builder_auto_accept_on_space():
    # When auto_accept_on_space=True, space completes the top suggestion
    builder = SentenceBuilder(hold_duration_frames=2, cooldown_frames=2, auto_accept_on_space=True)
    
    # Spell H - E
    for _ in range(2): builder.update("H", is_stable=True)
    builder.update("nothing", is_stable=True)
    for _ in range(2): builder.update("E", is_stable=True)
    builder.update("nothing", is_stable=True)
    assert builder.get_text() == "HE"
    
    # Now sign space: should auto-complete to HELLO + space
    for _ in range(2): builder.update("space", is_stable=True)
    assert builder.get_text() == "HELLO "


def test_sentence_builder_abbreviation_expansion_on_space():
    builder = SentenceBuilder(hold_duration_frames=2, cooldown_frames=2, auto_accept_on_space=True)
    
    # Spell T - Y
    for _ in range(2): builder.update("T", is_stable=True)
    builder.update("nothing", is_stable=True)
    for _ in range(2): builder.update("Y", is_stable=True)
    builder.update("nothing", is_stable=True)
    assert builder.get_text() == "TY"
    
    # Sign space -> should expand to "THANK YOU "
    for _ in range(2): builder.update("space", is_stable=True)
    assert builder.get_text() == "THANK YOU "


def test_sentence_builder_phrase_gesture_commit():
    builder = SentenceBuilder(hold_duration_frames=2, cooldown_frames=2)
    # Commit phrase gesture "I LOVE YOU"
    for _ in range(2): builder.update("I LOVE YOU", is_stable=True)
    assert builder.get_text() == "I LOVE YOU "

    builder.update("nothing", is_stable=True)

    # Next commit "YES"
    for _ in range(2): builder.update("YES", is_stable=True)
    assert builder.get_text() == "I LOVE YOU YES "


