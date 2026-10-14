"""
Unit tests for SignBridge AI Predictive Text and Autocomplete Engine.
"""

import pytest
from src.predictive_text import PredictiveTextEngine


def test_empty_text_suggestions():
    engine = PredictiveTextEngine()
    suggestions = engine.get_suggestions("")
    assert len(suggestions) > 0
    assert "HELLO" in suggestions or "HI" in suggestions


def test_prefix_completion_hello():
    engine = PredictiveTextEngine()
    suggestions = engine.get_suggestions("HE")
    assert len(suggestions) > 0
    assert "HELLO" in suggestions
    assert "HELP" in suggestions


def test_prefix_completion_case_insensitive():
    engine = PredictiveTextEngine()
    s_lower = engine.get_suggestions("he")
    s_upper = engine.get_suggestions("HE")
    assert s_lower == s_upper


def test_abbreviation_expansion_ty():
    engine = PredictiveTextEngine()
    suggestions = engine.get_suggestions("TY")
    assert "THANK YOU" in suggestions
    assert engine.expand_abbreviation("TY") == "THANK YOU"


def test_abbreviation_expansion_hru():
    engine = PredictiveTextEngine()
    suggestions = engine.get_suggestions("HRU")
    assert "HOW ARE YOU" in suggestions


def test_next_word_prediction_after_hello():
    engine = PredictiveTextEngine()
    # Sentence ends in space: word boundary
    suggestions = engine.get_suggestions("HELLO ")
    assert len(suggestions) > 0
    assert "HOW" in suggestions or "MY" in suggestions or "I" in suggestions


def test_multi_word_phrase_prediction():
    engine = PredictiveTextEngine()
    suggestions = engine.get_suggestions("HOW ARE ")
    assert "YOU" in suggestions

    suggestions_need = engine.get_suggestions("I NEED ")
    assert "HELP" in suggestions_need or "WATER" in suggestions_need


def test_active_prefix_in_multi_word_sentence():
    engine = PredictiveTextEngine()
    # "HELLO TH" -> should complete "TH" to "THANK", "THAT", etc.
    suggestions = engine.get_suggestions("HELLO TH")
    assert "THANK" in suggestions or "THANKS" in suggestions
    # It should NOT suggest "HOW" here because "TH" is being actively typed
    assert "HOW" not in suggestions
