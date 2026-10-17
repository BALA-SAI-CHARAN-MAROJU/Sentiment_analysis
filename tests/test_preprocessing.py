"""
tests/test_preprocessing.py - Unit tests for NLP preprocessing pipeline.
"""

import pytest
from src.preprocessing import clean_text, convert_emojis


def test_convert_emojis():
    text = "Great job! 😍🔥"
    converted = convert_emojis(text)
    assert "love" in converted.lower()
    assert "fire" in converted.lower()


def test_url_and_mention_removal():
    raw = "Check this out @user https://example.com/test #cool"
    cleaned = clean_text(raw)
    assert "@user" not in cleaned
    assert "http" not in cleaned
    assert "example" not in cleaned
    assert "cool" in cleaned


def test_empty_and_none_text():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   ") == ""


def test_negation_preservation():
    raw = "The phone is not good and cannot connect"
    cleaned = clean_text(raw, remove_stopwords=True)
    # Ensure negation words are preserved even when stopwords are removed
    assert "not" in cleaned
    assert "cannot" in cleaned


def test_character_elongation():
    raw = "This is sooooo goooood"
    cleaned = clean_text(raw)
    assert "sooooo" not in cleaned
    assert "goooood" not in cleaned
