"""Shared fixtures for distiller tests.

Note: the _reset_accelerate_state autouse fixture lives in the parent
conftest (tests/silverspoon_kd/conftest.py) so it covers ALL tests
that create TrainingArguments or Distiller instances — not just those
under tests/silverspoon_kd/distillers/.
"""
