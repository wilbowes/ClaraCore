"""
Memory output validation for Clara.
Validates summarisation model outputs before they're committed to the DB.
Catches common failure modes: raw conversation content, content loss,
role label leakage, and echo-back of existing memory.
"""

import re
import logging

log = logging.getLogger("clara-validator")


# ─── Failure modes ────────────────────────────────────────────────────

# Patterns that suggest raw conversation content leaked into output
CONVERSATION_ARTIFACTS = [
    r"^\[.*?\].*?:",           # timestamp headers like [Monday 14 April, 10:34 AM]
    r"^(user|assistant):",     # role labels
    r"^I need to flag",        # Clara speaking in first person about herself
    r"^I don't actually",
    r"^I was making that up",
    r"^Sorry about that",
    r"^I can see",
    r"^I've noted",
]

# Patterns that suggest the model is speaking as Clara rather than writing memory
FIRST_PERSON_CLARA = [
    r"\bI (can|will|have|don't|didn't|was|am)\b",
    r"\bI need to\b",
    r"\bI should\b",
]


class ValidationResult:
    def __init__(self, ok: bool, reason: str | None = None):
        self.ok = ok
        self.reason = reason

    def __bool__(self):
        return self.ok

    def __repr__(self):
        return f"ValidationResult(ok={self.ok}, reason={self.reason!r})"


def _check_conversation_artifacts(text: str) -> ValidationResult:
    """Reject if output looks like raw conversation content."""
    lines = text.strip().splitlines()
    for pattern in CONVERSATION_ARTIFACTS:
        for line in lines[:5]:  # check first 5 lines — artifacts usually appear early
            if re.match(pattern, line.strip(), re.IGNORECASE):
                return ValidationResult(False, f"Conversation artifact detected: {line.strip()[:60]!r}")
    return ValidationResult(True)


def _check_first_person(text: str, memory_type: str) -> ValidationResult:
    """
    Dossiers and rhythms should be third person.
    Conversation/day summaries are also third person.
    Reject if Clara is speaking as herself.
    """
    # Only check first 3 sentences — opening is most diagnostic
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())[:3]
    opening = " ".join(sentences)
    for pattern in FIRST_PERSON_CLARA:
        if re.search(pattern, opening, re.IGNORECASE):
            return ValidationResult(False, f"First-person Clara voice detected in {memory_type}")
    return ValidationResult(True)


def _check_length(text: str, memory_type: str,
                  min_chars: int, max_chars: int) -> ValidationResult:
    """Reject if output is suspiciously short or long."""
    n = len(text.strip())
    if n < min_chars:
        return ValidationResult(False, f"{memory_type} too short ({n} chars, min {min_chars})")
    if n > max_chars:
        return ValidationResult(False, f"{memory_type} too long ({n} chars, max {max_chars})")
    return ValidationResult(True)


def _check_content_loss(new_text: str, existing_text: str | None,
                         memory_type: str, threshold: float = 0.5) -> ValidationResult:
    """
    Reject if new output is dramatically shorter than existing memory.
    Suggests the model dropped content rather than updated it.
    Only applies when existing memory is substantial.
    """
    if not existing_text or len(existing_text.strip()) < 100:
        return ValidationResult(True)  # nothing to compare against

    ratio = len(new_text.strip()) / len(existing_text.strip())
    if ratio < threshold:
        return ValidationResult(
            False,
            f"{memory_type} content loss: new is {ratio:.0%} of existing "
            f"({len(new_text)} vs {len(existing_text)} chars)"
        )
    return ValidationResult(True)


def _check_echo(new_text: str, existing_text: str | None,
                memory_type: str, threshold: float = 0.95) -> ValidationResult:
    """
    Reject if new output is nearly identical to existing memory.
    Suggests the model echoed back the input rather than processing it.
    """
    if not existing_text or len(existing_text.strip()) < 50:
        return ValidationResult(True)

    # Normalise whitespace for comparison
    def _norm(s):
        return re.sub(r'\s+', ' ', s.strip().lower())

    n = _norm(new_text)
    e = _norm(existing_text)

    # Simple overlap ratio
    if n == e:
        return ValidationResult(False, f"{memory_type} is identical to existing — model echoed input")

    # Check if new is a substring of existing (content loss via truncation)
    if len(n) > 50 and n in e:
        return ValidationResult(False, f"{memory_type} is a subset of existing — possible truncation")

    return ValidationResult(True)


# ─── Public validators ────────────────────────────────────────────────

def validate_dossier(text: str, existing: str | None = None) -> ValidationResult:
    checks = [
        _check_conversation_artifacts(text),
        _check_first_person(text, "dossier"),
        _check_length(text, "dossier", min_chars=50, max_chars=2000),
        _check_content_loss(text, existing, "dossier", threshold=0.4),
        _check_echo(text, existing, "dossier"),
    ]
    for result in checks:
        if not result:
            return result
    return ValidationResult(True)


def validate_conversation_summary(text: str) -> ValidationResult:
    checks = [
        _check_conversation_artifacts(text),
        _check_first_person(text, "conversation summary"),
        _check_length(text, "conversation summary", min_chars=50, max_chars=3000),
    ]
    for result in checks:
        if not result:
            return result
    return ValidationResult(True)


def validate_day_summary(text: str) -> ValidationResult:
    checks = [
        _check_conversation_artifacts(text),
        _check_first_person(text, "day summary"),
        _check_length(text, "day summary", min_chars=100, max_chars=4000),
    ]
    for result in checks:
        if not result:
            return result
    return ValidationResult(True)


def validate_core_memory(text: str, existing: str | None = None) -> ValidationResult:
    checks = [
        _check_conversation_artifacts(text),
        _check_first_person(text, "core memory"),
        _check_length(text, "core memory", min_chars=200, max_chars=8000),
        _check_content_loss(text, existing, "core memory", threshold=0.5),
        _check_echo(text, existing, "core memory"),
    ]
    for result in checks:
        if not result:
            return result
    return ValidationResult(True)


def validate_rhythms(text: str, existing: str | None = None) -> ValidationResult:
    checks = [
        _check_conversation_artifacts(text),
        _check_first_person(text, "household rhythms"),
        _check_length(text, "household rhythms", min_chars=100, max_chars=5000),
        _check_content_loss(text, existing, "household rhythms", threshold=0.4),
        _check_echo(text, existing, "household rhythms"),
    ]
    for result in checks:
        if not result:
            return result
    return ValidationResult(True)