"""
urgency_rule_based.py
─────────────────────
Lightweight keyword-scoring heuristic for urgency detection.
Returns "High" | "Medium" | "Low".
"""

HIGH_KEYWORDS = [
    "urgent", "asap", "immediately", "critical",
    "system down", "emergency", "deadline",
]
MEDIUM_KEYWORDS = [
    "soon", "priority", "update", "review",
    "request", "schedule", "meeting", "approval",
]


def detect_urgency_rule(text: str) -> str:
    """Score the text against keyword lists and return an urgency label."""
    lowered = text.lower()
    score   = (
        sum(2 for kw in HIGH_KEYWORDS   if kw in lowered)
        + sum(1 for kw in MEDIUM_KEYWORDS if kw in lowered)
    )
    if score >= 2:
        return "High"
    if score == 1:
        return "Medium"
    return "Low"
