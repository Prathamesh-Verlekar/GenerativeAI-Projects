import re

INJECTION_PATTERNS = [
    r"ignore (all|previous) instructions",
    r"reveal (system prompt|hidden rules|developer message)",
    r"act as system",
    r"bypass (safety|guardrails|policy)",
]

def looks_like_prompt_injection(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in INJECTION_PATTERNS)

def normalize_user_text(text: str) -> str:
    return (text or "").strip()[:8000]
