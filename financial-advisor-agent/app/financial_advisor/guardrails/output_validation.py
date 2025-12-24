RISKY_PHRASES = [
    "guaranteed returns",
    "no risk",
    "sure profit",
    "risk-free",
]

def validate_financial_output(text: str) -> str:
    t = (text or "")
    lower = t.lower()
    if any(p in lower for p in RISKY_PHRASES):
        return (
            "I can’t present anything as guaranteed or risk-free. "
            "I can share risks, scenarios, and educational guidance.\n\n"
            + t
        )
    return t
