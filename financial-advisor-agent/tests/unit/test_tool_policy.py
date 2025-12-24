from app.financial_advisor.guardrails.tool_policy import validate_tool_call

def test_allow_quote():
    d = validate_tool_call("get_quote", {"symbol": "AAPL"})
    assert d.allow

def test_block_unknown_tool():
    d = validate_tool_call("delete_database", {})
    assert not d.allow

def test_block_large_history():
    d = validate_tool_call("get_history", {"symbol": "AAPL", "days": 99999})
    assert not d.allow
