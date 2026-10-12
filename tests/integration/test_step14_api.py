import pytest
from fastapi.testclient import TestClient
from trading_ai.server.app import app
from trading_ai.schemas.enums import SourceDomain, SourceAuthority, RelationshipType, TradeDecisionState

@pytest.fixture
def client():
    return TestClient(app)

def test_api_trade_logic_status(client):
    res = client.get("/api/trade-logic/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ACTIVE"
    assert "domains" in data
    assert "01_YouTube_Lectures" in data["domains"]
    assert "02_Mentorship_Sessions" in data["domains"]

def test_api_trade_logic_domains(client):
    res = client.get("/api/trade-logic/domains")
    assert res.status_code == 200
    domains = res.json()
    assert len(domains) == 4
    names = [d["domain"] for d in domains]
    assert "01_YouTube_Lectures" in names
    assert "02_Mentorship_Sessions" in names

def test_api_knowledge_ingest_and_retrieval(client):
    payload = {
        "title": "API Ingested FVG Rule",
        "content": "Look for 15m fair value gaps after London open sweep",
        "domain": "youtube_lectures",
        "rule_type": "SETUP_RULE",
        "authority": "APPROVED_LECTURE",
        "confidence": 0.88,
        "symbols": ["EURUSD"],
        "timeframes": ["15m"]
    }
    res = client.post("/api/trade-logic/knowledge/ingest", json=payload)
    assert res.status_code == 200
    item = res.json()
    assert item["title"] == payload["title"]
    assert item["source_domain"] in ["youtube_lectures", "01_YouTube_Lectures"]
    assert item["confidence"] == 0.88
    item_id = item["item_id"]

    # Query knowledge
    res_q = client.get(f"/api/trade-logic/knowledge?query=FVG&domain=youtube_lectures")
    assert res_q.status_code == 200
    items = res_q.json()
    assert any(i["item_id"] == item_id for i in items)

def test_api_relationship_create_and_query(client):
    # Ingest two items first
    p1 = {
        "title": "Rule A",
        "content": "Enter at breaker block",
        "domain": "youtube_lectures",
        "rule_type": "ENTRY"
    }
    p2 = {
        "title": "Filter B",
        "content": "Only take breaker block if aligned with daily trend",
        "domain": "mentorship_sessions",
        "rule_type": "FILTER"
    }
    i1 = client.post("/api/trade-logic/knowledge/ingest", json=p1).json()
    i2 = client.post("/api/trade-logic/knowledge/ingest", json=p2).json()

    # Link them
    link_payload = {
        "source_item_id": i2["item_id"],
        "target_item_id": i1["item_id"],
        "relationship_type": "REFINES",
        "description": "Mentorship rule B refines YouTube breaker block entry"
    }
    res = client.post("/api/trade-logic/relationships/create", json=link_payload)
    assert res.status_code == 200
    rel = res.json()
    assert rel["relationship_type"] == "REFINES"

    # Query relationships
    res_rel = client.get(f"/api/trade-logic/relationships?item_id={i1['item_id']}")
    assert res_rel.status_code == 200
    rels = res_rel.json()
    assert len(rels) >= 1

def test_api_trade_logic_evaluate(client):
    payload = {
        "symbol": "EURUSD",
        "timeframe": "15m",
        "direction": "BUY",
        "current_price": 1.0850,
        "high_timeframe_bias": "BULLISH",
        "market_structure": "SWEEP_AND_DISPLACEMENT"
    }
    res = client.post("/api/trade-logic/evaluate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "confluence" in data
    assert "why_not_trade" in data
    assert "trade_decision" in data
    decision = data["trade_decision"]
    assert "decision_state" in decision
    assert "evidence_references" in decision

def test_api_trade_logic_journal_feedback(client):
    res = client.get("/api/trade-logic/journal-feedback?symbol=EURUSD&limit=5")
    assert res.status_code == 200
    data = res.json()
    assert "symbol" in data
    assert "validated_lessons" in data
    assert "mistakes_to_avoid" in data
