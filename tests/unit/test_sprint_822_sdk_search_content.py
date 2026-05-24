"""Sprint 822 — PRSMClient.search_content() SDK method.

Mirrors sprint 808 (`content search`) CLI on the SDK side.
Closes Python integration for content discovery.

  async def search_content(
      self, query: str, *,
      limit: int = 20,
      min_tier: Optional[str] = None,
      exclude_new: bool = False,
  ) -> Dict[str, Any]:
      \"\"\"GET /content/search; return {results, count}.\"\"\"

Pin tests:
- search_content method exists.
- GETs /content/search.
- Query passed as `q` param.
- Default limit = 20.
- --limit override threaded.
- min_tier threaded only when set.
- exclude_new flag serialized to query param.
- Empty results returned verbatim.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


def _fake_session(captured):
    """Build a fake aiohttp session that records query params."""
    class _Resp:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def json(self):
            return captured.get("response", {
                "results": [], "count": 0,
            })

    class _Sess:
        def get(self, url, headers=None, timeout=None, params=None):
            captured["url"] = url
            captured["params"] = params
            return _Resp()

        async def close(self):
            pass

    return _Sess()


async def test_search_content_method_exists():
    from prsm.sdk.client import PRSMClient
    client = PRSMClient()
    assert hasattr(client, "search_content")
    assert callable(client.search_content)


async def test_search_content_gets_content_search():
    from prsm.sdk.client import PRSMClient
    captured = {}
    client = PRSMClient("http://node:8000")
    client._session = _fake_session(captured)

    await client.search_content("hello")
    assert "/content/search" in captured["url"]


async def test_query_passed_as_q():
    from prsm.sdk.client import PRSMClient
    captured = {}
    client = PRSMClient("http://node:8000")
    client._session = _fake_session(captured)

    await client.search_content("my search terms")
    params = captured.get("params") or {}
    assert params.get("q") == "my search terms"


async def test_default_limit_20():
    from prsm.sdk.client import PRSMClient
    captured = {}
    client = PRSMClient("http://node:8000")
    client._session = _fake_session(captured)

    await client.search_content("x")
    params = captured.get("params") or {}
    assert params.get("limit") in (20, "20")


async def test_limit_override():
    from prsm.sdk.client import PRSMClient
    captured = {}
    client = PRSMClient("http://node:8000")
    client._session = _fake_session(captured)

    await client.search_content("x", limit=50)
    params = captured.get("params") or {}
    assert params.get("limit") in (50, "50")


async def test_min_tier_only_when_set():
    """min_tier=None omits the param entirely (matches sprint
    808 CLI behavior: default = no tier filter)."""
    from prsm.sdk.client import PRSMClient
    captured = {}
    client = PRSMClient("http://node:8000")
    client._session = _fake_session(captured)

    await client.search_content("x")
    params = captured.get("params") or {}
    assert "min_tier" not in params

    captured.clear()
    client._session = _fake_session(captured)
    await client.search_content("x", min_tier="medium")
    params = captured.get("params") or {}
    assert params.get("min_tier") == "medium"


async def test_exclude_new_flag():
    from prsm.sdk.client import PRSMClient
    captured = {}
    client = PRSMClient("http://node:8000")
    client._session = _fake_session(captured)

    await client.search_content("x", exclude_new=True)
    params = captured.get("params") or {}
    val = params.get("exclude_new")
    assert val in (True, "true", "True", 1, "1")


async def test_results_returned_verbatim():
    from prsm.sdk.client import PRSMClient
    captured = {
        "response": {
            "results": [
                {"cid": "QmA", "filename": "doc.txt",
                 "creator_tier": "medium"},
            ],
            "count": 1,
        }
    }
    client = PRSMClient("http://node:8000")
    client._session = _fake_session(captured)

    result = await client.search_content("hello")
    assert result["count"] == 1
    assert result["results"][0]["cid"] == "QmA"
