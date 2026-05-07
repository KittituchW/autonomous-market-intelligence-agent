import unittest

from amia.quality.briefing_quality import (
    build_fallback_briefing,
    replace_sources_block,
    strip_reasoning,
    validate_briefing,
)


SOURCES = [
    {
        "id": 1,
        "ticker": "NVDA",
        "source": "Barchart.com",
        "data_type": "news",
        "sentiment": None,
        "url": "https://example.com/news",
        "snippet": "AI stocks recovered in April as investor optimism returned.",
    },
    {
        "id": 2,
        "ticker": "NVDA",
        "source": "stocktwits",
        "data_type": "social",
        "sentiment": "bullish",
        "url": "https://stocktwits.com/message/1",
        "snippet": "Retail traders remain bullish on NVDA.",
    },
    {
        "id": 3,
        "ticker": "NVDA",
        "source": "hackernews",
        "data_type": "social",
        "sentiment": None,
        "url": "https://news.ycombinator.com/item?id=1",
        "snippet": "Developers debated whether CUDA remains a moat.",
    },
]


class BriefingQualityTests(unittest.TestCase):
    def test_strip_reasoning_removes_think_blocks(self):
        text = (
            "<think>private chain of thought</think>\n"
            "Final Answer: **Summary:** clean"
        )
        self.assertEqual(strip_reasoning(text), "**Summary:** clean")

    def test_replace_sources_block_uses_only_body_citations(self):
        briefing = """**Summary:** NVDA has a news signal [1].

**Key Signals:**
- News: cited [1]
- Retail sentiment: uncited
- Risks: cited [3]

**Watch:** Watch updates.

**Sources:**
[1] https://hallucinated.example
[2] https://hallucinated.example
"""
        cleaned = replace_sources_block(briefing, SOURCES)

        self.assertIn("1. [NVDA/Barchart.com] https://example.com/news", cleaned)
        self.assertIn(
            "3. [NVDA/hackernews] https://news.ycombinator.com/item?id=1",
            cleaned,
        )
        self.assertNotIn("https://hallucinated.example", cleaned)
        self.assertNotIn("2. [NVDA/stocktwits]", cleaned)

    def test_validate_briefing_catches_invalid_citation(self):
        briefing = """**Summary:** invalid citation [99].

**Key Signals:**
- News: invalid [99]
- Retail sentiment: none
- Risks: none

**Watch:** Watch updates.

**Sources:**
99. [NVDA/example] https://example.com
"""
        quality = validate_briefing(briefing, SOURCES)

        self.assertFalse(quality.ok)
        self.assertTrue(any("invalid citation ids" in w for w in quality.warnings))

    def test_validate_briefing_catches_news_using_only_social_sources(self):
        briefing = """**Summary:** valid citations but wrong source type [2].

**Key Signals:**
- News: This is labeled as news but cites retail chatter [2]
- Retail sentiment: Retail traders remain bullish [2]
- Risks: none

**Watch:** Watch updates.

**Sources:**
2. [NVDA/stocktwits] [bullish] https://stocktwits.com/message/1
"""
        quality = validate_briefing(briefing, SOURCES)

        self.assertFalse(quality.ok)
        self.assertIn("news signal lacks news-source citation", quality.warnings)

    def test_fallback_briefing_is_valid_and_evidence_backed(self):
        fallback = build_fallback_briefing(
            "NVDA",
            SOURCES,
            reason="crew timeout",
        )
        quality = validate_briefing(fallback, SOURCES)

        self.assertTrue(quality.ok, quality.warnings)
        self.assertIn("**Summary:**", fallback)
        self.assertIn("**Key Signals:**", fallback)
        self.assertIn("**Watch:**", fallback)
        self.assertIn("**Sources:**", fallback)
        self.assertIn("https://example.com/news", fallback)

    def test_fallback_briefing_with_no_sources_still_has_sources_section(self):
        fallback = build_fallback_briefing("NVDA", [], reason="retrieval empty")
        quality = validate_briefing(fallback, [])

        self.assertTrue(quality.ok, quality.warnings)
        self.assertIn("**Sources:**", fallback)
        self.assertIn("No retrieved sources.", fallback)


if __name__ == "__main__":
    unittest.main()
