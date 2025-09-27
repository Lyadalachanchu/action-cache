"""
dom_extractors.py
-----------------
Optional DOM-first extractors for specific intents. These are token-free
and help demonstrate cache savings when LLM policy isn't "always".
"""

from datetime import datetime
from typing import Optional
import re

# -- Shared helpers --

def format_birthdate(raw: Optional[str]) -> Optional[str]:
    """Try to normalize YYYY-MM-DD → 'Month D, YYYY'; else return raw."""
    if not raw:
        return raw
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.strftime("%B %d, %Y")
        except Exception:
            pass
    return raw

def looks_like_date(text: str) -> bool:
    return bool(re.search(r"(\d{4}-\d{2}-\d{2})|([A-Za-z]+\s+\d{1,2},\s+\d{4})", text))


# -- Wikipedia birth date extraction (DOM/JSON-LD/heuristic) --

async def extract_birthdate_dom(page) -> Optional[str]:
    """
    Try: infobox .bday, <time datetime>, JSON-LD, then first-paragraph heuristic.
    Returns a raw date string or None.
    """
    return await page.evaluate("""
        () => {
          const b = document.querySelector('.infobox .bday');
          if (b && b.textContent) return b.textContent.trim();
          const t = document.querySelector('.infobox time[datetime]');
          if (t) {
            const dt = t.getAttribute('datetime');
            if (dt) return dt;
          }
          const scripts = document.querySelectorAll('script[type="application/ld+json"]');
          for (const s of scripts) {
            try {
              const data = JSON.parse(s.textContent);
              const arr = Array.isArray(data) ? data : [data];
              for (const item of arr) {
                if (item && item.birthDate) return item.birthDate;
                if (item && item.person && item.person.birthDate) return item.person.birthDate;
              }
            } catch {}
          }
          const p = document.querySelector('.mw-parser-output > p');
          if (p) {
            const txt = p.textContent || "";
            const m1 = txt.match(/\\(born\\s+([A-Za-z]+\\s+\\d{1,2},\\s+\\d{4})\\)/i);
            if (m1) return m1[1];
            const m2 = txt.match(/(\\d{4}-\\d{2}-\\d{2})/);
            if (m2) return m2[1];
          }
          return null;
        }
    """)
