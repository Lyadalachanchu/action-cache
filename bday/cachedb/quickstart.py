# cachedb/quickstart.py
"""
Quickstart for the hybrid cache DB.
1) Initialize DB and migrate existing JSON caches (if present in /mnt/data)
2) Run a sample query for an answer and a plan
"""
from .db import init_db
from .migrate_from_json import run as migrate_run
from .resolver import get_answer, get_plan, robust_click_hint
from .repos import AnswersRepo, PlansRepo, DOMRepo

def main():
    init_db()
    migrate_run()

    # Seed an example answer if migration had nothing
    ans_repo = AnswersRepo()
    ans_repo.put(
        canonical_q="ENTITY:Marie Curie|ATTR:birth_date",
        question_text="When was Marie Curie born?",
        answer_text="7 November 1867",
        confidence=1.0,
        evidence={"note": "seeded example"},
        sources=[{"title":"Wikipedia", "url":"https://en.wikipedia.org/wiki/Marie_Curie"}]
    )

    # Query it back via paraphrase
    print("Answer lookup:")
    print(get_answer("What is the birth date of Marie Curie?"))

    # Seed a plan template
    plan_repo = PlansRepo()
    plan_repo.put(
        intent_key="wikipedia_birth_date",
        goal_text="birth date lookup on Wikipedia",
        plan_json={
            "params": {"entity": "${ENTITY}"},
            "subgoals": [
                "open wikipedia page for ${ENTITY}",
                "find infobox birth date field",
                "extract date string"
            ],
            "actions": [
                {"type": "navigate", "url_tmpl": "https://en.wikipedia.org/wiki/${ENTITY}"},
                {"type": "extract", "role": "infobox.birth_date"}
            ]
        },
        site_domain="wikipedia.org",
        success_rate=0.7,
        version="v1"
    )

    print("Plan lookup:")
    print(get_plan("Find a person's birth date on Wikipedia", site_domain="wikipedia.org"))

    # DOM role recovery example
    dom_repo = DOMRepo()
    dom_repo.put("wikipedia.org", "table.infobox .bday", "infobox.birth_date", "7 November 1867", {"attr":"datetime"}, 0.8)
    print("DOM role search:")
    print(robust_click_hint("infobox birth date", site_domain="wikipedia.org"))

if __name__ == "__main__":
    main()
