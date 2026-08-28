"""The canonical key, measured on the pairs that defeated similarity.

`calibrate_l2.py` asked whether a similarity threshold exists that separates a
paraphrase from an opposite, and answered no: the ranges are inverted, so every
cut-off either serves a wrong answer or loses most paraphrases (finding 06).
This script runs the SAME pairs - imported, not copied, so the comparison
cannot drift - through the canonical `(entity, attribute)` key instead.

Two numbers come out, and only one of them is allowed to fail:

  SAFETY  pairs whose answers differ must not share a key. Zero, or the tier
          serves one question the other's answer and the exit code says so.
  RECALL  paraphrases that name a glossary entity should share a key. Partial
          by design - a question this tier cannot name produces no key, which
          is a miss, which is merely the model call that would have happened.

Note what this script does NOT need: an embedding, a provider, a key, a
network. `calibrate_l2.py` cannot run without credentials because similarity
is a property of a model; this runs offline because a canonical key is a
property of two tables you can read. That difference is the finding.

    PYTHONPATH=. uv run python scripts/calibrate_canonical.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.retrieval import GlossaryTable
from scripts.calibrate_l2 import DISTINCT, NEIGHBOURS, PARAPHRASES

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus.json"


def _fmt(key: tuple[str, str] | None) -> str:
    return f"{key[0]}.{key[1]}" if key else "-"


def main() -> int:
    table = GlossaryTable(json.loads(CORPUS.read_text(encoding="utf-8"))["glossary"])
    k = table.canonical_key

    collisions: list[tuple[str, str, tuple[str, str]]] = []
    shared = in_scope = 0

    for name, pairs, want in (
        ("PARAPHRASE", PARAPHRASES, "hit"),
        ("NEIGHBOUR", NEIGHBOURS, "miss"),
        ("DISTINCT", DISTINCT, "miss"),
    ):
        print(f"\n{name}  (we want a {want.upper()})")
        for a, b in pairs:
            ka, kb = k(a), k(b)
            same = ka is not None and ka == kb
            if want == "hit":
                # Only pairs the glossary can name are counted against recall;
                # the rest are out of scope, not failures.
                if ka is not None or kb is not None:
                    in_scope += 1
                    shared += same
                verdict = "hit " if same else ("MISS" if (ka or kb) else "n/a ")
            else:
                if same:
                    collisions.append((a, b, ka))  # type: ignore[arg-type]
                verdict = "COLLIDE" if same else "miss"
            print(f"  {verdict:8s} {_fmt(ka):22s} {_fmt(kb):22s} {a[:34]:34s} | {b[:34]}")

    print("\n" + "=" * 78)
    total_adversarial = len(NEIGHBOURS) + len(DISTINCT)
    print(f"SAFETY  {total_adversarial - len(collisions)}/{total_adversarial} "
          f"adversarial pairs kept apart")
    if in_scope:
        print(f"RECALL  {shared}/{in_scope} paraphrases that name an entity share one key "
              f"({len(PARAPHRASES) - in_scope} of {len(PARAPHRASES)} out of scope)")

    if collisions:
        print("\nCOLLISIONS - each one serves one question the other's answer:")
        for a, b, key in collisions:
            print(f"  {_fmt(key)}\n    {a}\n    {b}")
        print("\nFix the row, not a threshold: see _METRIC_ALIASES and")
        print("_ATTRIBUTE_CUES in app/retrieval.py.")
        return 1

    print("\nNo adversarial pair shares a key. Unlike a threshold, this is not a")
    print("trade against recall: the pairs that stay apart do so because the")
    print("glossary names them differently or does not name them at all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
