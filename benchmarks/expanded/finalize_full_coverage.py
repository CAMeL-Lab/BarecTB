#!/usr/bin/env python3
"""Original light parsing repairs extracted from finalize_full_coverage.py."""
from __future__ import annotations
from collections import Counter
from typing import Any
from fanar2_benchmark_common import DEPRELS

ARABIC_PROCLITICS = {"و", "ف", "ب", "ك", "ل", "س", "ال", "لل"}
PLACEHOLDER_FORM = "_EMPTY_SENT_"


def clean_form(value: Any) -> str:
    """Follow scripts/utils/clean_parsing_predictions.py: remove all FORM whitespace."""
    return "".join(str(value if value is not None else "").split())


def postprocess_parse(parsed: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    """Apply the light repairs used by scripts/utils/post_raw_parsing.py."""
    raw_items = parsed.get("parses")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("A complete retry is required before postprocessing")
    actions: Counter[str] = Counter()
    nodes = []
    for position, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            actions["dropped_non_object_item"] += 1
            continue
        try:
            old_id = int(item.get("id", position))
        except (TypeError, ValueError):
            old_id = position
            actions["replaced_invalid_id"] += 1
        try:
            old_head = int(item.get("head", 0))
        except (TypeError, ValueError):
            old_head = 0
            actions["replaced_invalid_head"] += 1
        old_form = item.get("form", "")
        form = clean_form(old_form)
        if form != old_form:
            actions["removed_form_whitespace"] += 1
        if not form or form == "+":
            actions["dropped_empty_or_plus_form"] += 1
            continue
        deprel = item.get("deprel")
        if deprel not in DEPRELS:
            deprel = "MOD"
            actions["replaced_invalid_deprel"] += 1
        nodes.append(
            {"old_id": old_id, "old_head": old_head, "form": form, "deprel": deprel}
        )

    while nodes and nodes[0]["form"].startswith("+"):
        stripped = nodes[0]["form"].lstrip("+")
        if not stripped:
            nodes.pop(0)
            actions["dropped_empty_initial_enclitic"] += 1
            continue
        if stripped in ARABIC_PROCLITICS and len(nodes) > 1:
            nodes[0]["form"] = stripped + "+"
        else:
            nodes[0]["form"] = stripped
        actions["fixed_sentence_initial_enclitic"] += 1
        break

    for current, following in zip(nodes, nodes[1:]):
        if current["form"].endswith("+") and following["form"].startswith("+"):
            current["form"] = current["form"][:-1]
            actions["fixed_proclitic_before_enclitic"] += 1
    if nodes and nodes[-1]["form"].endswith("+"):
        nodes[-1]["form"] = nodes[-1]["form"][:-1]
        actions["fixed_final_proclitic"] += 1

    if not nodes:
        actions["inserted_empty_sentence_placeholder"] += 1
        return {
            "parses": [
                {"id": 1, "form": PLACEHOLDER_FORM, "head": 0, "deprel": "MOD"}
            ]
        }, dict(actions)

    old_head_map = {}
    old_to_new = {}
    for new_id, node in enumerate(nodes, start=1):
        old_head_map.setdefault(node["old_id"], node["old_head"])
        old_to_new.setdefault(node["old_id"], new_id)
        if node["old_id"] != new_id:
            actions["renumbered_id"] += 1
    surviving = set(old_to_new)

    output = []
    for new_id, node in enumerate(nodes, start=1):
        head = node["old_head"]
        seen = set()
        while head and head not in surviving and head not in seen:
            seen.add(head)
            head = old_head_map.get(head, 0)
        new_head = old_to_new.get(head, 0) if head else 0
        if new_head == new_id:
            new_head = 0
        if new_head != node["old_head"]:
            actions["remapped_head"] += 1
        output.append(
            {
                "id": new_id,
                "form": node["form"],
                "head": new_head,
                "deprel": node["deprel"],
            }
        )
    return {"parses": output}, dict(actions)

