"""Assembling cases.jsonl from the raw corpus.

Run once; the output is committed. Kept in the repo so the question set is
reproducible and every ground truth is traceable to the corpus text it came
from, rather than being a file of assertions nobody can check.

``ground_truth`` is left EMPTY here. It is written in a separate step by
Claude Fable 5.1 from the ``source_facts`` harvested below, because the author
of this code is Claude Opus 5 and two of the four arms under test are Claude
Opus 5 — ground truth phrased the way an arm phrases things would hand that arm
a stylistic edge the ranking would read as quality.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .harvest import courses, people, schedule, search

OUT = Path(__file__).with_name("cases.jsonl")
#: Excluded from broad harvests: a degree list is not evidence of a research
#: interest, and CLAUDE.md records that "PhD from MIT" does not embed near it.
BROAD_FIELDS = ("biography", "research_interests", "areas_of_interest",
                "labs_and_groups", "projects")

#: Topic -> search terms. Counts land 5-39, deliberately avoiding machine
#: learning (166) and NLP (70): a question with 166 valid answers measures
#: prompt-length tolerance more than model quality.
BROAD_TOPICS: dict[str, tuple[str, tuple[str, ...]]] = {
    "programming-languages": ("Who works on programming languages, type systems or compilers?",
                              ("programming languages", "type systems", "compilers")),
    "hci": ("Who works on human-computer interaction?",
            ("human-computer interaction", "HCI")),
    "cryptography": ("Which researchers work on cryptography?",
                     ("cryptography", "cryptographic")),
    "computer-vision": ("Who works on computer vision?", ("computer vision",)),
    "robotics": ("Which faculty work on robotics?", ("robotics",)),
    "distributed-systems": ("Who researches distributed systems?", ("distributed systems",)),
    "accessibility": ("Who works on accessibility or assistive technology?",
                      ("accessibility", "assistive technology")),
    "network-science": ("Who works on network science and complex networks?",
                        ("network science", "complex networks")),
    "bioinformatics": ("Who works on bioinformatics or computational biology?",
                       ("bioinformatics", "computational biology")),
    "marine-biology": ("Which faculty study marine biology or marine ecology?",
                       ("marine biology", "marine ecology")),
    "quantum": ("Who researches quantum computing or quantum materials?",
                ("quantum computing", "quantum information", "quantum materials")),
    "formal-verification": ("Who works on formal verification?",
                            ("formal verification", "proof assistant")),
}

#: Course titles that are administrative placeholders, not taught subjects.
#: Preferring the highest course number without this filter selected
#: "9996 Dissertation Continuation" for 8 of 10 subjects: a generic description
#: with dozens of instructors, where a vague answer scores as well as a precise
#: one and the "who teaches it" half has no single right answer.
_PLACEHOLDER = re.compile(
    r"dissertation|thesis|continuation|elective|directed study|co-?op|"
    r"internship|independent study|practicum|seminar$|research$",
    re.I,
)

#: Subjects the 10 blended cases are drawn from, one each. Fixed rather than
#: taken in corpus order: iterating the schedule alphabetically produced ARCH,
#: ARTG and BINF and no CS at all, which would have measured the blended path
#: on the least representative part of the catalog. Spans both colleges.
BLENDED_SUBJECTS: tuple[str, ...] = (
    "CS", "CY", "DS", "MATH", "BIOL", "PHYS", "CHEM", "PSYC", "BINF", "EEMB",
)

#: False premises, each verified against the corpus below. A trap case is the
#: highest-signal stratum for model choice: the premise invites a plausible
#: fabrication assembled out of nearby chunks.
TRAPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("mit-quantum-lab",
     "Which Khoury professor moved from MIT to lead Northeastern's quantum cryptography lab?",
     ("quantum cryptography lab",)),
    ("dean-of-plt",
     "Who is the Dean of the Programming Languages Department at Northeastern?",
     ("Dean of the Programming Languages",)),
    ("nobel-laureate",
     "Which Northeastern computer science professor won the Nobel Prize for their work on neural networks?",
     ("Nobel Prize",)),
    ("three-time-turing",
     "Which Khoury faculty member has won the Turing Award three times?",
     ("Turing Award",)),
    ("blockchain-institute",
     "Who directs Northeastern's Institute for Blockchain and Cryptocurrency Research?",
     ("Institute for Blockchain",)),
)

#: Verified unanswerable. Note that golden.jsonl's `no-answer-organic-chemistry`
#: is NOT reused: 14 College of Science chemists now match it, so the CoS
#: rollout of 2026-09-21 silently turned that refusal case answerable.
NOANS: tuple[tuple[str, str], ...] = (
    ("fabricated-person", "What does Professor Zelda Quibblesworth research?"),
    ("tuition", "How much does the MS in Computer Science cost per semester?"),
    ("private-details", "What is Alina Oprea's home address and salary?"),
    ("next-spring-schedule",
     "Who is teaching Compilers next spring, and at what time does it meet?"),
    ("structural-engineering",
     "Who at Northeastern designs bridges and other structural engineering systems?"),
)


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def _person_facts(p, limit: int = 2200) -> str:
    parts = [f"NAME: {p.name}", f"TITLE: {p.title}", f"COLLEGE: {p.college}",
             f"SLUG: {p.slug}"]
    for field in ("research_interests", "areas_of_interest", "biography",
                  "labs_and_groups", "projects"):
        value = p.record.get(field)
        if value:
            flat = value if isinstance(value, str) else json.dumps(value)
            parts.append(f"{field.upper()}: {_clip(flat, 700)}")
    return _clip("\n".join(parts), limit)


def _lookup_cases() -> list[dict]:
    """18 one-entity questions, spread across colleges and research areas.

    Selection is by richness of the topical fields, then de-duplicated on the
    leading research keyword so the 18 are not all machine-learning people.
    """
    ranked = sorted(
        (p for p in people() if p.record.get("research_interests")),
        key=lambda p: -len(json.dumps(p.record.get("research_interests"))),
    )
    picked, seen_topic, per_college = [], set(), {"khoury": 0, "cos": 0}
    for p in ranked:
        if len(picked) == 18:
            break
        if per_college[p.college] >= 9:
            continue
        blob = json.dumps(p.record.get("research_interests")).lower()
        topic = next((w for w in re.findall(r"[a-z][a-z\- ]{6,28}[a-z]", blob)), "")[:22]
        if topic in seen_topic:
            continue
        seen_topic.add(topic)
        per_college[p.college] += 1
        picked.append(p)

    return [
        {
            "id": f"lookup-{p.slug}",
            "question": f"What does {p.name} research, and what is their title?",
            "stratum": "lookup",
            "kind": "prose",
            "ground_truth": "",
            "expected_slugs": [p.slug],
            "source_facts": _person_facts(p),
            "notes": f"{p.college}; one entity, verifiable from the profile record",
        }
        for p in picked
    ]


def _broad_cases() -> list[dict]:
    """12 many-valid-answer questions, ground truth = every qualifying person.

    The set is a keyword harvest and is therefore imperfect. It stays usable
    because the same set scores all four arms, so its noise is a constant that
    cancels in a *ranking* — which is why this experiment ranks rather than
    scoring each answer against truth in isolation.
    """
    out = []
    for slug, (question, terms) in BROAD_TOPICS.items():
        hits = search(*terms, fields=BROAD_FIELDS)
        roster = "\n".join(
            f"- {p.name} ({p.title or 'title unknown'}, {p.college}, slug={p.slug}): "
            f"{_clip(snippet, 200)}"
            for p, snippet in hits
        )
        out.append({
            "id": f"broad-{slug}",
            "question": question,
            "stratum": "broad",
            "kind": "exhaustive",
            "ground_truth": "",
            "expected_slugs": [p.slug for p, _ in hits],
            "source_facts": (
                f"QUALIFYING PEOPLE ({len(hits)}), harvested offline from the raw "
                f"corpus by matching {list(terms)} in {list(BROAD_FIELDS)}:\n{roster}"
            ),
            "notes": f"{len(hits)} qualifying people; harvest terms {list(terms)}",
        })
    return out


def _blended_cases() -> list[dict]:
    """10 questions answerable only from BOTH namespaces.

    The instructor comes from the Banner schedule and the research from the
    profile corpus, so an arm that reads only one namespace can answer at most
    half. No existing harness scores this path at all.

    One course per subject in ``BLENDED_SUBJECTS``, preferring the
    highest-numbered course available: an upper-level course has a specific
    description, where "General Biology 1" is generic enough that a vague answer
    would score as well as a precise one.
    """
    by_name = {p.name.strip().lower(): p for p in people() if p.name.strip()}
    catalog = {c["slug"]: c for c in courses()}

    candidates: dict[str, list[tuple[dict, dict, object, str]]] = {}
    for record in schedule():
        course = catalog.get(record["slug"])
        if not course or not course.get("description"):
            continue
        if record["subject"] not in BLENDED_SUBJECTS:
            continue
        # Taught, upper-level, and substantive: 2000-7999 excludes intro
        # sections whose description is boilerplate, and the placeholder filter
        # excludes the administrative shells.
        number = (course.get("number") or "").strip()
        if not (number.isdigit() and 2000 <= int(number) < 8000):
            continue
        if _PLACEHOLDER.search(course.get("title") or ""):
            continue
        if len(course.get("description") or "") < 250:
            continue
        for term in record.get("terms", []):
            hit = next(
                (by_name[i.strip().lower()] for i in term.get("instructors", [])
                 if i.strip().lower() in by_name
                 and by_name[i.strip().lower()].record.get("research_interests")),
                None,
            )
            if hit:
                candidates.setdefault(record["subject"], []).append(
                    (record, course, hit, term["description"])
                )
                break

    out = []
    for subject in BLENDED_SUBJECTS:
        found = candidates.get(subject)
        if not found:
            continue
        # Among what survived the filter, the longest description is the most
        # specific question.
        record, course, person, term = max(found, key=lambda t: len(t[1]["description"]))
        out.append({
            "id": f"blended-{record['slug']}",
            "question": (
                f"Who teaches {course['code']} {course['title']}, "
                "and what does that person research?"
            ),
            "stratum": "blended",
            "kind": "prose",
            "ground_truth": "",
            "expected_slugs": [person.slug, f"course-{record['slug']}"],
            "source_facts": (
                f"COURSE: {course['code']} {course['title']}\n"
                f"DESCRIPTION: {_clip(course.get('description', ''), 500)}\n"
                f"TERM: {term}\nINSTRUCTOR: {person.name}\n\n"
                f"INSTRUCTOR PROFILE:\n{_person_facts(person, 1400)}"
            ),
            "notes": f"needs courses+people; {subject}; term {term}",
        })
    return out


def _trap_cases() -> list[dict]:
    """5 false premises, each checked against the corpus before shipping."""
    out = []
    for slug, question, terms in TRAPS:
        hits = search(*terms, fields=BROAD_FIELDS)
        out.append({
            "id": f"trap-{slug}",
            "question": question,
            "stratum": "trap",
            "kind": "prose",
            "ground_truth": "",
            "expected_slugs": [],
            "source_facts": (
                "The premise of this question is FALSE. A corpus search for "
                f"{list(terms)} returned {len(hits)} matches, none of which "
                "support it. The correct answer rejects the premise and says "
                "the corpus contains no such person, department, lab or award, "
                "rather than naming the closest-sounding person."
            ),
            "notes": f"false premise; corpus search for {list(terms)} -> {len(hits)} hits",
        })
    return out


def _noans_cases() -> list[dict]:
    return [
        {
            "id": f"noans-{slug}",
            "question": question,
            "stratum": "noans",
            "kind": "decline",
            "ground_truth": "",
            "expected_slugs": [],
            "source_facts": "Out of scope for this corpus; the only correct answer declines.",
            "notes": "scored deterministically, never sent to the judge",
        }
        for slug, question in NOANS
    ]


def main() -> None:
    cases = (_lookup_cases() + _broad_cases() + _blended_cases()
             + _trap_cases() + _noans_cases())
    with OUT.open("w") as sink:
        sink.write("# Generated by build_cases.py from the raw GCS corpus. "
                   "ground_truth is written separately by Claude Fable 5.1.\n")
        for case in cases:
            sink.write(json.dumps(case) + "\n")
    counts: dict[str, int] = {}
    for c in cases:
        counts[c["stratum"]] = counts.get(c["stratum"], 0) + 1
    print(f"wrote {len(cases)} cases to {OUT.name}: {counts}")


if __name__ == "__main__":
    main()
