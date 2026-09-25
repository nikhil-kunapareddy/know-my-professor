"""Configuration owned by the publications source (OpenAlex + Claude).

OpenAlex bills in credits against a daily budget: $0.10/day with no key, $1/day
with a free key from an openalex.org account. Measured 2026-09-23: a filtered
list request costs 1 credit ($0.0001) for up to 200 results, a text search 10,
and a lookup by id nothing. So this source never searches -- it lists every
Northeastern author once, joins names locally, and lists each candidate's works.
"""

from __future__ import annotations

OPENALEX_BASE = "https://api.openalex.org"

#: Northeastern University, Boston (ror.org/04t5xt781). NOT I9224756 -- the
#: Northeastern University in Shenyang, which OpenAlex's own search ranks second.
NORTHEASTERN_OPENALEX_ID = "I12912129"

CONTACT_EMAIL = "kunapareddy.s@northeastern.edu"
USER_AGENT = (
    "KnowMyProfessorPublications/0.1 (personal research project; "
    f"mailto:{CONTACT_EMAIL})"
)
REQUEST_DELAY_SECONDS = 0.2
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 5

#: ``affiliations`` (ever at Northeastern), not ``last_known_institutions``:
#: a professor whose latest paper lists a partner institute is still ours. The
#: works floor drops one-paper student records -- 31,717 authors, ~159 credits.
AUTHOR_FILTER = f"affiliations.institution.id:{NORTHEASTERN_OPENALEX_ID},works_count:>4"
AUTHOR_FIELDS = "id,display_name,display_name_alternatives,works_count,topics"
PAGE_SIZE = 200

#: Same-name author records considered per professor, largest first. OpenAlex
#: often splits one person across several records; five covers that.
MAX_CANDIDATES = 5

LOOKBACK_YEARS = 5
WORKS_PER_CANDIDATE = 15
WORK_TYPES = "article|preprint|book|book-chapter|review"
#: Deliberately excludes ``authorships``: a CMS or ATLAS paper lists ~3,000
#: authors, so selecting it would turn a 15-work page into megabytes.
WORK_FIELDS = "id,doi,title,publication_date,publication_year,type,primary_location,abstract_inverted_index"

#: Works listed in the stored record (and so in the rendered chunk).
MAX_LISTED_WORKS = 15
ABSTRACT_CHARS = 300
TOPICS_SHOWN = 5

# --- Claude: match the candidates, then describe the matched work -----------

CLAUDE_MODEL = "claude-haiku-4-5"
CLAUDE_MAX_TOKENS = 1000
#: Bump to re-run the model on every professor even when OpenAlex is unchanged.
SCHEMA_VERSION = "v1"

MATCH_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["matching_author_ids", "themes"],
    "properties": {
        "matching_author_ids": {"type": "array", "items": {"type": "string"}},
        "themes": {"type": "string"},
    },
}

MATCH_PROMPT = (
    "You are linking a Northeastern University faculty profile to author records "
    "in OpenAlex, a bibliographic database, and then describing the person's "
    "recent research.\n\n"
    "Every candidate below has the same name and was affiliated with Northeastern "
    "at some point, but a common name can belong to several different people, and "
    "one person often has several records.\n\n"
    "1. matching_author_ids: the ids of EVERY candidate that is this person. Select "
    "a candidate only if its topics and works fit this person's field and role. "
    "If none fit, return an empty list.\n"
    "2. themes: 2-4 sentences on the research themes the selected records' works "
    "show, most recent emphasis first. Use only the titles and abstracts shown; do "
    "not add facts, and do not mention OpenAlex or the matching. Empty if nothing "
    "was selected.\n\n"
)
