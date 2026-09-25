"""Configuration owned by the grants source (NSF Award Search + NIH RePORTER).

Both APIs are free, keyless, and public-domain US government data. Measured
2026-09-23: 285 NSF awards and 134 NIH projects were active at Northeastern.
"""

from __future__ import annotations

USER_AGENT = (
    "KnowMyProfessorGrants/0.1 (personal research project; "
    "contact: kunapareddy.s@northeastern.edu)"
)
#: NIH asks for no more than one request per second; NSF states no limit, and
#: gets the same.
REQUEST_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 5

#: Awards still running, plus those that ended within this window. "Who has
#: funding for X" is mostly about now, but a grant that ended last year still
#: says what the lab works on.
LOOKBACK_YEARS = 3

NSF_API = "https://api.nsf.gov/services/v1/awards.json"
#: Quoted: the API otherwise matches any awardee containing either word.
NSF_AWARDEE = '"Northeastern University"'
NSF_PAGE_SIZE = 25  # the API's maximum
NSF_FIELDS = (
    "id,title,abstractText,piFirstName,piLastName,pdPIName,coPDPI,"
    "startDate,expDate,estimatedTotalAmt,fundProgramName,dirAbbr,divAbbr"
)
NSF_AWARD_URL = "https://www.nsf.gov/awardsearch/showAward?AWD_ID={id}"

NIH_API = "https://api.reporter.nih.gov/v2/projects/search"
NIH_ORG = "NORTHEASTERN UNIVERSITY"
NIH_PAGE_SIZE = 500  # the API's maximum
NIH_FIELDS = [
    "ApplId", "CoreProjectNum", "ProjectNum", "ProjectTitle", "AbstractText",
    "PrincipalInvestigators", "ProjectStartDate", "ProjectEndDate",
    "AwardAmount", "FiscalYear", "AgencyIcAdmin",
]
NIH_PROJECT_URL = "https://reporter.nih.gov/project-details/{appl_id}"

# --- Claude: one plain summary per award -------------------------------------

CLAUDE_MODEL = "claude-haiku-4-5"
CLAUDE_MAX_TOKENS = 300
ABSTRACT_CHARS = 6_000
#: Summaries are cached here by award, keyed on a hash of title + abstract, so
#: an award shared by three PIs is summarised once and never again unless the
#: text changes. Outside ``grants/`` on purpose: ingest reads every JSON file
#: under a source's prefix as a record.
SUMMARY_CACHE_KEY = "grants_summaries.json"
#: Bump to regenerate every summary (prompt changes).
SCHEMA_VERSION = "v1"

SUMMARY_PROMPT = (
    "Summarise this research grant in one or two plain sentences for someone "
    "looking for a professor who works on its topic: what the project studies, "
    "and how. Leave out budgets, broader impacts, training plans and 'national "
    "interest' boilerplate. Use only what the text says. Reply with the summary "
    "alone.\n\n"
)
