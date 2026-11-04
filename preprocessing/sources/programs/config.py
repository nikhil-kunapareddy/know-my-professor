"""Configuration owned by the programs source.

Programs live on the same CourseLeaf catalog as courses, so the host, the
politeness delay and the user agent are the courses source's, not copies.
robots.txt (checked 2026-09-29) disallows only admin, search and asset paths;
every program page is allowed, and the sitemap is published for crawlers.
"""

from __future__ import annotations

from ..courses.config import CATALOG_BASE

SITEMAP_URL = f"{CATALOG_BASE}/sitemap.xml"

#: Catalog trees that hold programs. The rest of the sitemap is the course
#: descriptions (the courses source) and handbook/policy pages.
PROGRAM_ROOTS = ("undergraduate", "graduate", "professional-studies")

#: A program page is at least ``{root}/{college}/{program}/``; anything shallower
#: is a college or catalog index.
MIN_PATH_SEGMENTS = 3

#: Pages under these segments are policy text, never programs; skipping them
#: saves ~100 requests a run without changing what is stored.
SKIP_SEGMENTS = frozenset({"academic-policies-procedures"})

#: CourseLeaf names each tab's container ``{tab}textcontainer``. The overview is
#: ``textcontainer``; the requirements tab varies by program type --
#: ``programrequirements``, ``minorrequirements``, ``advancedentryphdprogramrequirements``
#: -- but always ends the same way, and its presence is what makes a page a program.
OVERVIEW_TAB = "textcontainer"
REQUIREMENTS_TAB_SUFFIX = "requirementstextcontainer"

#: Prefix on every program's entity id. Programs share the ``courses`` namespace
#: with ``course-*`` ids, so the prefix is what keeps the two id spaces apart.
ENTITY_PREFIX = "program-"

LEVEL_LABELS = {
    "undergraduate": "Undergraduate program",
    "graduate": "Graduate program",
    "professional-studies": "College of Professional Studies program",
}
