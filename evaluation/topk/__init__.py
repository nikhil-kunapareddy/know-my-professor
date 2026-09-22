"""Choosing ``top_k`` by measurement rather than by intuition.

``DEFAULT_TOP_K = 8`` was never measured. This package measures it, once per
namespace, by asking a judge the only question that decides the number: of the
k chunks retrieval returns, how many would a correct answer actually use?

The experiment is cheap because retrieval is deterministic and ranked, so the
chunks at k=3 are a *prefix* of the chunks at k=11. Retrieve once at the
largest k, judge each chunk once, and every smaller k is a slice — no second
query, no second judge call. Five k values cost what one costs.

Read ``judge.py`` for what "relevant" means (the rubric is the experiment),
``experiment.py`` for the prefix trick, and ``report.py`` for the decision rule,
which is fixed before the data is seen on purpose.
"""
