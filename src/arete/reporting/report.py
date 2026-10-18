"""Emit evaluation reports.

Two outputs:

* :func:`write_json_report` — the full machine-readable result.
* :func:`write_html_report` — a self-contained, dependency-free HTML page with
  the aggregate scores and a per-query drill-down table.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Union

from arete.runners.harness import EvalResult

PathLike = Union[str, Path]


def write_json_report(result: EvalResult, path: PathLike) -> Path:
    """Write the full result to a JSON file and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def write_html_report(result: EvalResult, path: PathLike, title: str = "ARETE report") -> Path:
    """Write a standalone HTML report and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    agg = result.aggregate()
    metric_names = result.metric_names()

    agg_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td class='num'>{_fmt(val)}</td></tr>"
        for name, val in agg.items()
    )

    header_cells = "".join(f"<th>{html.escape(n)}</th>" for n in metric_names)
    body_rows = []
    for qr in result.per_query:
        smap = qr.score_map()
        cells = "".join(
            f"<td class='num'>{_fmt(smap[n]) if n in smap else '&ndash;'}</td>"
            for n in metric_names
        )
        body_rows.append(
            "<tr>"
            f"<td class='id'>{html.escape(qr.example.id)}</td>"
            f"<td class='q'>{html.escape(qr.example.query)}</td>"
            f"<td class='a'>{html.escape(qr.trace.answer[:200])}</td>"
            f"{cells}"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 1100px;
          padding: 0 1rem; }}
  h1 {{ font-size: 1.4rem; }}
  .meta {{ color: #777; margin-bottom: 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 14px; }}
  th, td {{ border: 1px solid #8884; padding: 6px 10px; text-align: left; }}
  th {{ background: #8882; position: sticky; top: 0; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .q {{ max-width: 260px; }}
  .a {{ max-width: 320px; color: #666; }}
  .scorecard {{ display: inline-block; margin-right: 1rem; }}
  .wrap {{ overflow-x: auto; }}
</style></head>
<body>
<h1>{html.escape(title)}</h1>
<p class="meta">{len(result.per_query)} examples &middot; {len(metric_names)} metrics</p>

<h2>Aggregate scores</h2>
<table><thead><tr><th>Metric</th><th class="num">Mean</th></tr></thead>
<tbody>{agg_rows}</tbody></table>

<h2>Per-query results</h2>
<div class="wrap"><table><thead><tr>
<th>ID</th><th>Query</th><th>Answer</th>{header_cells}
</tr></thead><tbody>
{''.join(body_rows)}
</tbody></table></div>
</body></html>"""

    path.write_text(doc, encoding="utf-8")
    return path
