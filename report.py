"""Write an HTML report with interactive Plotly charts."""
import html
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

import fakeprint as fpm

COLORS = {"real": "#2563eb", "ai": "#dc2626", "ai-shifted": "#ea580c", "ai-unseen": "#9333ea",
          "suno": "#dc2626", "udio": "#0d9488"}
LABELS = {"real": "Real", "ai": "AI", "ai-shifted": "AI, pitch +1 semitone", "ai-unseen": "AI (unseen decoder)",
          "suno": "Suno", "udio": "Udio"}


def _groups(items):
    out = []
    for it in items:
        if it["group"] not in out:
            out.append(it["group"])
    return out


def _style(fig, title, xtitle, ytitle, height=380):
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15)),
        template="plotly_white", height=height,
        margin=dict(l=60, r=20, t=60, b=50),
        legend=dict(orientation="h", y=1.0, x=1, xanchor="right", yanchor="bottom"),
        xaxis_title=xtitle, yaxis_title=ytitle,
    )
    return fig


def fig_spectrum(items):
    fig = go.Figure()
    for g in _groups(items):
        members = [it for it in items if it["group"] == g]
        f = members[0]["spec_freqs"]
        mean = np.mean([it["spec_db"] for it in members], axis=0)
        fig.add_trace(go.Scatter(x=f[::2], y=mean[::2], name=LABELS.get(g, g),
                                 line=dict(color=COLORS.get(g), width=1)))
    band = items[0]["freqs"]
    fig.add_vrect(x0=float(band[0]), x1=float(band[-1]), fillcolor="#f59e0b", opacity=0.10, line_width=0,
                  annotation_text="band used for detection", annotation_position="top left")
    return _style(fig, "1 · Average spectrum (mean over clips)", "Frequency (Hz)", "Power (dB)")


def fig_mean_fakeprint(items):
    fig = go.Figure()
    for g in _groups(items):
        members = [it for it in items if it["group"] == g]
        fig.add_trace(go.Scatter(x=members[0]["freqs"], y=np.mean([it["fp"] for it in members], axis=0),
                                 name=LABELS.get(g, g), line=dict(color=COLORS.get(g), width=1)))
    return _style(fig, "2 · Fakeprint: spectrum minus its lower envelope (mean over clips)",
                  "Frequency (Hz)", "Normalised peak height")


def fig_zoom(items, threshold, lo=None, hi=None):
    """One example clip per group, stacked, with the detected comb marked."""
    if lo is None or hi is None:  # 1 kHz window in the middle of the band
        band = items[0]["freqs"]
        mid = round((band[0] + band[-1]) / 2000) * 1000
        lo, hi = mid - 500.0, mid + 500.0
    fig = go.Figure()
    for row, g in enumerate(_groups(items)):
        members = [it for it in items if it["group"] == g]
        ex = max(members, key=lambda it: it["comb_score"]) if g != "real" else members[0]
        sel = (ex["freqs"] >= lo) & (ex["freqs"] <= hi)
        offset = 1.2 * row
        fig.add_trace(go.Scatter(x=ex["freqs"][sel], y=ex["fp"][sel] + offset,
                                 name=f"{LABELS.get(g, g)}: {ex['name']}",
                                 line=dict(color=COLORS.get(g), width=1.2)))
        if ex["comb_score"] >= threshold and np.isfinite(ex["spacing_hz"]):
            s = ex["spacing_hz"]
            for m in np.arange(np.ceil(lo / s), np.floor(hi / s) + 1) * s:
                fig.add_shape(type="line", x0=m, x1=m, y0=offset, y1=offset + 1.05,
                              line=dict(color=COLORS.get(g), width=1, dash="dot"), opacity=0.5)
    fig.update_yaxes(showticklabels=False)
    return _style(fig, f"3 · Zoom {lo/1000:.0f}–{hi/1000:.0f} kHz, one clip per group "
                       "(dotted lines = multiples of the detected spacing)",
                  "Frequency (Hz)", "", height=420)


def fig_comb(items, threshold):
    fig = go.Figure()
    for g in _groups(items):
        members = [it for it in items if it["group"] == g]
        fig.add_trace(go.Box(y=[it["comb_score"] for it in members], name=LABELS.get(g, g),
                             text=[it["name"] for it in members], boxpoints="all", jitter=0.5,
                             pointpos=0, marker=dict(color=COLORS.get(g), size=7),
                             line=dict(color=COLORS.get(g)), fillcolor="rgba(0,0,0,0)"))
    fig.add_hline(y=threshold, line=dict(color="#6b7280", dash="dash"),
                  annotation_text=f"threshold {threshold}", annotation_position="top right")
    fig.update_layout(showlegend=False)
    return _style(fig, "4 · Comb score per clip (how regularly the peaks repeat)", "", "Comb score (0–1)")


def fig_coef(freqs, coef):
    fig = go.Figure(go.Scatter(x=freqs, y=coef, line=dict(color="#111827", width=1), name="weight"))
    fig.add_hline(y=0, line=dict(color="#9ca3af", width=1))
    return _style(fig, "5 · What the logistic regression learned (positive = evidence of AI)",
                  "Frequency (Hz)", "Weight")


def _table(items, threshold):
    rows = []
    for it in items:
        p = it.get("prob")
        truth_ai = it["group"] != "real"
        verdict = "" if p is None else ("AI" if p >= 0.5 else "real")
        ok = "" if p is None else ("✓" if (p >= 0.5) == truth_ai else "✗")
        spacing = f"{it['spacing_hz']:.1f} Hz" if it["comb_score"] >= threshold else "–"
        rows.append(
            f"<tr><td>{html.escape(it['name'])}</td>"
            f"<td><span class='tag' style='background:{COLORS.get(it['group'], '#6b7280')}'>"
            f"{LABELS.get(it['group'], it['group'])}</span></td>"
            f"<td>{'' if p is None else f'{p:.3f}'}</td><td>{verdict}</td><td class='ok'>{ok}</td>"
            f"<td>{it['comb_score']:.3f}</td><td>{spacing}</td></tr>"
        )
    return ("<table><thead><tr><th>Clip</th><th>True label</th><th>P(AI)</th><th>Model says</th>"
            "<th></th><th>Comb score</th><th>Peak spacing</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def write_report(path, title, intro_html, items, cards, threshold, coef=None, notes=()):
    figs = [fig_spectrum(items), fig_mean_fakeprint(items), fig_zoom(items, threshold), fig_comb(items, threshold)]
    if coef is not None:
        figs.append(fig_coef(items[0]["freqs"], coef))
    charts = "".join(
        # Embed plotly.js once (about 4.5 MB) so the report works offline and can be emailed.
        f"<div class='chart'>{f.to_html(full_html=False, include_plotlyjs=(i == 0))}</div>"
        for i, f in enumerate(figs)
    )
    card_html = "".join(
        f"<div class='card'><div class='v'>{html.escape(v)}</div><div class='l'>{html.escape(l)}</div>"
        f"<div class='s'>{html.escape(s)}</div></div>" for l, v, s in cards
    )
    notes_html = "".join(f"<li>{n}</li>" for n in notes)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
 body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }}
 main {{ max-width: 1100px; margin: 0 auto; padding: 28px 16px 60px; }}
 h1 {{ font-size: 26px; margin: 0 0 6px; }}
 .intro {{ color: #334155; max-width: 820px; line-height: 1.55; }}
 .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 22px 0; }}
 .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px; }}
 .card .v {{ font-size: 26px; font-weight: 650; }}
 .card .l {{ font-size: 13px; font-weight: 600; margin-top: 2px; }}
 .card .s {{ font-size: 12px; color: #64748b; margin-top: 4px; }}
 .chart {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 8px; margin: 14px 0; }}
 table {{ width: 100%; border-collapse: collapse; background: #fff; font-size: 13px; border: 1px solid #e2e8f0; }}
 th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid #f1f5f9; }}
 th {{ background: #f1f5f9; font-weight: 600; }}
 .tag {{ color: #fff; border-radius: 4px; padding: 1px 7px; font-size: 12px; }}
 .ok {{ font-weight: 700; }}
 .notes {{ background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px; padding: 12px 18px 12px 34px; color: #78350f; line-height: 1.5; }}
 h2 {{ font-size: 18px; margin: 30px 0 8px; }}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<div class="intro">{intro_html}</div>
<div class="cards">{card_html}</div>
{f'<ul class="notes">{notes_html}</ul>' if notes_html else ''}
{charts}
<h2>Per-clip results</h2>
{_table(items, threshold)}
</main></body></html>"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(page, encoding="utf-8")
    return Path(path)
