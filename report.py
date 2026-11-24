"""
Relatório em PDF da configuração da medida móvel.

Recebe dados já preparados pela interface (textos formatados + DataFrames para os gráficos)
e devolve os bytes do PDF. Gráficos em matplotlib (sem dependência de navegador).
"""
from __future__ import annotations

import io
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

# ---------------------------------------------------------------- identidade visual
NAVY = colors.HexColor("#1f2a44")
ACCENT = colors.HexColor("#2a78d6")
INK = colors.HexColor("#222222")
MUTED = colors.HexColor("#6b6b66")
RULE = colors.HexColor("#d9dce1")
HEAD_BG = colors.HexColor("#eef1f5")
ZEBRA = colors.HexColor("#f8f9fb")
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#8a6d3b"]

PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm
CONTENT_W = PAGE_W - 2 * MARGIN

ST = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=colors.white),
    "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=10.5, leading=14,
                               textColor=colors.HexColor("#c9d3e6")),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=12.5, leading=16, textColor=NAVY,
                         spaceBefore=14, spaceAfter=6),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9, leading=12.5, textColor=INK),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.5, leading=11, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=INK),
    "head": ParagraphStyle("head", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=NAVY),
    "label": ParagraphStyle("label", fontName="Helvetica", fontSize=7.5, leading=9.5, textColor=MUTED),
    "value": ParagraphStyle("value", fontName="Helvetica-Bold", fontSize=9.5, leading=12, textColor=INK),
    "caption": ParagraphStyle("caption", fontName="Helvetica-Oblique", fontSize=7.8, leading=10,
                              textColor=MUTED, spaceBefore=3, alignment=TA_LEFT),
}


def _p(text, style="cell"):
    return Paragraph(str(text).replace("\n", "<br/>"), ST[style])


# ---------------------------------------------------------------- gráficos
def _mpl_style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9a9a95")
    ax.tick_params(colors="#55554f", labelsize=7.5, length=3)
    ax.grid(axis="y", color="#e6e6e3", linewidth=0.6)
    ax.set_axisbelow(True)


def _fig_to_image(fig, width_cm: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width_cm * cm
    img.drawHeight = width_cm * cm * ratio
    return img


def age_chart(age: dict) -> Image:
    """age = {df: faixa/valor/n[/sex], stat_label, by_sex, geral}"""
    df = age["df"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 3.6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    groups = list(df.groupby("sex", dropna=False)) if age["by_sex"] else [("todos", df)]
    total = df["n"].sum()
    groups = [(g, sub) for g, sub in groups if sub["n"].sum() >= 0.005 * total]  # omite grupos residuais (<0,5%)
    n_g = len(groups)
    width = age.get("width", 1) * 0.8 / max(n_g, 1)
    for i, (g, sub) in enumerate(groups):
        col = CAT[i % len(CAT)]
        ax1.plot(sub["faixa"], sub["valor"], color=col, lw=1.4, marker="o", ms=2.6, label=str(g))
        ax2.bar(sub["faixa"] + (i - (n_g - 1) / 2) * width, sub["n"], width=width, color=col, label=str(g))
    ax1.axhline(age["geral"], color="#8a8984", lw=0.9, ls=":", label=f"{age['stat_label'].lower()} geral")
    ax1.set_ylabel(f"{age['stat_label']} do resultado", fontsize=8)
    ax2.set_ylabel("nº de resultados", fontsize=8)
    ax2.set_xlabel("idade (anos)", fontsize=8)
    for ax in (ax1, ax2):
        _mpl_style(ax)
    ax1.legend(fontsize=7, frameon=False, ncol=min(n_g + 1, 5), loc="upper left", bbox_to_anchor=(0, 1.18))
    fig.tight_layout(h_pad=0.6)
    return _fig_to_image(fig, CONTENT_W / cm)


def _plot_items(ax, itens, x_is_time, annotate):
    multi = len(itens) > 1 or annotate is False
    for i, it in enumerate(itens):
        col = it.get("cor") or (CAT[i % len(CAT)] if multi else CAT[0])
        x = pd.to_datetime(it["x"]) if x_is_time else np.asarray(it["x"])
        y = np.asarray(it["y"], float)
        ax.plot(x, y, color=col, lw=0.9, label=it["nome"])
        al = np.asarray(it["alarme"], bool)
        if al.any():
            ax.scatter(np.asarray(x)[al], y[al], s=10, color=col if multi else "#e34948", zorder=3,
                       edgecolors="white", linewidths=0.5)
        lim = it["lim"]
        lim_col = col if multi else "#e34948"
        x0, x1 = (x.min(), x.max())
        for key, ls, c_ in (("LS", "--", lim_col), ("LI", "--", lim_col), ("alvo", ":", col if multi else "#3d3d3a")):
            ax.hlines(lim[key], x0, x1, colors=c_, linestyles=ls, lw=0.8)
        if annotate and not multi:
            for key in ("LS", "alvo", "LI"):
                ax.annotate(f"{key} {lim[key]:.4g}".replace(".", ","), xy=(1.0, lim[key]),
                            xycoords=("axes fraction", "data"), xytext=(3, 0), textcoords="offset points",
                            va="center", fontsize=7, color="#55554f")
    if x_is_time:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
    _mpl_style(ax)


def series_chart(serie: dict) -> Image:
    """serie = {itens: [...]} ou {paineis: [{titulo, itens}]}; item = {nome, cor?, x, y, lim{alvo,LI,LS}, alarme}"""
    paineis = serie.get("paineis")
    xlabel = "data do último resultado do bloco" if serie["x_is_time"] else "nº do resultado"
    if paineis:
        fig, axes = plt.subplots(1, len(paineis), figsize=(7.2, 3.2), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, pn in zip(axes, paineis):
            _plot_items(ax, pn["itens"], serie["x_is_time"], annotate=False)
            ax.set_title(pn["titulo"], fontsize=8.5, color="#3d3d3a")
            ax.set_xlabel(xlabel, fontsize=7.5)
        axes[0].set_ylabel(serie.get("ylabel", "medida móvel"), fontsize=8)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=7, frameon=False, ncol=min(len(labels), 4), loc="upper left",
                   bbox_to_anchor=(0.06, 1.06))
    else:
        fig, ax = plt.subplots(figsize=(7.2, 2.8))
        itens = serie["itens"]
        _plot_items(ax, itens, serie["x_is_time"], annotate=True)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(serie.get("ylabel", "medida móvel"), fontsize=8)
        if len(itens) > 1:
            ax.legend(fontsize=7, frameon=False, ncol=min(len(itens), 4), loc="upper left", bbox_to_anchor=(0, 1.14))
    fig.tight_layout()
    return _fig_to_image(fig, CONTENT_W / cm)


# ---------------------------------------------------------------- tabelas
def kv_grid(pairs: list[tuple[str, str]], cols: int = 2) -> Table:
    """Grade de identificação: rótulo pequeno em cima, valor em negrito embaixo."""
    cells = [[_p(lbl, "label"), _p(val or "—", "value")] for lbl, val in pairs]
    rows = []
    for i in range(0, len(cells), cols):
        chunk = cells[i:i + cols] + [[_p(""), _p("")]] * (cols - len(cells[i:i + cols]))
        rows.append([Table([[c[0]], [c[1]]], style=[("LEFTPADDING", (0, 0), (-1, -1), 0),
                                                       ("TOPPADDING", (0, 0), (-1, -1), 0),
                                                       ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]) for c in chunk])
    t = Table(rows, colWidths=[CONTENT_W / cols] * cols)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def data_table(header: list[str], rows: list[list], widths: list[float], bold_first: bool = True) -> Table:
    data = [[_p(h, "head") for h in header]]
    for r in rows:
        data.append([_p(v, "cellb" if (j == 0 and bold_first) else "cell") for j, v in enumerate(r)])
    t = Table(data, colWidths=[w * CONTENT_W for w in widths], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    for i in range(2, len(data), 2):
        style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(style))
    return t


def signature_block(roles: list[str], names: dict[str, str]) -> Table:
    col_w = CONTENT_W / 3  # largura fixa: um campo só não ocupa a página inteira
    cells = []
    for r in roles:
        nm = names.get(r) or ""
        cells.append(Table([
            [Spacer(1, 9.5)],  # mesma altura do antigo rótulo ("ELABORADO POR"), mantendo a distância até a linha
            [Spacer(1, 16)],
            [_p(nm or "&nbsp;", "cell")],
            [_p("Assinatura / data", "label")],
        ], colWidths=[col_w - 12], style=[
            ("LINEABOVE", (0, 2), (0, 2), 0.6, INK),
            ("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 1),
        ]))
    t = Table([cells], colWidths=[col_w] * len(roles), hAlign="LEFT")
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    return t


# ---------------------------------------------------------------- documento
def _new_doc(buf, info: dict) -> BaseDocTemplate:
    """Documento A4 com faixa de título na 1ª página, cabeçalho nas demais e rodapé com data e página."""
    emitted: datetime = info.get("emitido") or datetime.now()
    header_right = " · ".join(v for v in [info.get("analito"), info.get("equip_curto")] if v)

    def on_page(canv, doc):
        canv.saveState()
        if doc.page > 1:
            canv.setFont("Helvetica-Bold", 8)
            canv.setFillColor(NAVY)
            canv.drawString(MARGIN, PAGE_H - 1.15 * cm, info["titulo"])
            canv.setFont("Helvetica", 8)
            canv.setFillColor(MUTED)
            canv.drawRightString(PAGE_W - MARGIN, PAGE_H - 1.15 * cm, header_right)
            canv.setStrokeColor(ACCENT)
            canv.setLineWidth(1.2)
            canv.line(MARGIN, PAGE_H - 1.3 * cm, PAGE_W - MARGIN, PAGE_H - 1.3 * cm)
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.5)
        canv.line(MARGIN, 1.35 * cm, PAGE_W - MARGIN, 1.35 * cm)
        canv.setFont("Helvetica", 7.5)
        canv.setFillColor(MUTED)
        canv.drawString(MARGIN, 0.95 * cm,
                        f"Emitido em {emitted:%d/%m/%Y %H:%M} · {info.get('rodape', 'PBRTQC Planner')}")
        canv.drawRightString(PAGE_W - MARGIN, 0.95 * cm, f"Página {doc.page}")
        canv.restoreState()

    def on_first(canv, doc):
        # faixa de título
        band_h = 3.6 * cm
        canv.saveState()
        canv.setFillColor(NAVY)
        canv.rect(0, PAGE_H - band_h, PAGE_W, band_h, stroke=0, fill=1)
        canv.setFillColor(ACCENT)
        canv.rect(0, PAGE_H - band_h - 0.12 * cm, PAGE_W, 0.12 * cm, stroke=0, fill=1)
        canv.restoreState()
        on_page(canv, doc)

    doc = BaseDocTemplate(buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=1.9 * cm,
                          bottomMargin=1.8 * cm, title=info["titulo"], author=info.get("elaborado", ""),
                          subject=info.get("analito", ""))
    frame_first = Frame(MARGIN, 1.8 * cm, CONTENT_W, PAGE_H - 1.8 * cm - 0.9 * cm, id="first",
                        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    frame_rest = Frame(MARGIN, 1.8 * cm, CONTENT_W, PAGE_H - 1.8 * cm - 1.9 * cm, id="rest",
                       leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate("first", [frame_first], onPage=on_first),
                          PageTemplate("rest", [frame_rest], onPage=on_page)])

    return doc


def _start(info: dict):
    """Elementos iniciais (título e subtítulo na faixa) e numerador de seções."""
    from reportlab.platypus import NextPageTemplate
    el = [NextPageTemplate("rest"), Paragraph(info["titulo"], ST["title"]), Spacer(1, 4),
          Paragraph(info.get("subtitulo", ""), ST["subtitle"]), Spacer(1, 1.15 * cm)]
    counter = {"n": 0}

    def h1(text):
        counter["n"] += 1
        return Paragraph(f'<font color="#2a78d6">{counter["n"]}.</font>&nbsp;&nbsp;{text}', ST["h1"])
    return el, h1


def _closing(info: dict, h1) -> KeepTogether:
    obs = (info.get("observacoes") or "").strip()
    closing = [h1("Observações e responsável" if obs else "Responsável")]
    if obs:
        closing += [Paragraph(obs.replace("\n", "<br/>"), ST["body"]), Spacer(1, 14)]
    closing += [Spacer(1, 4), signature_block(["Elaborado por"], {"Elaborado por": info.get("elaborado", "")})]
    return KeepTogether(closing)


def build_pdf(info: dict) -> bytes:
    buf = io.BytesIO()
    doc = _new_doc(buf, info)
    el, h1 = _start(info)

    # 1. Identificação
    el += [h1("Identificação do estudo"), kv_grid(info["identificacao"], cols=3)]

    # 2. Preparo dos dados
    el.append(KeepTogether([h1("Preparo dos dados"),
                            data_table(["Item", "Definição"], [[a, b] for a, b in info["preparo"]], [0.32, 0.68])]))
    if info.get("idade"):
        fig_block = [Spacer(1, 8), age_chart(info["idade"]),
                     Paragraph(info["idade"].get("legenda", ""), ST["caption"])]
        el.append(KeepTogether(fig_block))

    # 3. Configuração
    el.append(KeepTogether([h1("Configuração da medida móvel"),
                            data_table(["Parâmetro", "Valor", "Fonte / critério"], info["config"], [0.28, 0.2, 0.52])]))

    # 4. Limites de controle
    block = [h1("Valor-alvo e limites de controle"),
             data_table(["Parâmetro", "Valor", "Fonte / critério"], info["limites"], [0.28, 0.2, 0.52])]
    el.append(KeepTogether(block))
    if info.get("limites_equip") is not None:
        df = info["limites_equip"]
        el.append(KeepTogether([Spacer(1, 8), Paragraph("Valores por estrato", ST["cellb"]), Spacer(1, 3),
                                data_table(list(df.columns), df.values.tolist(),
                                           [0.30] + [0.70 / (len(df.columns) - 1)] * (len(df.columns) - 1))]))
    if info.get("fontes_estrato") is not None:
        df = info["fontes_estrato"]
        el.append(KeepTogether([Spacer(1, 8), Paragraph("Fontes do alvo e do DP por estrato", ST["cellb"]), Spacer(1, 3),
                                data_table(list(df.columns), df.values.tolist(), [0.30, 0.35, 0.35])]))

    # 5. Gráfico da medida móvel
    if info.get("serie"):
        el.append(KeepTogether([h1("Medida móvel simulada no período"), series_chart(info["serie"]),
                                Paragraph(info["serie"].get("legenda", ""), ST["caption"])]))

    # Observações e assinatura
    el.append(_closing(info, h1))
    doc.build(el)
    return buf.getvalue()


def build_review_pdf(info: dict) -> bytes:
    """Relatório da revisão de uma média móvel cadastrada (resumo ponderado do período)."""
    buf = io.BytesIO()
    doc = _new_doc(buf, info)
    el, h1 = _start(info)

    el += [h1("Identificação da revisão"), kv_grid(info["identificacao"], cols=3)]

    if info.get("cadastro"):
        el.append(KeepTogether([h1("Parâmetros cadastrados no middleware"),
                                data_table(["Parâmetro", "Valor", "Observação"], info["cadastro"], [0.28, 0.27, 0.45])]))

    el.append(KeepTogether([h1("Resumo ponderado do período"),
                            data_table(["Medida", "Valor", "Cálculo"], info["resumo"], [0.32, 0.16, 0.52]),
                            Paragraph(info.get("nota_resumo", ""), ST["caption"])]))

    df = info["plataformas"]
    el.append(KeepTogether([h1("Por plataforma"),
                            data_table(list(df.columns), df.values.tolist(),
                                       [0.15, 0.13, 0.08, 0.14, 0.13, 0.185, 0.185])]))

    df = info["instrumentos"]
    widths = [0.14, 0.44, 0.10, 0.16, 0.16] if len(df.columns) == 5 else [0.56, 0.12, 0.16, 0.16]
    el += [h1("Por instrumento"), data_table(list(df.columns), df.values.tolist(), widths)]

    el.append(_closing(info, h1))
    doc.build(el)
    return buf.getvalue()
