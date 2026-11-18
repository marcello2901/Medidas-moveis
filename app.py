"""
PBRTQC Planner: planejamento e simulação da medida móvel de pacientes
Etapa 1: estatística descritiva · Etapa 2: preparo dos dados · Etapa 3: configuração · Etapa 4: simulação de bias

Executar:  streamlit run app.py
"""
from __future__ import annotations

import io
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import streamlit as st
from plotly.subplots import make_subplots

import mm_core as mm
import report

st.set_page_config(page_title="PBRTQC Planner", layout="wide")

# Paleta (categórica em ordem fixa + cores de papel)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CAT = [C1, C2, C3, C4, "#e87ba4", "#008300", "#6250d6", "#8a6d3b"]  # ordem fixa, nunca reciclada
SEX_COLORS = [C1, C2, C3, C4]
INK, MUTED, ALARM = "#3d3d3a", "#8a8984", "#e34948"
NONE = "— nenhuma —"
ROW_ORDER = "— ordem das linhas da planilha —"


def base_layout(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    kw.setdefault("hovermode", "x unified")
    fig.update_layout(height=height, template="plotly_white", margin=dict(l=10, r=70, t=90, b=10),
                      title=dict(text=kw.pop("title", None), y=0.98, yanchor="top"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), **kw)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#ececea")
    return fig


def guess(cols: list[str], prefs: list[str], fallback: int = 0) -> int:
    low = [c.lower().strip() for c in cols]
    for p in prefs:  # correspondência exata primeiro
        if p in low:
            return low.index(p)
    for p in prefs:
        for i, c in enumerate(low):
            if p in c:
                return i
    return fallback


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_data(show_spinner=False)
def cached_read(b: bytes, name: str, sheet: str | None):
    return mm.read_table(b, name, sheet)


@st.cache_data(show_spinner=False)
def cached_sheets(b: bytes, name: str):
    return mm.list_sheets(b, name)


# =============================================================================
# Termos de uso e proteção de dados (aceite obrigatório antes do upload)
# =============================================================================
if not st.session_state.get("termos_aceitos", False):
    st.title("Termos de uso e proteção de dados")
    st.markdown(
        "Esta ferramenta processa planilhas de resultados laboratoriais para planejar e simular a medida móvel de "
        "pacientes. Os arquivos enviados podem conter **dados pessoais sensíveis** (como nome completo, data de "
        "nascimento, números de documentos de identificação e informações de saúde), cujo tratamento é regulado por "
        "leis de proteção de dados como a **Lei Geral de Proteção de Dados (LGPD – Lei nº 13.709/2018)** e o "
        "**Regulamento Geral sobre a Proteção de Dados (GDPR)**.\n\n"
        "É de sua **exclusiva responsabilidade** garantir que todos os dados utilizados nesta ferramenta estejam em "
        "conformidade com a legislação de proteção de dados aplicável. Recomendamos fortemente que você utilize "
        "**apenas dados previamente anonimizados**, para proteger a privacidade dos titulares. A ferramenta precisa "
        "somente de resultado, data/hora, idade, sexo e equipamento: remova nomes, documentos, códigos de amostra ou "
        "de atendimento e qualquer outro identificador antes do envio.\n\n"
        "A responsabilidade pela natureza dos dados tratados é exclusivamente sua.\n\n"
        "Para prosseguir, você deve confirmar que os dados a serem utilizados foram devidamente tratados e anonimizados."
    )
    aceite = st.checkbox("Ao marcar esta caixa, confirmo que os dados fornecidos estão anonimizados.")
    if st.button("Continuar", type="primary", disabled=not aceite):
        st.session_state["termos_aceitos"] = True
        st.rerun()
    st.stop()

# =============================================================================
# Barra lateral — upload e mapeamento de colunas
# =============================================================================
st.sidebar.title("Dados de entrada")
up = st.sidebar.file_uploader("Planilha (xlsx, xls ou csv)", type=["xlsx", "xlsm", "xls", "csv", "txt"])

st.title("PBRTQC Planner")
st.caption("Planejamento e simulação da medida móvel de pacientes (patient-based real-time quality control)")
if up is None:
    st.info("Envie a planilha na barra lateral. Ela precisa ter, no mínimo, uma coluna de **resultado** e "
            "uma de **data/hora** (a ordem cronológica é essencial para simular a medida móvel). "
            "Idade, sexo, equipamento e teste são opcionais.")
    st.stop()

fbytes = up.getvalue()
sheets = cached_sheets(fbytes, up.name)
sheet = st.sidebar.selectbox("Aba", sheets) if sheets else None
raw = cached_read(fbytes, up.name, sheet)
cols = list(raw.columns)

st.sidebar.subheader("Colunas")
c_res = st.sidebar.selectbox("Resultado *", cols, index=guess(cols, ["resultado", "resultado numerico", "result", "valor", "value"]))
date_opts = [ROW_ORDER] + cols
c_date = st.sidebar.selectbox("Data (ou data e hora) *", date_opts,
                              index=guess(date_opts, ["data,res", "data res", "data resultado", "data", "date", "datetime"]))
time_opts = [NONE] + cols
c_time = st.sidebar.selectbox("Hora (se estiver em coluna separada)", time_opts,
                              index=guess(time_opts, ["hora,res", "hora res", "hora resultado", "hora", "time"]))
c_age = st.sidebar.selectbox("Idade", [NONE] + cols, index=guess([NONE] + cols, ["idade", "age"]))
c_sex = st.sidebar.selectbox("Sexo", [NONE] + cols, index=guess([NONE] + cols, ["sexo", "gender", "sex", "genero"]))
c_ref = st.sidebar.selectbox(
    "Intervalo de referência", [NONE] + cols,
    index=guess([NONE] + cols, ["ref. ranges", "ref, ranges", "ref ranges", "reference range", "valor de referência",
                                "valores de referência", "intervalo de referência", "vr", "ref."]),
    help="Texto como '2.52 - 39.20', '3,50 - 5,10', '< 5,0' ou '> 10'. Usado como sugestão de truncamento.")

st.sidebar.subheader("Filtros")
c_test = st.sidebar.selectbox("Coluna do teste/analito", [NONE] + cols, index=guess([NONE] + cols, ["teste", "exame", "analito", "test"]))
c_inst = st.sidebar.selectbox("Coluna do equipamento", [NONE] + cols, index=guess([NONE] + cols, ["instrument", "equipamento", "analisador", "instrumento"]))

df = raw.copy()
sel_i: list[str] = []
sel_t = None
if c_test != NONE:
    tests = sorted(df[c_test].dropna().astype(str).unique())
    sel_t = st.sidebar.selectbox("Teste", tests) if tests else None
    if sel_t:
        df = df[df[c_test].astype(str) == sel_t]
if c_inst != NONE:
    insts = sorted(df[c_inst].dropna().astype(str).unique())
    sel_i = st.sidebar.multiselect("Equipamento(s)", insts, default=insts[:1])
    df = df[df[c_inst].astype(str).isin(sel_i)]
    if len(sel_i) > 1:
        st.sidebar.warning("Vários equipamentos viram um único fluxo em ordem cronológica. Na rotina a medida móvel "
                           "é avaliada por equipamento: um erro em só um deles fica diluído aqui. Para ver cada um, "
                           "marque \"Ver por equipamento\" em Limites de controle.")

if df.empty:
    st.error("Nenhuma linha após os filtros.")
    st.stop()

cmap = mm.ColumnMap(
    result=c_res,
    date=None if c_date == ROW_ORDER else c_date,
    time=None if c_time == NONE or c_date == ROW_ORDER else c_time,
    age=None if c_age == NONE else c_age,
    sex=None if c_sex == NONE else c_sex,
    group=None if c_inst == NONE else c_inst,
    ref=None if c_ref == NONE else c_ref,
)
data, rep = mm.prepare_dataset(df, cmap)
has_time = cmap.date is not None
has_age = cmap.age is not None and data["age"].notna().any()
has_sex = cmap.sex is not None

if len(data) < 30:
    st.error(f"Só {len(data)} resultados numéricos válidos: insuficiente.")
    st.stop()

with st.sidebar.expander("Resumo da leitura", expanded=False):
    st.write({k.replace("_", " "): v for k, v in rep.items()})
    if has_time:
        st.write(f"Período: {data['t'].min():%d/%m/%Y %H:%M} → {data['t'].max():%d/%m/%Y %H:%M}")
if not has_time:
    st.warning("Sem coluna de data: a ordem das linhas da planilha será tomada como ordem cronológica.")

tab0, tab1, tab2, tab3 = st.tabs(["1 · Estatística descritiva", "2 · Preparo dos dados",
                                  "3 · Configuração da ferramenta", "4 · Simulação de bias"])

# =============================================================================
# Etapa 1 — estatística descritiva
# =============================================================================
INT_MEASURES = {"n", "valores distintos", "n no valor mínimo", "n no valor máximo"}


def fmt_stat(medida: str, v: float) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    if medida in INT_MEASURES:
        return fmt(v, 0)
    return fmt(v, 4) if abs(v) < 1000 else fmt(v, 2)


with tab0:
    xd = data["val"].to_numpy()
    st.caption("Todos os resultados numéricos do teste/equipamento(s) selecionados na barra lateral, antes de "
               "qualquer exclusão de outliers, truncamento ou filtro de sexo/idade.")
    top = st.columns(6)
    top[0].metric("Resultados válidos", fmt(len(xd), 0))
    top[1].metric("Linhas descartadas", fmt(rep["linhas_lidas"] - rep["linhas_validas"], 0),
                  help="Sem resultado numérico" + (" ou sem data/hora" if has_time else ""))
    if has_time:  # período em dois cartões (um só cartão truncava o texto)
        top[2].metric("Início do período", f"{data['t'].min():%d/%m/%y}", help=f"{data['t'].min():%d/%m/%Y %H:%M}")
        top[3].metric("Fim do período", f"{data['t'].max():%d/%m/%y}", help=f"{data['t'].max():%d/%m/%Y %H:%M}")
        top[4].metric("Dias com resultado", fmt(data["t"].dt.normalize().nunique(), 0))
        top[5].metric("Resultados / dia (média)", fmt(len(xd) / max(data["t"].dt.normalize().nunique(), 1), 1))

    dfull = mm.describe_full(xd)
    left, right = st.columns([1, 1.4])
    with left:
        st.subheader("Medidas")
        show = dfull.assign(valor=[fmt_stat(m_, v_) for m_, v_ in zip(dfull["medida"], dfull["valor"])])
        st.dataframe(show, hide_index=True, width="stretch", height=640)
        sk = dfull.set_index("medida")["valor"]
        msgs = []
        if sk.get("n no valor mínimo", 0) > 0.01 * len(xd):
            msgs.append(f"{fmt(sk['n no valor mínimo'], 0)} resultados ({100 * sk['n no valor mínimo'] / len(xd):.1f}%) "
                        "empilhados no valor mínimo: provável limite de quantificação/medição (resultados censurados)")
        for m_ in msgs:
            st.info(m_)
    with right:
        st.subheader("Distribuição")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.03)
        fig.add_trace(go.Histogram(x=xd, nbinsx=80, marker_color=C1, name="resultados",
                                   hovertemplate="%{x}: %{y} resultados<extra></extra>"), 1, 1)
        fig.add_trace(go.Box(x=xd, marker_color=C1, name="", boxpoints="outliers", marker_size=3,
                             hoverinfo="x"), 2, 1)
        fig.update_yaxes(title="nº de resultados", row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=2, col=1)
        st.plotly_chart(base_layout(fig, 360, showlegend=False, bargap=0.05, hovermode="closest"), width="stretch")
        # Q-Q normal
        osm, osr = stats.probplot(xd, dist="norm", fit=False)
        sl, ic, _ = stats.probplot(xd, dist="norm")[1]
        idx = np.unique(np.linspace(0, len(osm) - 1, min(len(osm), 3000)).astype(int))  # amostra para desenhar
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=osm[idx], y=osr[idx], mode="markers", marker=dict(size=4, color=C1),
                                   name="resultados", hovertemplate="quantil teórico %{x:.2f}: %{y}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[osm.min(), osm.max()], y=[ic + sl * osm.min(), ic + sl * osm.max()],
                                 mode="lines", line=dict(color=INK, width=1, dash="dash"), name="normal de referência",
                                 hoverinfo="skip"))
        fig.update_xaxes(title="quantil teórico da normal")
        fig.update_yaxes(title="resultado")
        st.plotly_chart(base_layout(fig, 320, title="Q-Q normal (pontos fora da reta = desvio da normalidade)",
                                    showlegend=False, hovermode="closest"), width="stretch")

    st.subheader("Por grupo")
    groups = []
    if has_sex:
        groups.append(("Sexo", "sex", data))
    if cmap.group and data["group"].nunique() > 0:
        groups.append(("Equipamento", "group", data))
    if has_age:
        dage = data.assign(faixa=pd.cut(data["age"], [0, 1, 12, 18, 30, 40, 50, 60, 70, 80, 200], right=False,
                                        labels=["<1", "1–11", "12–17", "18–29", "30–39", "40–49", "50–59",
                                                "60–69", "70–79", "80+"]))
        groups.append(("Faixa etária (anos)", "faixa", dage))
    if has_time:
        dias = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]
        dwd = data.assign(dia=pd.Categorical(data["t"].dt.dayofweek.map(dict(enumerate(dias))), dias, ordered=True))
        groups.append(("Dia da semana", "dia", dwd))
        dh = data.assign(turno=pd.cut(data["t"].dt.hour, [0, 6, 12, 18, 24], right=False,
                                      labels=["00–05h", "06–11h", "12–17h", "18–23h"]))
        groups.append(("Turno", "turno", dh))
    grp_tables = {}
    if groups:
        sub = st.tabs([g_[0] for g_ in groups])
        for t_, (lab, colname, dsrc) in zip(sub, groups):
            gt = mm.group_describe(dsrc, colname).rename(columns={colname: lab})
            grp_tables[lab] = gt
            t_.dataframe(gt.round(3), hide_index=True, width="stretch")
        st.caption("Grupos com média diferente da geral e muitos resultados são os que mais mexem na medida móvel "
                   "quando sua proporção muda ao longo do dia ou da semana.")
    else:
        st.info("Informe sexo, idade, equipamento ou data/hora para ver os resumos por grupo.")

    buf_d = io.BytesIO()
    with pd.ExcelWriter(buf_d, engine="openpyxl") as xw:
        dfull.to_excel(xw, sheet_name="Descritiva", index=False)
        for lab, gt in grp_tables.items():
            gt.to_excel(xw, sheet_name=lab[:31].replace("/", "-"), index=False)
    st.download_button("Baixar estatística descritiva (.xlsx)", buf_d.getvalue(), file_name="estatistica_descritiva.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =============================================================================
# Etapa 2 — preparo dos dados
# =============================================================================
with tab1:
    x = data["val"].to_numpy()

    st.subheader("Outliers: transformação + Tukey")
    a, b, c, d_ = st.columns([1.4, 1, 1, 1.4])
    tr_label = a.radio("Transformação antes do Tukey", ["Box-Cox", "Logarítmica", "Nenhuma (dados brutos)"], horizontal=False)
    tr = {"Box-Cox": "boxcox", "Logarítmica": "log", "Nenhuma (dados brutos)": "none"}[tr_label]
    k_tukey = b.number_input("k do Tukey (IQR)", 0.5, 5.0, 1.5, 0.5, help="1,5 = outlier; 3,0 = outlier extremo")
    iterative = c.checkbox("Tukey iterativo", help="Repete até não restarem outliers")
    use_out = d_.toggle("Excluir outliers dos dados usados nas etapas 3 e 4", value=False,
                        help="O Lab Operations não aplica essa exclusão em tempo real. Se ligar, as simulações usam "
                             "uma população mais limpa do que a que o middleware vai receber.")
    oa = mm.outlier_analysis(x, tr, k_tukey, iterative)
    st.caption(
        f"Cercas em unidade original: **{fmt(oa['lo'])} a {fmt(oa['hi'])}**: "
        f"{oa['n_out']} resultados fora ({100 * oa['n_out'] / len(x):.2f}%). "
        + (f"λ Box-Cox = {oa['lambda']:.3f}. " if "lambda" in oa else "")
        + "Na etapa 3 você escolhe se a sugestão de truncamento vem destas cercas ou do intervalo de referência da planilha."
    )
    if use_out:
        st.warning(f"Exclusão ativa: {oa['n_out']} resultados removidos das etapas 3 e 4.")

    h1, h2 = st.columns(2)
    fig = go.Figure(go.Histogram(x=x, nbinsx=60, marker_color=C1, name="resultados"))
    for v, lab in [(oa["lo"], "cerca inf."), (oa["hi"], "cerca sup.")]:
        if np.isfinite(v):
            fig.add_vline(x=v, line_dash="dash", line_color=ALARM, annotation_text=lab)
    h1.plotly_chart(base_layout(fig, 320, title="Distribuição original", showlegend=False, bargap=0.05),
                    width="stretch")
    fig = go.Figure(go.Histogram(x=oa["y"], nbinsx=60, marker_color=C3))
    fig.add_vline(x=oa["lo_t"], line_dash="dash", line_color=ALARM)
    fig.add_vline(x=oa["hi_t"], line_dash="dash", line_color=ALARM)
    h2.plotly_chart(base_layout(fig, 320, title=f"Escala transformada ({tr_label})", showlegend=False, bargap=0.05),
                    width="stretch")

    data2 = data[oa["keep"]].reset_index(drop=True) if use_out else data
    age_info = None

    if has_age:
        st.subheader("Resultado por idade × quantidade de resultados")
        a, b, c = st.columns(3)
        width = a.selectbox("Faixa etária (anos)", [1, 2, 5, 10], index=0)
        stat_age = b.radio("Estatística", ["média", "mediana"], horizontal=True)
        by_sex = c.checkbox("Separar por sexo", value=has_sex, disabled=not has_sex)
        ap = mm.age_profile(data2, width, "mean" if stat_age == "média" else "median", by_sex and has_sex)
        age_info = {"df": ap, "stat_label": stat_age.capitalize(), "by_sex": bool(by_sex and has_sex), "width": width,
                    "geral": float(np.mean(data2["val"]) if stat_age == "média" else np.median(data2["val"]))}
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06,
                            subplot_titles=(f"{stat_age.capitalize()} do resultado", "Nº de resultados"))
        groups = [(None, ap)] if not (by_sex and has_sex) else list(ap.groupby("sex", dropna=False))
        for i, (g, sub) in enumerate(groups):
            name = "todos" if g is None else str(g)
            col = SEX_COLORS[i % 4]
            fig.add_trace(go.Scatter(x=sub["faixa"], y=sub["valor"], mode="lines+markers", name=name,
                                     line=dict(color=col, width=2), marker=dict(size=6),
                                     customdata=sub["n"], hovertemplate="%{y:.3f} (n=%{customdata})"), 1, 1)
            fig.add_trace(go.Bar(x=sub["faixa"], y=sub["n"], name=name, marker_color=col, showlegend=False,
                                 hovertemplate="n=%{y}"), 2, 1)
        fig.add_hline(y=np.mean(data2["val"]) if stat_age == "média" else np.median(data2["val"]),
                      line_dash="dot", line_color=MUTED, row=1, col=1)
        fig.update_layout(barmode="group", bargap=0.1)
        fig.update_xaxes(title_text="idade (anos)", row=2, col=1)
        st.plotly_chart(base_layout(fig, 520), width="stretch")
        st.caption("Faixas com muitos resultados e média distante da linha pontilhada (média geral) são as que mais "
                   "deslocam a medida móvel quando a composição da população muda. Candidatas a filtro na etapa 3.")
    else:
        st.info("Informe a coluna de idade para ver o gráfico por idade.")

    if has_time:
        with st.expander("Resultado por hora do dia (mudança de população ao longo do dia)"):
            hp = mm.hour_profile(data2)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.08,
                                subplot_titles=("Média do resultado", "Nº de resultados"))
            fig.add_trace(go.Scatter(x=hp["hora"], y=hp["valor"], mode="lines+markers", line=dict(color=C1, width=2)), 1, 1)
            fig.add_trace(go.Bar(x=hp["hora"], y=hp["n"], marker_color=C1), 2, 1)
            fig.update_xaxes(title_text="hora", dtick=2, row=2, col=1)
            st.plotly_chart(base_layout(fig, 420, showlegend=False), width="stretch")

# =============================================================================
# Etapa 3 — configuração
# =============================================================================
with tab2:
    xv = data2["val"].to_numpy()
    st.subheader("Truncamento e população")
    # Intervalos de referência presentes na planilha (mais frequente primeiro)
    ref_opts = []
    if cmap.ref:
        vc = data2["ref"].dropna().value_counts()
        ref_opts = [(txt, int(n), mm.parse_ref_range(txt)) for txt, n in vc.items()]
        ref_opts = [o for o in ref_opts if o[2] is not None]
    SRC_RI, SRC_TK, SRC_PC, SRC_MAN = ("Intervalo de referência da planilha", "Cercas de Tukey (etapa 2)",
                                       "Percentis dos resultados", "Manual")
    src_opts = ([SRC_RI] if ref_opts else []) + [SRC_TK, SRC_PC, SRC_MAN]
    a, b = st.columns([1, 2])
    src = a.radio("Origem dos limites de truncamento", src_opts,
                  help="Para digitar valores próprios, escolha 'Manual' e informe a fonte.")
    if cmap.ref and not ref_opts:
        b.info("A coluna de intervalo de referência não tem textos reconhecíveis (ex.: '2.52 - 39.20').")
    tk_lo = oa["lo"] if np.isfinite(oa["lo"]) else float(xv.min())
    tk_hi = oa["hi"] if np.isfinite(oa["hi"]) else float(xv.max())
    tk_src = (f"Cercas de Tukey (k = {fmt(k_tukey, 1)}{', iterativo' if iterative else ''}) "
              f"sobre dados com transformação {tr_label.lower()}")
    if src == SRC_RI or (src == SRC_MAN and ref_opts):
        tot = sum(o[1] for o in ref_opts)
        ref_sel = (b.selectbox("Intervalo a usar", ref_opts,
                               format_func=lambda o: f"{o[0]}  ({o[1]} resultados, {100 * o[1] / tot:.1f}%)")
                   if src == SRC_RI else ref_opts[0])
        if src == SRC_RI and len(ref_opts) > 1:
            b.caption(f"A coluna tem {len(ref_opts)} intervalos diferentes (em geral por idade/sexo). O truncamento "
                      "usa um único par de limites: escolha o intervalo da população que o perfil vai cobrir e "
                      "ajuste o filtro de sexo/idade abaixo de acordo.")
        r_lo, r_hi = ref_sel[2]
        sug_lo = r_lo if r_lo is not None else float(xv.min())  # lado aberto = sem truncamento naquele lado
        sug_hi = r_hi if r_hi is not None else float(xv.max())
        trunc_src = f"Intervalo de referência da planilha ({ref_sel[0]})"
    else:
        sug_lo, sug_hi, trunc_src = tk_lo, tk_hi, tk_src
    if src == SRC_PC:
        pa, pb = b.columns(2)
        p_lo = pa.number_input("Percentil inferior", 0.0, 50.0, 2.5, 0.5)
        p_hi = pb.number_input("Percentil superior", 50.0, 100.0, 97.5, 0.5)
        sug_lo, sug_hi = (float(v_) for v_ in np.percentile(xv, [p_lo, p_hi]))
        trunc_src = (f"Percentis P{fmt(p_lo, 1)} e P{fmt(p_hi, 1)} dos {fmt(len(xv), 0)} resultados "
                     "(após filtros de teste/equipamento" + (" e exclusão de outliers)" if use_out else ")"))
    if src == SRC_MAN:
        man_src = b.text_input("Fonte dos limites de truncamento *",
                               placeholder="ex.: bula do fabricante; literatura (autor, ano); decisão do setor em dd/mm",
                               key="fonte_trunc")
        trunc_src = f"Manual: {man_src.strip()}" if man_src.strip() else "Manual (fonte não informada)"
        if not man_src.strip():
            b.warning("Informe a fonte: ela vai para o relatório em PDF.")
    a, b, c = st.columns(3)
    t_lo = a.number_input("Limite inferior de truncamento", value=float(round(sug_lo, 4)), format="%.4f",
                          disabled=src != SRC_MAN)
    t_hi = b.number_input("Limite superior de truncamento", value=float(round(sug_hi, 4)), format="%.4f",
                          disabled=src != SRC_MAN)
    t_mode_lab = c.radio("Resultados fora dos limites", ["Excluir do cálculo", "Substituir pelo limite (winsorizar)"])
    t_mode = "exclude" if t_mode_lab.startswith("Excluir") else "winsor"

    a, b = st.columns(2)
    sexes = None
    if has_sex:
        sex_vals = sorted(data2["sex"].dropna().astype(str).unique())
        sel_sex = a.multiselect("Sexo", sex_vals, default=sex_vals)
        sexes = None if set(sel_sex) == set(sex_vals) else tuple(sel_sex)
    age_min = age_max = None
    if has_age and data2["age"].min() < data2["age"].max():
        amin, amax = float(np.floor(data2["age"].min())), float(np.ceil(data2["age"].max()))
        r = b.slider("Faixa de idade (anos)", amin, amax, (amin, amax), step=1.0)
        age_min = r[0] if r[0] > amin else None
        age_max = r[1] if r[1] < amax else None
        if age_min is not None or age_max is not None:
            b.caption("Resultados sem idade são excluídos quando há filtro de idade.")

    st.subheader("Ferramenta")
    a, b, c, d_ = st.columns(4)
    stat_lab = a.selectbox("Estatística", list(mm.STAT_LABELS.values()))
    stat = [k for k, v in mm.STAT_LABELS.items() if v == stat_lab][0]
    N = int(b.number_input("Tamanho do bloco (N)", 2, 2000, 50, 5))
    step = int(c.number_input("Novo ponto a cada n resultados", 1, 1000, 1, 1,
                              help="50-5 = bloco de 50 resultados; novo ponto plotado a cada 5 resultados seguintes"))
    lam = None
    if stat == "EWMA":
        lam = d_.number_input("λ (EWMA)", 0.001, 1.0, round(2 / (N + 1), 4), format="%.4f",
                              help="Padrão 2/(N+1): mesmo 'centro de massa' da média móvel de N")
    st.caption(f"Configuração **{N}-{step}**: bloco de {N} resultados elegíveis, novo ponto a cada {step}."
               + (" O EWMA usa os N primeiros só para iniciar a série." if stat == "EWMA" else ""))

    # A série não depende do alvo/DP/k: calcula antes para oferecer a média dos blocos como alvo
    cfg = mm.MMConfig(stat=stat, N=N, step=step, lam=lam, trunc_lo=t_lo, trunc_hi=t_hi, trunc_mode=t_mode,
                      sexes=sexes, age_min=age_min, age_max=age_max)
    pop = mm.population_mask(data2, cfg)
    elig, _ = mm.apply_truncation(xv, pop, cfg)
    n_trunc = int((pop & ((xv < t_lo) | (xv > t_hi))).sum())
    ser, ctx = mm.compute_series(data2, cfg)
    if len(ser) < 5:
        st.error(f"Só {int(elig.sum())} resultados elegíveis: insuficiente para bloco de {N}.")
        st.stop()
    blocks_mean = float(ser["valor"].mean())
    blocks_sd = float(ser["valor"].std(ddof=1))

    st.subheader("Limites de controle")
    a, b, c, d_ = st.columns(4)
    k_lim = a.number_input("k (limites = alvo ± k·DP)", 0.5, 6.0, 2.0, 0.5)
    tgt_mode = b.radio("Média-alvo", ["Média dos blocos plotados", "Manual"],
                       help="Média dos blocos plotados = média de todos os pontos da medida móvel no gráfico "
                            "(média das médias móveis; na mediana móvel, média das medianas).")
    if "alvo_manual" not in st.session_state:
        st.session_state["alvo_manual"] = round(blocks_mean, 4)
    tgt_manual = tgt_mode == "Manual"
    if tgt_manual:
        tgt = b.number_input("Alvo manual", format="%.4f", key="alvo_manual")
        f_ = b.text_input("Fonte do alvo *", placeholder="ex.: alvo cadastrado no Lab Operations em dd/mm",
                          key="fonte_alvo").strip()
        tgt_src = f"Manual: {f_}" if f_ else "Manual (fonte não informada)"
        if not f_:
            b.warning("Informe a fonte do alvo.")
    else:
        tgt = blocks_mean
        b.caption(f"Alvo = {fmt(blocks_mean, 4)} ({len(ser)} blocos)")
        tgt_src = f"Média dos {fmt(len(ser), 0)} blocos plotados (média das médias móveis)"
    sd_mode = c.radio("DP", ["DP dos blocos plotados", "Manual"])
    if "dp_manual" not in st.session_state:
        st.session_state["dp_manual"] = round(blocks_sd, 4)
    sd_manual = sd_mode == "Manual"
    if sd_manual:
        sdv = c.number_input("DP manual", min_value=0.0, format="%.4f", key="dp_manual")
        f_ = c.text_input("Fonte do DP *", placeholder="ex.: DP da série histórica jan–mar/2026", key="fonte_dp").strip()
        sd_src = f"Manual: {f_}" if f_ else "Manual (fonte não informada)"
        if not f_:
            c.warning("Informe a fonte do DP.")
    else:
        sdv = blocks_sd
        c.caption(f"DP = {fmt(blocks_sd, 4)}")
        sd_src = f"DP dos {fmt(len(ser), 0)} blocos plotados (DP das médias móveis)"
    consec = int(d_.number_input("Pontos consecutivos fora para alarmar", 1, 10, 1, 1))
    multi_inst = len(sel_i) > 1
    # Valores de sexo elegíveis para estratificar (ignora grupos residuais < 1%)
    sex_counts = data2["sex"].dropna().astype(str).value_counts() if has_sex else pd.Series(dtype=int)
    sex_pool = list(cfg.sexes) if cfg.sexes is not None else list(sex_counts.index)
    sex_vals = [v for v in sex_pool if sex_counts.get(v, 0) >= 0.01 * max(len(data2), 1)]
    sex_residual = [v for v in sex_pool if v not in sex_vals]
    ST_NONE, ST_EQ, ST_SEX, ST_EQSEX = "Nenhuma", "Equipamento", "Sexo", "Equipamento e sexo"
    strat_opts = [ST_NONE] + ([ST_EQ] if multi_inst else []) + ([ST_SEX] if len(sex_vals) > 1 else []) \
        + ([ST_EQSEX] if multi_inst and len(sex_vals) > 1 else [])
    strat = st.radio("Estratificar por", strat_opts, horizontal=True,
                     help="Cada estrato vira um fluxo próprio, em ordem cronológica, com alvo e DP próprios, como cada "
                          "perfil cadastrado no Lab Operations. Equipamento exige 2+ equipamentos na barra lateral; "
                          "sexo exige a coluna de sexo e 2+ sexos no filtro de população.")
    if strat in (ST_SEX, ST_EQSEX) and sex_residual:
        st.caption(f"Sexo(s) com menos de 1% dos resultados ficaram fora dos estratos: {', '.join(sex_residual)}.")

    cfg = replace(cfg, k=k_lim, consec=consec,
                  target=tgt if tgt_manual else None, sd=sdv if (sd_manual and sdv > 0) else None)
    lim = mm.control_limits(ser, cfg)
    fa = mm.false_alarm_summary(ser, lim, cfg, ctx.n_stream)

    def short_names(names: list[str]) -> dict[str, str]:
        """Remove sufixo comum após ' - ' (ex.: ' - Laboratory 1') para a legenda."""
        parts = [n.rsplit(" - ", 1) for n in names]
        if all(len(p_) == 2 for p_ in parts) and len({p_[1] for p_ in parts}) == 1:
            return {n: p_[0] for n, p_ in zip(names, parts)}
        return {n: n for n in names}

    def volume(dg: pd.DataFrame, ctx_) -> dict | None:
        """Resultados elegíveis (dentro do truncamento e da população) por dia com resultado."""
        if not has_time or len(dg) == 0:
            return None
        days = dg["t"].dt.normalize()
        all_days = days.unique()
        per = days.iloc[ctx_.elig_pos].value_counts().reindex(all_days, fill_value=0)
        mean = ctx_.elig_pos.size / max(len(all_days), 1)
        return {"mean": mean, "med": float(per.median()), "p10": float(per.quantile(0.10)),
                "max": int(per.max()) if len(per) else 0, "daily": per,
                "hours": (N / mean * 24) if mean > 0 else np.inf}

    def stratum(dg: pd.DataFrame, label: str, color=None, eq=None, sex=None, base=None) -> dict:
        dg = dg.reset_index(drop=True)
        vg = dg["val"].to_numpy()
        pop_g = mm.population_mask(dg, cfg)
        ser_g, ctx_g = base if base else mm.compute_series(dg, cfg)
        ok = len(ser_g) >= 5
        lim_g = (lim if base else mm.control_limits(ser_g, cfg)) if ok else None
        return dict(label=label, color=color, eq=eq, sex=sex, ser=ser_g, lim=lim_g,
                    fa=(fa if base else mm.false_alarm_summary(ser_g, lim_g, cfg, ctx_g.n_stream)) if ok else None,
                    n=len(dg), n_elig=int(ctx_g.elig_pos.size), n_pop=int(pop_g.sum()),
                    n_trunc=int((pop_g & ((vg < t_lo) | (vg > t_hi))).sum()), vol=volume(dg, ctx_g))

    def sex_label(v) -> str:
        return {"female": "Feminino", "f": "Feminino", "feminino": "Feminino",
                "male": "Masculino", "m": "Masculino", "masculino": "Masculino"}.get(str(v).strip().lower(), str(v))

    SEX_COLORS = ["#6250d6", "#1baf7a", "#eda100", "#e87ba4"]
    geral = stratum(data2, "Geral (todos os estratos juntos)" if strat != ST_NONE else stat_lab, base=(ser, ctx))
    strata: list[dict] = []
    eq_names = [g.strip() for g in sel_i]
    eq_short = short_names(eq_names) if eq_names else {}
    if strat == ST_EQ:
        for i, g in enumerate(eq_names):
            strata.append(stratum(data2[data2["group"] == g], eq_short[g], CAT[i % len(CAT)], eq=eq_short[g]))
    elif strat == ST_SEX:
        for i, sx in enumerate(sex_vals):
            strata.append(stratum(data2[data2["sex"] == sx], sex_label(sx), SEX_COLORS[i % 4], sex=sex_label(sx)))
    elif strat == ST_EQSEX:
        for i, g in enumerate(eq_names):
            for sx in sex_vals:
                sub = data2[(data2["group"] == g) & (data2["sex"] == sx)]
                strata.append(stratum(sub, f"{eq_short[g]} · {sex_label(sx)}", CAT[i % len(CAT)],
                                      eq=eq_short[g], sex=sex_label(sx)))

    def metric_row(r_: dict):
        cm = st.columns(7)
        cm[0].metric(f"Elegíveis (de {fmt(r_['n'], 0)})", fmt(r_["n_elig"], 0))
        cm[1].metric("Fora do truncamento", f"{fmt(r_['n_trunc'], 0)} ({100 * r_['n_trunc'] / max(r_['n_pop'], 1):.1f}%)")
        v_ = r_["vol"]
        cm[2].metric("Elegíveis / dia", fmt(v_["mean"], 1) if v_ else "—",
                     help=(f"Média de resultados dentro do truncamento por dia com resultado. Mediana {fmt(v_['med'], 0)}, "
                           f"P10 {fmt(v_['p10'], 0)}, máximo {v_['max']}.") if v_ else "Requer coluna de data.")
        if r_["lim"] is None:
            cm[3].metric("Alvo", "—")
            cm[4].caption(f"Insuficiente para bloco de {N}.")
            return
        cm[3].metric("Alvo", fmt(r_["lim"]["alvo"]))
        cm[4].metric("DP", fmt(r_["lim"]["DP"], 4))
        cm[5].metric("Pontos em alarme", f"{r_['fa']['% pontos em alarme']:.1f}%")
        cm[6].metric("Episódios / dia" if has_time else "Episódios / 1000 res.",
                     fmt(r_["fa"]["episódios / dia"] if has_time else r_["fa"]["episódios / 1000 resultados"], 2))
        if v_ and np.isfinite(v_["hours"]):
            h_ = v_["hours"]
            txt = f"Bloco de {N} se renova em ≈ {h_:.0f} h" + (f" (≈ {fmt(h_ / 24, 1)} dias)" if h_ >= 24 else "")
            if h_ > 48:
                st.warning(txt + ": o alarme reage devagar para este estrato.")
            else:
                st.caption(txt)

    def row_label(text, color=None):
        mark = f'<span style="color:{color};font-size:1.1em">■</span> ' if color else ""
        st.markdown(f"{mark}**{text}**", unsafe_allow_html=True)

    if strat != ST_NONE:
        row_label(geral["label"])
    metric_row(geral)
    if strat == ST_EQSEX:
        eq_order = list(dict.fromkeys(r_["eq"] for r_ in strata))
        for t_, eq in zip(st.tabs(eq_order), eq_order):
            with t_:
                for r_ in strata:
                    if r_["eq"] == eq:
                        row_label(r_["label"], r_["color"])
                        metric_row(r_)
    else:
        for r_ in strata:
            row_label(r_["label"], r_["color"])
            metric_row(r_)
    if strata and (tgt_manual or sd_manual):
        st.caption("Alvo/DP manuais são aplicados igualmente a todos os estratos.")
    if not (tgt_manual or sd_manual):
        st.caption("Alvo e DP calculados dos próprios blocos plotados (dentro da amostra): com k=2 cerca de 5% dos "
                   "pontos ficam fora por construção. Para avaliar limites já em uso no Lab Operations, use valores manuais.")

    a, b, c = st.columns([1, 1, 2])
    xaxis = a.radio("Eixo X", ["data/hora", "sequência"], horizontal=True, disabled=not has_time)
    show_raw = b.checkbox("Mostrar resultados individuais")
    tea = c.number_input("Erro sistemático máximo permitido (%) para desenhar no gráfico (0 = não desenhar)",
                         0.0, 100.0, 0.0, 0.5)

    def hover_cd(ser_):
        """customdata para o hover: [data/hora do último resultado do bloco, nº do resultado no fluxo]."""
        if has_time:
            ts = pd.to_datetime(ser_["t"]).dt.strftime("%d/%m/%Y %H:%M:%S").fillna("sem data")
        else:
            ts = pd.Series("sem data", index=ser_.index)
        return np.c_[ts.to_numpy(dtype=object), ser_["seq"].to_numpy()]

    HOVER_PT = "<br>%{customdata[0]} · resultado nº %{customdata[1]}"

    def series_figure(ser_, lim_, title, bias_start=None, raw_vals=None):
        xs = ser_["t"] if (has_time and xaxis == "data/hora") else ser_["seq"]
        al = mm.alarm_flags(ser_["valor"].to_numpy(), lim_, cfg.consec)
        fig = go.Figure()
        if raw_vals is not None:
            rx = data2["t"] if (has_time and xaxis == "data/hora") else np.arange(1, len(data2) + 1)
            fig.add_trace(go.Scattergl(x=rx, y=raw_vals, mode="markers", name="resultados",
                                       marker=dict(size=3, color=MUTED, opacity=0.35), hoverinfo="skip"))
        cd = hover_cd(ser_)
        fig.add_trace(go.Scatter(x=xs, y=ser_["valor"], mode="lines", name=stat_lab, line=dict(color=C1, width=2),
                                 customdata=cd, hovertemplate="%{y:.4f}" + HOVER_PT + "<extra></extra>"))
        fig.add_trace(go.Scatter(x=xs[al], y=ser_["valor"][al], mode="markers", name="alarme",
                                 marker=dict(color=ALARM, size=8, line=dict(color="white", width=1.5)),
                                 customdata=cd[al], hovertemplate="alarme: %{y:.4f}" + HOVER_PT + "<extra></extra>"))
        fig.add_hline(y=lim_["alvo"], line_color=INK, line_width=1, annotation_text="alvo", annotation_position="right")
        for kname in ("LI", "LS"):
            fig.add_hline(y=lim_[kname], line_color=ALARM, line_dash="dash", line_width=1,
                          annotation_text=f"{kname} ({cfg.k:g} DP)", annotation_position="right")
        if tea > 0:
            for sgn in (-1, 1):
                fig.add_hline(y=lim_["alvo"] * (1 + sgn * tea / 100), line_color=C4, line_dash="dot",
                              annotation_text=f"ESM {'+' if sgn > 0 else '−'}{tea:g}%", annotation_position="left")
        if bias_start is not None:  # shape + anotação (add_vline falha com eixo de datas)
            fig.add_shape(type="line", x0=bias_start, x1=bias_start, yref="paper", y0=0, y1=1,
                          line=dict(color=C2, width=2))
            fig.add_annotation(x=bias_start, yref="paper", y=1, text="início do erro", showarrow=False,
                               yanchor="bottom", font=dict(color=C2))
        if raw_vals is None:
            pad = (lim_["LS"] - lim_["LI"]) * 0.6
            lo = min(lim_["LI"] - pad, ser_["valor"].min())
            hi = max(lim_["LS"] + pad, ser_["valor"].max())
            fig.update_yaxes(range=[lo, hi])
        return base_layout(fig, 460, title=title, xaxis_title="data/hora" if (has_time and xaxis == "data/hora") else "nº do resultado")

    x_is_time = has_time and xaxis == "data/hora"
    if strata:
        panels = list(dict.fromkeys(r_["sex"] for r_ in strata)) if strat == ST_EQSEX else [None]
        if len(panels) > 1:
            fig = make_subplots(rows=1, cols=len(panels), shared_yaxes=True, horizontal_spacing=0.03,
                                subplot_titles=panels)
        else:
            fig = go.Figure()
        alarm_legend, seen_legend = False, set()
        for r_ in strata:
            if r_["lim"] is None:
                continue
            ser_g, lim_g, col = r_["ser"], r_["lim"], r_["color"]
            pos = dict(row=1, col=panels.index(r_["sex"]) + 1) if len(panels) > 1 else {}
            leg_name = r_["eq"] if strat == ST_EQSEX else r_["label"]
            xs = ser_g["t"] if x_is_time else ser_g["seq"]
            cd = hover_cd(ser_g)
            fig.add_trace(go.Scatter(x=xs, y=ser_g["valor"], mode="lines", name=leg_name, legendgroup=leg_name,
                                     showlegend=leg_name not in seen_legend, line=dict(color=col, width=1.8),
                                     customdata=cd,
                                     hovertemplate=f"{r_['label']}: %{{y:.4f}}" + HOVER_PT + "<extra></extra>"), **pos)
            seen_legend.add(leg_name)
            x_rng = [xs.min(), xs.max()]
            for key, dash in (("LS", "dash"), ("LI", "dash"), ("alvo", "dot")):
                fig.add_trace(go.Scatter(x=x_rng, y=[lim_g[key]] * 2, mode="lines", legendgroup=leg_name,
                                         showlegend=False, line=dict(color=col, dash=dash, width=1),
                                         hovertemplate=f"{r_['label']} {key}: {lim_g[key]:.4f}<extra></extra>"), **pos)
            al = mm.alarm_flags(ser_g["valor"].to_numpy(), lim_g, cfg.consec)
            fig.add_trace(go.Scatter(x=xs[al], y=ser_g["valor"][al], mode="markers", legendgroup=leg_name,
                                     showlegend=False, marker=dict(color=col, size=9, line=dict(color="white", width=1.5)),
                                     customdata=cd[al],
                                     hovertemplate=f"alarme {r_['label']}: %{{y:.4f}}" + HOVER_PT + "<extra></extra>"), **pos)
            alarm_legend = alarm_legend or bool(al.any())
        if alarm_legend:  # entrada de legenda neutra: o alarme herda a cor do estrato
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="alarme (cor do estrato)",
                                     marker=dict(color=MUTED, size=9, line=dict(color="white", width=1.5))))
        fig.update_xaxes(title="data/hora" if x_is_time else "nº do resultado no estrato")
        ttl = (f"{stat_lab}: {N}-{step} por equipamento, um painel por sexo" if strat == ST_EQSEX else
               f"{stat_lab}: {N}-{step} por {strat.lower()}")
        fig = base_layout(fig, 540 if len(panels) > 1 else 500,
                          title=ttl + " (tracejado = LI/LS, pontilhado = alvo)", hovermode="closest")
        if len(panels) > 1:  # legenda acima dos títulos dos painéis
            fig.update_layout(margin=dict(t=125), legend=dict(y=1.16))
        st.plotly_chart(fig, width="stretch")

        def table_row(r_: dict) -> dict:
            v_, l_, f_ = r_["vol"], r_["lim"], r_["fa"]
            row = {"estrato": r_["label"], "resultados": r_["n"], "elegíveis": r_["n_elig"],
                   "fora do truncamento": r_["n_trunc"],
                   "elegíveis/dia": round(v_["mean"], 1) if v_ else np.nan,
                   "mediana/dia": v_["med"] if v_ else np.nan,
                   "bloco se renova em (h)": round(v_["hours"]) if v_ and np.isfinite(v_["hours"]) else np.nan}
            if l_ is not None:
                row.update({"alvo": l_["alvo"], "DP": l_["DP"], "LI": l_["LI"], "LS": l_["LS"],
                            "Δ alvo vs geral (DP)": (l_["alvo"] - lim["alvo"]) / l_["DP"] if l_["DP"] else np.nan,
                            "% pontos em alarme": f_["% pontos em alarme"], "episódios": f_["episódios de alarme"],
                            "episódios/dia": f_["episódios / dia"]})
            return row

        st.markdown("**Comparação entre estratos**")
        st.dataframe(pd.DataFrame([table_row(geral)] + [table_row(r_) for r_ in strata]).round(4),
                     hide_index=True, width="stretch")
        st.caption("Cada estrato tem fluxo, alvo e DP próprios (a não ser que alvo/DP sejam manuais). "
                   "Δ alvo vs geral = distância entre o alvo do estrato e o alvo geral, em DPs do próprio estrato: "
                   "acima de ~0,3 DP, um alvo único desloca os falsos alarmes para um lado. "
                   "Clique na legenda do gráfico para esconder/mostrar um estrato.")
    else:
        st.plotly_chart(series_figure(ser, lim, f"{stat_lab}: {N}-{step}", raw_vals=xv if show_raw else None),
                        width="stretch")

    if has_time:
        st.markdown("**Volume diário dentro do truncamento**")
        vrows = []
        for r_ in [geral] + strata:
            v_ = r_["vol"]
            if v_ is None:
                continue
            dly = v_["daily"]
            vrows.append({
                "estrato": r_["label"] if strata else "Geral",
                "dias com resultado": len(dly),
                "elegíveis": r_["n_elig"],
                "média/dia": round(v_["mean"], 1),
                "mediana/dia": v_["med"],
                "P10/dia": v_["p10"],
                "mín./dia": int(dly.min()) if len(dly) else np.nan,
                "máx./dia": v_["max"],
                f"dias com menos de {N} (%)": round(100 * (dly < N).mean(), 1) if len(dly) else np.nan,
                f"bloco de {N} se renova em (h)": round(v_["hours"]) if np.isfinite(v_["hours"]) else np.nan,
            })
        st.dataframe(pd.DataFrame(vrows), hide_index=True, width="stretch")
        st.caption(f"Resultados dentro do truncamento e do filtro de população, por dia com resultado. Nos dias com "
                   f"menos de {N} elegíveis, um bloco não se completa no próprio dia: a medida móvel reage a um erro "
                   "só no dia seguinte ou depois.")

    with st.expander("Comparar as quatro estatísticas com a mesma configuração (falsos alarmes)"):
        rows = []
        for s_key, s_lab in mm.STAT_LABELS.items():
            cfg_s = replace(cfg, stat=s_key, lam=None)
            ser_s, ctx_s = mm.compute_series(data2, cfg_s)
            lim_s = mm.control_limits(ser_s, cfg_s)
            fa_s = mm.false_alarm_summary(ser_s, lim_s, cfg_s, ctx_s.n_stream)
            rows.append({"estatística": s_lab, "alvo": lim_s["alvo"], "DP": lim_s["DP"], "LI": lim_s["LI"], "LS": lim_s["LS"],
                         "% pontos em alarme": fa_s["% pontos em alarme"], "episódios": fa_s["episódios de alarme"],
                         "episódios / dia": fa_s["episódios / dia"]})
        st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
        st.caption("Com limites calculados da própria série, o % de pontos em alarme fica parecido entre as estatísticas. "
                   "O que muda é quantos episódios isso gera e, principalmente, a detecção de erro (etapa 4).")


    # ------------------------------------------------------------------ relatório PDF
    st.subheader("Relatório em PDF")
    st.caption("Resume a configuração escolhida nesta etapa para documentar o estudo. A simulação de bias não entra.")
    eq_short = short_names([g.strip() for g in sel_i]) if sel_i else {}
    with st.form("form_pdf"):
        a, b = st.columns(2)
        rp_analito = a.text_input("Analito", value=str(sel_t) if sel_t else c_res)
        rp_setor = b.text_input("Laboratório / setor")
        a, _ = st.columns(2)
        rp_elab = a.text_input("Elaborado por")
        rp_obs = st.text_area("Observações (opcional)", height=80)
        gen = st.form_submit_button("Gerar relatório PDF", type="primary")
    if gen:
        pct_out = 100 * oa["n_out"] / len(data) if len(data) else 0
        prep_rows = [
            ["Exclusão de outliers",
             (f"Sim: {fmt(oa['n_out'], 0)} resultados removidos ({fmt(pct_out, 2)}%). Cercas de Tukey "
              f"(k = {fmt(k_tukey, 1)}{', iterativo' if iterative else ''}) sobre dados com transformação "
              f"{tr_label.lower()}: {fmt(oa['lo'])} a {fmt(oa['hi'])}.") if use_out else "Não"],
            ["Filtro de sexo", "Todos" if cfg.sexes is None else ", ".join(cfg.sexes)],
            ["Filtro de idade", "Todas as idades" if (cfg.age_min is None and cfg.age_max is None) else
             f"{fmt(cfg.age_min, 0) if cfg.age_min is not None else 'mín.'} a "
             f"{fmt(cfg.age_max, 0) if cfg.age_max is not None else 'máx.'} anos"],
            ["Ordem dos resultados", f"Cronológica ({c_date}{' + ' + c_time if cmap.time else ''})" if has_time
             else "Ordem das linhas da planilha"],
            ["Resultados lidos", f"{fmt(rep['linhas_validas'], 0)} válidos de {fmt(rep['linhas_lidas'], 0)} linhas "
             "(descartadas: sem resultado numérico" + (" ou sem data/hora)" if has_time else ")")],
        ]
        prep_rows.append(["Estratificação", {
            ST_NONE: "Nenhuma: fluxo único" + (" com os equipamentos em ordem cronológica" if multi_inst else ""),
            ST_EQ: "Por equipamento: fluxo, alvo e DP próprios em cada um",
            ST_SEX: "Por sexo: fluxo, alvo e DP próprios em cada um",
            ST_EQSEX: "Por equipamento e sexo: fluxo, alvo e DP próprios em cada combinação"}[strat]])
        cfg_rows = [
            ["Tipo de medida móvel", stat_lab, "—"],
            ["Tamanho do bloco (N)", f"{N} resultados", "—"],
            ["Novo ponto a cada (pulo)", f"{step} resultado{'s' if step > 1 else ''}", f"Configuração {N}-{step}"],
        ]
        if stat == "EWMA":
            cfg_rows.append(["Fator lambda (EWMA)", fmt(cfg.lam_eff, 4), "2/(N+1)" if lam is None or
                             abs(lam - 2 / (N + 1)) < 1e-4 else "Definido pelo usuário"])
        cfg_rows += [
            ["Truncamento inferior", fmt(t_lo, 4), trunc_src],
            ["Truncamento superior", fmt(t_hi, 4), trunc_src],
            ["Resultados fora do truncamento", "Excluídos do cálculo" if t_mode == "exclude" else
             "Substituídos pelo limite", f"{fmt(n_trunc, 0)} resultados ({fmt(100 * n_trunc / max(pop.sum(), 1), 1)}%) "
             "no período"],
        ]
        if geral["vol"]:
            gv = geral["vol"]
            cfg_rows.append(["Resultados elegíveis por dia" + (" (geral)" if strata else ""), fmt(gv["mean"], 1),
                             f"Média nos dias com resultado (mediana {fmt(gv['med'], 0)}); bloco de {N} se renova "
                             f"em cerca de {gv['hours']:.0f} h"])
        tag = " (geral)" if strata else ""
        lim_rows = [
            [f"Valor-alvo{tag}", fmt(lim["alvo"], 4), tgt_src],
            [f"DP{tag}", fmt(lim["DP"], 4), sd_src],
            ["Multiplicador k", fmt(cfg.k, 1), "Limites = alvo ± k × DP"],
            [f"Limite inferior (LI){tag}", fmt(lim["LI"], 4), "alvo - k × DP"],
            [f"Limite superior (LS){tag}", fmt(lim["LS"], 4), "alvo + k × DP"],
            ["Regra de alarme", "1 ponto fora" if cfg.consec == 1 else f"{cfg.consec} pontos consecutivos fora",
             "Mesmo lado do alvo" if cfg.consec > 1 else "—"],
        ]
        lim_eq = None
        x_is_time = has_time
        if strata:
            lim_eq = pd.DataFrame([{
                "Estrato": r_["label"],
                "Eleg./dia": fmt(r_["vol"]["mean"], 1) if r_["vol"] else "—",
                "Alvo": fmt(r_["lim"]["alvo"], 4) if r_["lim"] else "—",
                "DP": fmt(r_["lim"]["DP"], 4) if r_["lim"] else "—", "LI": fmt(r_["lim"]["LI"], 4) if r_["lim"] else "—",
                "LS": fmt(r_["lim"]["LS"], 4) if r_["lim"] else "—",
                "Pontos fora (%)": fmt(r_["fa"]["% pontos em alarme"], 1) if r_["fa"] else "—"} for r_ in strata])

            def pdf_item(r_):
                return {"nome": r_["eq"] if strat == ST_EQSEX else r_["label"], "cor": r_["color"],
                        "x": r_["ser"]["t"] if x_is_time else r_["ser"]["seq"], "y": r_["ser"]["valor"], "lim": r_["lim"],
                        "alarme": mm.alarm_flags(r_["ser"]["valor"].to_numpy(), r_["lim"], cfg.consec)}
            ok_strata = [r_ for r_ in strata if r_["lim"] is not None]
            if strat == ST_EQSEX:
                paineis = [{"titulo": sx, "itens": [pdf_item(r_) for r_ in ok_strata if r_["sex"] == sx]}
                           for sx in dict.fromkeys(r_["sex"] for r_ in ok_strata)]
                itens = None
            else:
                paineis = None
                itens = [pdf_item(r_) for r_ in ok_strata]
            fig_note = ("Cada estrato com fluxo, alvo e DP próprios" if not (tgt_manual or sd_manual)
                        else "Alvo/DP manuais aplicados a todos os estratos")
        else:
            paineis = None
            itens = [{"nome": stat_lab, "x": ser["t"] if x_is_time else ser["seq"], "y": ser["valor"], "lim": lim,
                      "alarme": mm.alarm_flags(ser["valor"].to_numpy(), lim, cfg.consec)}]
            fig_note = (f"{fmt(fa['% pontos em alarme'], 1)}% dos pontos fora dos limites; "
                        f"{fa['episódios de alarme']} episódios de alarme no período")
        if age_info:
            age_info["legenda"] = (f"Figura 1 – {age_info['stat_label']} do resultado (painel superior) e número de "
                                   f"resultados (painel inferior) por idade, faixas de {age_info['width']} ano(s)"
                                   + (", por sexo" if age_info["by_sex"] else "") + ". Linha pontilhada: "
                                   f"{age_info['stat_label'].lower()} geral.")
        period = (f"{data2['t'].min():%d/%m/%Y} a {data2['t'].max():%d/%m/%Y}" if has_time else "—")
        equip_txt = (", ".join(eq_short.get(g.strip(), g) for g in sel_i) if sel_i else "Não informado")
        info = {
            "titulo": "Relatório de configuração da medida móvel",
            "subtitulo": f"{rp_analito} · {stat_lab} {N}-{step} · " + (
                eq_short.get(sel_i[0].strip(), sel_i[0]) if len(sel_i) == 1 else
                f"{len(sel_i)} equipamentos" if sel_i else "equipamento não informado"),
            "analito": rp_analito,
            "equip_curto": (eq_short.get(sel_i[0].strip(), sel_i[0]) if len(sel_i) == 1 else
                            f"{len(sel_i)} equipamentos" if sel_i else ""),
            "identificacao": [("Analito", rp_analito), ("Equipamento(s)", equip_txt),
                              ("Laboratório / setor", rp_setor or "—"),
                              ("Período dos dados", period), ("Resultados avaliados", fmt(len(data2), 0)),
                              ("Arquivo de origem", f"{up.name}" + (f" (aba {sheet})" if sheet else ""))],
            "preparo": prep_rows, "idade": age_info,
            "config": cfg_rows, "limites": lim_rows, "limites_equip": lim_eq,
            "serie": {"itens": itens, "paineis": paineis, "x_is_time": x_is_time, "ylabel": stat_lab.split(" (")[0].lower(),
                      "legenda": f"Figura {2 if age_info else 1} – {stat_lab} {N}-{step} calculada sobre os dados do "
                                 f"período, com alvo (pontilhado) e limites LI/LS (tracejado). {fig_note}."},
            "observacoes": rp_obs, "elaborado": rp_elab,
            "emitido": pd.Timestamp.now(tz="America/Sao_Paulo").to_pydatetime(),
        }
        with st.spinner("Gerando PDF…"):
            st.session_state["pdf"] = report.build_pdf(info)
            st.session_state["pdf_name"] = (f"relatorio_medida_movel_{rp_analito}_{stat}_{N}-{step}.pdf"
                                            .replace(" ", "_").replace("/", "-"))
    if "pdf" in st.session_state:
        st.download_button("Baixar relatório PDF", st.session_state["pdf"], file_name=st.session_state["pdf_name"],
                           mime="application/pdf")
        st.caption("O PDF reflete a configuração do momento em que foi gerado. Mudou algo? Gere de novo.")

# =============================================================================
# Etapa 4 — simulação de bias
# =============================================================================
with tab3:
    st.markdown(
        "Um erro sistemático é injetado em pontos aleatórios do fluxo real de resultados e mantido daí em diante. "
        "O erro entra **antes** do truncamento: um bias grande empurra resultados para fora dos limites e muda o que a "
        "ferramenta enxerga. Para cada bias, mede-se **quantos resultados de pacientes são liberados entre o início do "
        "erro e o alarme** (NPed). Só conta o alarme no sentido do erro. Os limites são os da etapa 3."
    )
    a, b, c, d_ = st.columns(4)
    kind = a.radio("Tipo de bias", ["%", "unidade do resultado"], horizontal=True)
    kind_k = "%" if kind == "%" else "abs"
    default_b = "-10, -5, -3, -2, 2, 3, 5, 10" if kind_k == "%" else "-0.3, -0.2, -0.1, 0.1, 0.2, 0.3"
    b_txt = b.text_input("Vieses a simular (separados por vírgula ou ;)", default_b)
    n_sims = int(c.number_input("Simulações por bias", 20, 2000, 200, 20))
    horizon = int(d_.number_input("Horizonte máx. (resultados após o erro)", 50, 100000,
                                  int(min(1000, max(50, len(xv) // 3))), 50,
                                  help="Sem alarme dentro do horizonte = não detectado"))
    a, b = st.columns([1, 3])
    seed = int(a.number_input("Semente aleatória", 0, 10**6, 42))
    compare = b.checkbox("Comparar as quatro estatísticas (mesmo N, passo, truncamento, k e mesmos pontos de início)")

    try:
        biases = [float(t.replace(",", ".")) for t in b_txt.replace(";", " ").replace(", ", " ").split() if t.strip()]
    except ValueError:
        st.error("Lista de vieses inválida. Ex.: -10 -5 -3 3 5 10")
        st.stop()
    biases = [v for v in biases if v != 0]

    sig = (up.name, len(data2), repr(cfg), tuple(biases), kind_k, n_sims, horizon, seed, compare)
    if st.button("Rodar simulação", type="primary"):
        results = {}
        keys = list(mm.STAT_LABELS) if compare else [stat]
        prog = st.progress(0.0, text="Simulando…")
        try:
            for i, s_key in enumerate(keys):
                cfg_s = cfg if s_key == stat else replace(cfg, stat=s_key, lam=None)
                ser_s, ctx_s = (ser, ctx) if s_key == stat else mm.compute_series(data2, cfg_s)
                lim_s = lim if s_key == stat else mm.control_limits(ser_s, replace(cfg_s, target=None, sd=None))
                fa_s = mm.false_alarm_summary(ser_s, lim_s, cfg_s, ctx_s.n_stream)
                tab, rawsim, h_eff = mm.simulate_bias(ctx_s, cfg_s, lim_s, biases, kind_k, n_sims, horizon, seed)
                results[s_key] = dict(tab=tab, raw=rawsim, h=h_eff, lim=lim_s, fa=fa_s)
                prog.progress((i + 1) / len(keys), text=f"Simulando… {mm.STAT_LABELS[s_key]}")
        except ValueError as e:
            st.error(str(e))
            st.stop()
        prog.empty()
        st.session_state["sim"] = (sig, results)

    if "sim" not in st.session_state:
        st.info("Ajuste os parâmetros e clique em **Rodar simulação**.")
    else:
        sig_old, results = st.session_state["sim"]
        if sig_old != sig:
            st.warning("Os parâmetros mudaram desde a última simulação. Rode de novo para atualizar.")
        main_key = stat if stat in results else next(iter(results))
        r0 = results[main_key]
        if r0["h"] < horizon:
            st.warning(f"Dados curtos: o horizonte foi reduzido para {r0['h']} resultados.")
        unit = "%" if kind_k == "%" else ""

        def show_tab(t: pd.DataFrame, h: int) -> pd.DataFrame:
            out = t.copy()
            for col in ["MNPed (mediana)", "P25", "P75", "P95"]:
                out[col] = out[col].map(lambda v: f"> {h}" if not np.isfinite(v) else f"{v:.0f}")
            out["ANPed (média, detectados)"] = out["ANPed (média, detectados)"].round(1)
            out["% detectado"] = out["% detectado"].round(1)
            out["bias"] = out["bias"].map(lambda v: f"{v:+g}{unit}")
            return out

        st.subheader(f"{mm.STAT_LABELS[main_key]}: {cfg.N}-{cfg.step}")
        st.caption(f"Falso alarme sem bias: {r0['fa']['% pontos em alarme']:.1f}% dos pontos, "
                   f"{r0['fa']['episódios de alarme']} episódios"
                   + (f" ({r0['fa']['episódios / dia']:.2f}/dia)" if has_time else "") + ".")
        st.dataframe(show_tab(r0["tab"], r0["h"]), hide_index=True, width="stretch")

        # Curva de detecção: MNPed por bias (com faixa P25–P75)
        fig = go.Figure()
        for i, (s_key, r) in enumerate(results.items()):
            t = r["tab"].sort_values("bias")
            cap = r["h"] * 1.05
            y = t["MNPed (mediana)"].clip(upper=cap)
            col = [C1, C2, C3, C4][list(mm.STAT_LABELS).index(s_key)]
            fig.add_trace(go.Scatter(
                x=t["bias"], y=y, mode="lines+markers", name=mm.STAT_LABELS[s_key], line=dict(color=col, width=2),
                marker=dict(size=8),
                error_y=dict(type="data", symmetric=False, array=(t["P75"].clip(upper=cap) - y),
                             arrayminus=(y - t["P25"].clip(upper=cap)), thickness=1, width=4) if len(results) == 1 else None,
                customdata=np.c_[t["% detectado"], t["MNPed (mediana)"].map(lambda v: f"> {r['h']}" if not np.isfinite(v) else f"{v:.0f}")],
                hovertemplate="bias %{x:+g}" + unit + ": MNPed %{customdata[1]} · detectado %{customdata[0]:.0f}%",
            ))
        fig.add_hline(y=r0["h"], line_dash="dot", line_color=MUTED, annotation_text="horizonte (não detectado)")
        fig.update_yaxes(type="log", title="MNPed: resultados até o alarme (escala log)",
                         tickvals=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000])
        fig.update_xaxes(title=f"bias injetado ({'%' if kind_k == '%' else 'unidade'})", zeroline=True, zerolinecolor="#cfcfcb")
        st.plotly_chart(base_layout(fig, 440, title="Curva de detecção de erro (menor = mais rápido)", hovermode="closest"),
                        width="stretch")

        if len(results) > 1:
            comp = []
            for s_key, r in results.items():
                row = {"estatística": mm.STAT_LABELS[s_key], "% pontos em alarme (sem bias)": round(r["fa"]["% pontos em alarme"], 2),
                       "episódios sem bias": r["fa"]["episódios de alarme"]}
                for _, t in r["tab"].iterrows():
                    v = t["MNPed (mediana)"]
                    row[f"MNPed {t['bias']:+g}{unit}"] = f"> {r['h']}" if not np.isfinite(v) else f"{v:.0f}"
                comp.append(row)
            st.markdown("**Comparação (MNPed por bias)**")
            st.dataframe(pd.DataFrame(comp), hide_index=True, width="stretch")
            st.caption("Na comparação, cada estatística usa limites alvo ± k·DP da própria série, para que todas partam "
                       "de uma taxa de falso alarme comparável.")

        # Visualizar uma injeção
        with st.expander("Ver uma injeção de bias no gráfico"):
            a, b = st.columns(2)
            b_one = a.selectbox("Bias", biases, format_func=lambda v: f"{v:+g}{unit}")
            pos_pct = b.slider("Início do erro (% da série)", 5, 95, 50)
            p0 = int(len(xv) * pos_pct / 100)
            v2 = xv.copy()
            v2[p0:] = mm._biased(xv[p0:], b_one, kind_k)
            ser_b, _ = mm.compute_series(data2, cfg, vals=v2)
            x0 = data2["t"].iloc[p0] if (has_time and xaxis == "data/hora") else p0 + 1
            nped = mm.simulate_one(ctx, cfg, lim, b_one, kind_k, p0, len(xv))
            st.plotly_chart(series_figure(ser_b, lim, f"Bias {b_one:+g}{unit} a partir do resultado {p0 + 1}",
                                          bias_start=x0), width="stretch")
            st.caption(f"Primeiro alarme no sentido do erro após **{nped} resultados**." if nped else
                       "Sem alarme no sentido do erro até o fim dos dados.")

        # Exportação
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            mm.config_table(cfg, lim, {
                "arquivo": up.name, "aba": sheet, "coluna resultado": c_res,
                "origem do alvo": tgt_mode, "origem do DP": sd_mode, "fonte do truncamento": trunc_src, "fonte do alvo": tgt_src, "fonte do DP": sd_src,
                "exclusão de outliers": f"{tr_label}, k={k_tukey}, iterativo={iterative}" if use_out else "não",
                "falso alarme (% pontos)": round(fa["% pontos em alarme"], 3),
                "episódios de alarme": fa["episódios de alarme"], "tipo de bias": kind,
                "simulações por bias": n_sims, "horizonte": r0["h"], "semente": seed,
            }).to_excel(xw, sheet_name="Configuração", index=False)
            for s_key, r in results.items():
                r["tab"].to_excel(xw, sheet_name=f"Bias {s_key}", index=False)
            ser.drop(columns=["j"]).to_excel(xw, sheet_name="Série", index=False)
        st.download_button("Baixar configuração e resultados (.xlsx)", buf.getvalue(),
                           file_name=f"medida_movel_{stat}_{N}-{step}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
