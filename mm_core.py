"""
Núcleo de cálculo do configurador de medidas móveis (PBRTQC).

Separado da interface para poder ser testado e reutilizado em scripts.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, asdict, replace

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from scipy.signal import lfilter
from scipy.special import inv_boxcox

STAT_LABELS = {
    "MA": "Média móvel",
    "MM": "Mediana móvel",
    "WMA": "Média móvel ponderada",
    "EWMA": "Média móvel exponencialmente ponderada (EWMA)",
}


# =============================================================================
# Leitura e preparo
# =============================================================================
def list_sheets(file_bytes: bytes, name: str) -> list[str]:
    if name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names
    return []


def read_table(file_bytes: bytes, name: str, sheet: str | None = None) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet or 0)
    else:
        head = file_bytes[:20000].decode("utf-8", errors="replace").splitlines()[0] if file_bytes else ""
        # ';' e tab têm prioridade: vírgula costuma aparecer em decimais e em nomes de coluna
        sep = ";" if ";" in head else ("\t" if "\t" in head else ",")
        df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, encoding_errors="replace")
    df.columns = [str(c) for c in df.columns]
    return df


def parse_numeric(s: pd.Series) -> pd.Series:
    """Converte para número aceitando vírgula decimal. Textos (ex.: '<0,5') viram NaN."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    txt = s.astype(str).str.strip().str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(txt, errors="coerce")


_AGE_RE = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*([a-zA-Zçê]*)\s*$")
_AGE_UNIT = {"": 1.0, "a": 1.0, "y": 1.0, "ano": 1.0, "anos": 1.0,
             "m": 1 / 12, "mes": 1 / 12, "mês": 1 / 12, "meses": 1 / 12,
             "d": 1 / 365.25, "dia": 1 / 365.25, "dias": 1 / 365.25,
             "s": 7 / 365.25, "w": 7 / 365.25, "sem": 7 / 365.25}


def parse_age(s: pd.Series) -> pd.Series:
    """Idade em anos. Aceita números e textos como '11M' (meses), '15D' (dias), '2A'."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)

    def conv(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float, np.number)):
            return float(v)
        m = _AGE_RE.match(str(v))
        if not m:
            return np.nan
        unit = _AGE_UNIT.get(m.group(2).lower())
        if unit is None:
            return np.nan
        return float(m.group(1).replace(",", ".")) * unit

    return s.map(conv).astype(float)


def build_datetime(df: pd.DataFrame, date_col: str | None, time_col: str | None) -> pd.Series:
    if not date_col:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    d = df[date_col]
    if not pd.api.types.is_datetime64_any_dtype(d):
        d = pd.to_datetime(d, errors="coerce", dayfirst=True, format="mixed")
    if not time_col:
        return d
    t = df[time_col]
    if pd.api.types.is_datetime64_any_dtype(t):
        td = t - t.dt.normalize()
    else:
        td = pd.to_timedelta(t.astype(str).str.strip(), errors="coerce")
        frac = pd.to_numeric(t, errors="coerce")  # Excel: fração do dia
        td = td.fillna(pd.to_timedelta(frac, unit="D"))
    return d.dt.normalize() + td


@dataclass
class ColumnMap:
    result: str
    date: str | None = None
    time: str | None = None
    age: str | None = None
    sex: str | None = None


def prepare_dataset(df: pd.DataFrame, cmap: ColumnMap) -> tuple[pd.DataFrame, dict]:
    """Dataset padronizado (row, val, t, age, sex) em ordem cronológica + relatório de exclusões."""
    out = pd.DataFrame(index=df.index)
    out["row"] = np.arange(len(df)) + 2  # linha no Excel (cabeçalho = linha 1)
    out["val"] = parse_numeric(df[cmap.result])
    out["t"] = build_datetime(df, cmap.date, cmap.time)
    out["age"] = parse_age(df[cmap.age]) if cmap.age else np.nan
    out["sex"] = df[cmap.sex].astype("string").str.strip() if cmap.sex else pd.NA

    rep = {"linhas_lidas": len(out), "sem_resultado_numerico": int(out["val"].isna().sum())}
    out = out[out["val"].notna()]
    if cmap.date:
        rep["sem_data_hora"] = int(out["t"].isna().sum())
        out = out[out["t"].notna()].sort_values(["t", "row"], kind="stable")
    if cmap.age:
        rep["idade_ilegivel"] = int(out["age"].isna().sum())
    rep["linhas_validas"] = len(out)
    return out.reset_index(drop=True), rep


# =============================================================================
# Etapa 1 — outliers
# =============================================================================
def describe(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    return {
        "n": len(x), "média": x.mean(), "DP": x.std(ddof=1), "CV%": 100 * x.std(ddof=1) / x.mean(),
        "mediana": np.median(x), "P2,5": np.percentile(x, 2.5), "P97,5": np.percentile(x, 97.5),
        "mín": x.min(), "máx": x.max(), "assimetria": stats.skew(x),
    }


def transform(x: np.ndarray, method: str):
    """Retorna (y, inversa, info). method: 'none' | 'log' | 'boxcox'."""
    x = np.asarray(x, float)
    if method == "none":
        return x.copy(), (lambda y: np.asarray(y, float)), {}
    shift = 0.0 if x.min() > 0 else 1.0 - x.min()
    if method == "log":
        return np.log(x + shift), (lambda y: np.exp(y) - shift), {"deslocamento": shift}
    if method == "boxcox":
        y, lam = stats.boxcox(x + shift)

        def inv(v):
            with np.errstate(all="ignore"):
                return inv_boxcox(np.asarray(v, float), lam) - shift

        return y, inv, {"lambda": float(lam), "deslocamento": shift}
    raise ValueError(method)


def tukey_mask(y: np.ndarray, k: float = 1.5, iterative: bool = False):
    """Máscara dos valores mantidos e cercas de Tukey (Q1 - k·IQR, Q3 + k·IQR)."""
    mask = np.ones(len(y), bool)
    while True:
        q1, q3 = np.percentile(y[mask], [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        new = mask & (y >= lo) & (y <= hi)
        changed = new.sum() != mask.sum()
        mask = new
        if not iterative or not changed:
            return mask, lo, hi


def outlier_analysis(x: np.ndarray, method: str = "boxcox", k: float = 1.5, iterative: bool = False) -> dict:
    x = np.asarray(x, float)
    y, inv, info = transform(x, method)
    mask, lo_t, hi_t = tukey_mask(y, k, iterative)
    lo, hi = inv([lo_t])[0], inv([hi_t])[0]
    # Cerca fora do domínio da inversa (Box-Cox) → sem limite naquele lado
    lo = float(lo) if np.isfinite(lo) else -np.inf
    hi = float(hi) if np.isfinite(hi) else np.inf
    return {"y": y, "keep": mask, "lo_t": lo_t, "hi_t": hi_t, "lo": lo, "hi": hi,
            "n_out_low": int((x < lo).sum()) if np.isfinite(lo) else 0,
            "n_out_high": int((x > hi).sum()) if np.isfinite(hi) else 0,
            "n_out": int((~mask).sum()), **info}


def age_profile(d: pd.DataFrame, width: float = 1.0, stat: str = "mean", by_sex: bool = False) -> pd.DataFrame:
    g = d[d["age"].notna()].copy()
    g["faixa"] = np.floor(g["age"] / width) * width
    keys = ["faixa", "sex"] if by_sex else ["faixa"]
    res = g.groupby(keys, dropna=False)["val"].agg(n="size", valor=stat).reset_index()
    return res


def hour_profile(d: pd.DataFrame, stat: str = "mean") -> pd.DataFrame:
    g = d[d["t"].notna()].copy()
    g["hora"] = g["t"].dt.hour
    return g.groupby("hora")["val"].agg(n="size", valor=stat).reset_index()


# =============================================================================
# Etapa 2 — configuração da medida móvel
# =============================================================================
@dataclass
class MMConfig:
    stat: str = "MA"             # MA | MM | WMA | EWMA
    N: int = 50                  # tamanho do bloco
    step: int = 1                # novo ponto a cada n resultados elegíveis
    lam: float | None = None     # EWMA; None → 2/(N+1)
    trunc_lo: float = -np.inf
    trunc_hi: float = np.inf
    trunc_mode: str = "exclude"  # exclude | winsor
    sexes: tuple | None = None   # None = todos
    age_min: float | None = None
    age_max: float | None = None
    k: float = 2.0
    target: float | None = None  # None → média da série
    sd: float | None = None      # None → DP da série
    consec: int = 1              # nº de pontos consecutivos fora para alarmar

    @property
    def lam_eff(self) -> float:
        return self.lam if self.lam else 2.0 / (self.N + 1)


def population_mask(d: pd.DataFrame, cfg: MMConfig) -> np.ndarray:
    m = np.ones(len(d), bool)
    if cfg.sexes is not None:
        m &= d["sex"].isin(list(cfg.sexes)).to_numpy()
    if cfg.age_min is not None:
        m &= (d["age"] >= cfg.age_min).to_numpy()
    if cfg.age_max is not None:
        m &= (d["age"] <= cfg.age_max).to_numpy()
    return m


def apply_truncation(v: np.ndarray, pop: np.ndarray, cfg: MMConfig):
    """Retorna (elegíveis, valores usados)."""
    if cfg.trunc_mode == "winsor":
        return pop.copy(), np.clip(v, cfg.trunc_lo, cfg.trunc_hi)
    return pop & (v >= cfg.trunc_lo) & (v <= cfg.trunc_hi), v


def _windows(hist: np.ndarray, new: np.ndarray, cfg: MMConfig, ewma_prev: float | None) -> np.ndarray:
    """Estatística para cada valor de `new`, dado o histórico imediatamente anterior."""
    if len(new) == 0:
        return np.empty(0)
    if cfg.stat == "EWMA":
        lam = cfg.lam_eff
        return lfilter([lam], [1.0, -(1.0 - lam)], new, zi=[(1.0 - lam) * ewma_prev])[0]
    N = cfg.N
    comb = np.concatenate([hist[len(hist) - (N - 1):] if N > 1 else hist[:0], new])
    w = sliding_window_view(comb, N)
    if cfg.stat == "MA":
        return w.mean(axis=1)
    if cfg.stat == "MM":
        return np.median(w, axis=1)
    if cfg.stat == "WMA":
        wt = np.arange(1, N + 1, dtype=float)
        return w @ (wt / wt.sum())
    raise ValueError(cfg.stat)


def rolling_stat(e: np.ndarray, cfg: MMConfig) -> np.ndarray:
    """Estatística ao fim de cada valor elegível (NaN até completar o primeiro bloco)."""
    out = np.full(len(e), np.nan)
    N = cfg.N
    if len(e) < N:
        return out
    if cfg.stat == "EWMA":
        z0 = e[:N].mean()  # inicialização: média do 1º bloco
        out[N - 1] = z0
        out[N:] = _windows(e[:N], e[N:], cfg, z0)
    else:
        out[N - 1:] = _windows(e[:N - 1], e[N - 1:], cfg, None)
    return out


def eval_mask(n_elig: int, cfg: MMConfig, offset: int = 0) -> np.ndarray:
    """Índices elegíveis (offset + i) que geram ponto: bloco completo e passo de n resultados."""
    j = np.arange(n_elig) + offset
    return (j >= cfg.N - 1) & ((j - (cfg.N - 1)) % cfg.step == 0)


@dataclass
class Context:
    """Tudo o que a simulação precisa reaproveitar da série sem viés."""
    vals: np.ndarray       # resultados brutos do fluxo (já sem outliers, se escolhido)
    pop: np.ndarray        # máscara de população (sexo/idade)
    elig_pos: np.ndarray   # posições no fluxo dos resultados elegíveis
    e: np.ndarray          # valores elegíveis usados no cálculo
    stat_e: np.ndarray     # estatística em cada elegível
    n_stream: int


def compute_series(d: pd.DataFrame, cfg: MMConfig, vals: np.ndarray | None = None) -> tuple[pd.DataFrame, Context]:
    v = d["val"].to_numpy(float) if vals is None else vals
    pop = population_mask(d, cfg)
    elig, used = apply_truncation(v, pop, cfg)
    pos = np.flatnonzero(elig)
    e = used[pos]
    st = rolling_stat(e, cfg)
    em = eval_mask(len(e), cfg)
    ser = pd.DataFrame({
        "j": np.flatnonzero(em), "pos": pos[em], "valor": st[em],
        "t": d["t"].to_numpy()[pos[em]] if len(pos) else [],
    })
    ser["seq"] = ser["pos"] + 1
    return ser, Context(v, pop, pos, e, st, len(v))


def control_limits(ser: pd.DataFrame, cfg: MMConfig) -> dict:
    s = ser["valor"].to_numpy()
    target = cfg.target if cfg.target is not None else float(np.mean(s))
    sd = cfg.sd if cfg.sd is not None else float(np.std(s, ddof=1))
    return {"alvo": target, "DP": sd, "LI": target - cfg.k * sd, "LS": target + cfg.k * sd}


def run_alarm(out: np.ndarray, consec: int) -> np.ndarray:
    """Alarme quando há >= consec pontos consecutivos fora (mesmo lado)."""
    if consec <= 1:
        return out.copy()
    run, r = np.zeros(len(out), int), 0
    for i, o in enumerate(out):
        r = r + 1 if o else 0
        run[i] = r
    return run >= consec


def alarm_flags(values: np.ndarray, lim: dict, consec: int) -> np.ndarray:
    hi = run_alarm(values > lim["LS"], consec)
    lo = run_alarm(values < lim["LI"], consec)
    return hi | lo


def false_alarm_summary(ser: pd.DataFrame, lim: dict, cfg: MMConfig, n_stream: int) -> dict:
    a = alarm_flags(ser["valor"].to_numpy(), lim, cfg.consec)
    events = int((a & ~np.r_[False, a[:-1]]).sum())  # início de cada episódio de alarme
    days = np.nan
    if ser["t"].notna().any():
        span = (ser["t"].max() - ser["t"].min()) / pd.Timedelta(days=1)
        days = max(span, 1e-9)
    return {
        "pontos": len(a), "pontos em alarme": int(a.sum()),
        "% pontos em alarme": 100 * a.mean() if len(a) else np.nan,
        "episódios de alarme": events,
        "episódios / 1000 resultados": 1000 * events / n_stream if n_stream else np.nan,
        "episódios / dia": events / days if np.isfinite(days) else np.nan,
    }


# =============================================================================
# Simulação de injeção de bias
# =============================================================================
def _biased(v: np.ndarray, bias: float, kind: str) -> np.ndarray:
    return v * (1 + bias / 100.0) if kind == "%" else v + bias


def start_positions(ctx: Context, cfg: MMConfig, n_sims: int, horizon: int, seed: int):
    """Posições de início do erro (mesmas para todos os vieses → comparação pareada)."""
    need = cfg.N  # histórico completo antes do erro
    if len(ctx.elig_pos) <= need + 1:
        raise ValueError("Poucos resultados elegíveis para simular com esse tamanho de bloco.")
    p_min = int(ctx.elig_pos[need - 1]) + 1
    p_max = ctx.n_stream - 1 - horizon
    if p_max < p_min:  # dados curtos: encurta o horizonte
        horizon = (ctx.n_stream - 1 - p_min) // 2
        p_max = ctx.n_stream - 1 - horizon
    if horizon < 1:
        raise ValueError("Dados insuficientes para simular.")
    rng = np.random.default_rng(seed)
    return rng.integers(p_min, p_max + 1, size=n_sims), horizon


def simulate_one(ctx: Context, cfg: MMConfig, lim: dict, bias: float, kind: str, p0: int, horizon: int):
    """Nº de resultados do fluxo (a partir do início do erro, inclusive) até o alarme; None se não detectou."""
    m0 = int(np.searchsorted(ctx.elig_pos, p0))   # elegíveis antes do erro
    hist = ctx.e[max(0, m0 - (cfg.N - 1)):m0]
    ewma_prev = ctx.stat_e[m0 - 1] if cfg.stat == "EWMA" else None
    p_end = min(ctx.n_stream, p0 + horizon)
    seg = _biased(ctx.vals[p0:p_end], bias, kind)
    elig, used = apply_truncation(seg, ctx.pop[p0:p_end], cfg)
    newpos = np.flatnonzero(elig) + p0
    sv = _windows(hist, used[elig], cfg, ewma_prev)
    em = eval_mask(len(sv), cfg, offset=m0)
    sv, sp = sv[em], newpos[em]
    out = sv > lim["LS"] if bias > 0 else sv < lim["LI"]  # só conta alarme no sentido do erro
    a = run_alarm(out, cfg.consec)
    idx = np.flatnonzero(a)
    return int(sp[idx[0]] - p0 + 1) if len(idx) else None


def _q(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q, method="inverted_cdf"))


def simulate_bias(ctx: Context, cfg: MMConfig, lim: dict, biases, kind: str = "%",
                  n_sims: int = 200, horizon: int = 1000, seed: int = 42):
    starts, horizon = start_positions(ctx, cfg, n_sims, horizon, seed)
    rows, raw = [], {}
    for b in biases:
        if b == 0:
            continue
        r = [simulate_one(ctx, cfg, lim, b, kind, int(p0), horizon) for p0 in starts]
        det = np.array([x for x in r if x is not None], float)
        full = np.array([x if x is not None else np.inf for x in r], float)
        raw[b] = full
        rows.append({
            "bias": b, "simulações": len(r), "% detectado": 100 * len(det) / len(r),
            "ANPed (média, detectados)": det.mean() if len(det) else np.nan,
            # quantis sem interpolação: não detectados (= infinito) entram como "> horizonte"
            "MNPed (mediana)": _q(full, 0.50),
            "P25": _q(full, 0.25), "P75": _q(full, 0.75), "P95": _q(full, 0.95),
        })
    return pd.DataFrame(rows), raw, horizon


def config_table(cfg: MMConfig, lim: dict, extra: dict | None = None) -> pd.DataFrame:
    d = asdict(cfg)
    d["estatística"] = STAT_LABELS[cfg.stat]
    if cfg.stat == "EWMA":
        d["lambda efetivo"] = cfg.lam_eff
    d.update({f"limite {k}": v for k, v in lim.items()})
    if extra:
        d.update(extra)
    return pd.DataFrame({"parâmetro": list(d.keys()), "valor": [str(v) for v in d.values()]})


__all__ = [n for n in dir() if not n.startswith("_")] + ["replace"]
