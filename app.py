import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import scipy.stats as sps
from statsmodels.api import OLS, add_constant
from typing import Tuple

# ======================================
# Generic helpers
# ======================================

def file_scope_key(suffix: str, uploaded_file) -> str:
    base = uploaded_file.name if uploaded_file else "DEFAULT_FILE"
    return f"{base}__{suffix}"

def safe_std(x):
    x = np.asarray(x, dtype=float)
    return np.nan if x.size == 0 else np.nanstd(x, ddof=1)

def safe_mean(x):
    x = np.asarray(x, dtype=float)
    return np.nan if x.size == 0 else np.nanmean(x)

def safe_div(a, b):
    if b is None or (isinstance(b, float) and np.isnan(b)) or b == 0:
        return np.nan
    return a / b

def is_numeric_series(s: pd.Series) -> bool:
    try:
        pd.to_numeric(s.dropna().head(5))
        return True
    except Exception:
        return False

def normalize_cols(cols):
    out = []
    for c in cols:
        s = "" if c is None else str(c)
        s = " ".join(s.strip().split())
        out.append(s)
    return out

def resolve_col(selection: str, columns: list[str]) -> str | None:
    if selection is None:
        return None
    columns_norm = normalize_cols(columns)
    sel_norm = " ".join(str(selection).strip().split())
    if sel_norm in columns_norm:
        return columns[columns_norm.index(sel_norm)]
    sel_cf = sel_norm.casefold()
    for i, c in enumerate(columns_norm):
        if c.casefold() == sel_cf:
            return columns[i]
    sel_compact = "".join(sel_norm.split()).casefold()
    for i, c in enumerate(columns_norm):
        if "".join(c.split()).casefold() == sel_compact:
            return columns[i]
    return None

def assert_in_df(df: pd.DataFrame, col: str | None, label: str):
    if col is None or col not in df.columns:
        st.error(f"❌ {label} column not found: `{col}`.\n\n**Available columns:** {list(df.columns)}")
        st.stop()

def detect_return_scale(r: pd.Series) -> bool:
    r = pd.to_numeric(pd.Series(r).dropna(), errors="coerce")
    if r.empty:
        return False
    med_abs = float(np.median(np.abs(r)))
    max_abs = float(np.max(np.abs(r)))
    return (0.5 <= med_abs <= 100) or (2 < max_abs < 200)

def reset_with_date(frame: pd.DataFrame) -> pd.DataFrame:
    tmp = frame.reset_index()
    dt_candidates = [c for c in tmp.columns if pd.api.types.is_datetime64_any_dtype(tmp[c])]
    dcol = dt_candidates[0] if dt_candidates else tmp.columns[0]
    tmp[dcol] = pd.to_datetime(tmp[dcol], errors="coerce")
    if dcol != "Date":
        tmp = tmp.rename(columns={dcol: "Date"})
    return tmp

def guess_date_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if 'date' in c.lower():
            return c
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    best_c, best_score = None, -1
    for c in df.columns:
        score = pd.to_datetime(df[c], errors="coerce").notna().sum()
        if score > best_score:
            best_c, best_score = c, score
    return best_c or df.columns[0]

def guess_price_col(cols):
    prefer = ["adj close", "adj_close", "close", "close price", "price", "last", "px_last"]
    low = [c.lower() for c in cols]
    for name in prefer:
        if name in low:
            return cols[low.index(name)]
    return None

def first_numeric(cols, df):
    for c in cols:
        if is_numeric_series(df[c]):
            return c
    return None

def clamp_pct(x):
    if x is None or np.isnan(x):
        return np.nan
    return float(np.minimum(1.0, np.maximum(0.0, x)))

# ======================================
# Black–Scholes
# ======================================
def black_scholes(S, K, r, sigma, T, option_type="call"):
    try:
        S, K, r, sigma, T = map(float, [S, K, r, sigma, T])
        if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
            return (np.nan,)*6
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            price = S * sps.norm.cdf(d1) - K * np.exp(-r * T) * sps.norm.cdf(d2)
            rho =  K * T * np.exp(-r * T) * sps.norm.cdf(d2)
            delta = sps.norm.cdf(d1)
        else:
            price = K * np.exp(-r * T) * sps.norm.cdf(-d2) - S * sps.norm.cdf(-d1)
            rho = -K * T * np.exp(-r * T) * sps.norm.cdf(-d2)
            delta = -sps.norm.cdf(-d1)
        gamma = sps.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S * sps.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * (sps.norm.cdf(d2) if option_type=="call" else sps.norm.cdf(-d2)))
        vega = S * sps.norm.pdf(d1) * np.sqrt(T)
        return price, delta, gamma, theta, vega, rho
    except Exception:
        return (np.nan,)*6

# ======================================
# VaR helpers (per your spec)
# ======================================
def k_from_conf(conf: float) -> float:
    if conf >= 0.99: return 0.01
    if conf >= 0.95: return 0.05
    return 0.10

def z_from_conf(conf: float) -> float:
    if conf >= 0.99: return 2.33
    if conf >= 0.95: return 1.65
    return 1.28

def historical_var_pct_daily(returns: pd.Series, conf: float) -> float:
    r = pd.Series(returns).dropna().astype(float).values
    if r.size == 0: return np.nan
    k = k_from_conf(conf) * 100.0
    q = np.percentile(r, k)
    return clamp_pct(max(0.0, -q))

def analytical_var_pct_daily(returns: pd.Series, conf: float) -> float:
    r = pd.Series(returns).dropna().astype(float).values
    if r.size == 0: return np.nan
    z = z_from_conf(conf)
    sd = float(np.nanstd(r, ddof=1))
    return clamp_pct(max(0.0, z * sd))

def mc_var_pct_daily(returns: pd.Series, conf: float, n_sims: int, seed: int | None = None) -> float:
    r = pd.Series(returns).dropna().astype(float).values
    if r.size == 0 or n_sims < 100: return np.nan
    if seed is not None: np.random.seed(int(seed))
    mu = float(np.nanmean(r)); sd = float(np.nanstd(r, ddof=1))
    u = np.random.rand(int(n_sims))
    sims = sps.norm.ppf(u, loc=mu, scale=sd)
    k = k_from_conf(conf) * 100.0
    q = np.percentile(sims, k)
    return clamp_pct(max(0.0, -q))

# ======================================
# Data loading (strict policy)
# ======================================
DEFAULT_FILE = "icici dashboard data.xlsx"

def load_data(uploaded_file) -> Tuple[pd.DataFrame, str, bool]:
    if uploaded_file:
        try:
            xls = pd.ExcelFile(uploaded_file)
            return pd.read_excel(xls, sheet_name=xls.sheet_names[0]), uploaded_file.name.split(".")[0], False
        except Exception as e:
            st.error(f"Failed to read the uploaded workbook: {e}"); st.stop()
    else:
        try:
            df = pd.read_excel(DEFAULT_FILE)
            return df, "ICICI", True
        except FileNotFoundError:
            st.error(f"Default dataset `{DEFAULT_FILE}` not found. Add it to the repo or upload a workbook."); st.stop()
        except Exception as e:
            st.error(f"Failed to read `{DEFAULT_FILE}`: {e}"); st.stop()

# ======================================
# App
# ======================================
st.set_page_config(page_title="Risk & Portfolio Dashboard", layout="wide")
st.title("📊 Risk & Portfolio Dashboard")

# ---------- Upload ----------
uploaded_file = st.file_uploader("Upload Excel (any stock + optional benchmark)", type=["xlsx"])
raw, default_name_asset, using_default = load_data(uploaded_file)

# If uploaded has multiple sheets, allow selection
if uploaded_file:
    try:
        xls = pd.ExcelFile(uploaded_file)
        if len(xls.sheet_names) > 1:
            sheet = st.sidebar.selectbox("Worksheet", options=xls.sheet_names, index=0,
                                         key=file_scope_key("sheet", uploaded_file))
            raw = pd.read_excel(xls, sheet_name=sheet)
    except Exception as e:
        st.error(f"Failed to enumerate sheets: {e}"); st.stop()

# Normalize column names
raw.columns = normalize_cols(raw.columns)
raw_cols = list(raw.columns)

# ---------- Column Inspector ----------
with st.expander("🔎 Column Inspector"):
    st.write("**Detected columns:**", raw_cols)
    st.dataframe(raw.head(10), use_container_width=True)

# ---------- Heuristic guesses ----------
date_guess = guess_date_col(raw)
numeric_cols = [c for c in raw_cols if is_numeric_series(raw[c])]
price_guess = guess_price_col(numeric_cols) or (first_numeric(numeric_cols, raw))
bench_guess = None
if price_guess and len(numeric_cols) > 1:
    others = [c for c in numeric_cols if c != price_guess]
    bench_guess = guess_price_col(others) or (others[0] if others else None)
ret_guess = next((c for c in raw_cols if 'return' in c.lower() or '%' in c.lower()), "<None>")
bench_ret_guess = "<None>"

# ---------- Column mapping ----------
st.sidebar.header("🔧 Column Mapping")
date_sel = st.sidebar.selectbox("Date column",
                                options=raw_cols,
                                index=(raw_cols.index(date_guess) if date_guess in raw_cols else 0),
                                key=file_scope_key("date_col", uploaded_file))
asset_price_sel = st.sidebar.selectbox("Asset Price column",
                                       options=numeric_cols,
                                       index=(numeric_cols.index(price_guess) if price_guess in numeric_cols else 0),
                                       key=file_scope_key("asset_price_col", uploaded_file))
bench_price_options = ["<None>"] + numeric_cols
bench_default_idx = 0
if bench_guess and bench_guess in numeric_cols:
    bench_default_idx = bench_price_options.index(bench_guess) if bench_guess in bench_price_options else 0
bench_price_sel = st.sidebar.selectbox("Benchmark Price column (optional)",
                                       options=bench_price_options,
                                       index=bench_default_idx,
                                       key=file_scope_key("bench_price_col", uploaded_file))

asset_ret_options = ["<None>"] + raw_cols
asset_ret_default_idx = (asset_ret_options.index(ret_guess) if ret_guess in asset_ret_options else 0)
asset_ret_sel = st.sidebar.selectbox("Asset Return column (optional)",
                                     options=asset_ret_options,
                                     index=asset_ret_default_idx,
                                     key=file_scope_key("asset_ret_col", uploaded_file))

bench_ret_options = ["<None>"] + raw_cols
bench_ret_default_idx = (bench_ret_options.index(bench_ret_guess) if bench_ret_guess in bench_ret_options else 0)
bench_ret_sel = st.sidebar.selectbox("Benchmark Return column (optional)",
                                     options=bench_ret_options,
                                     index=bench_ret_default_idx,
                                     key=file_scope_key("bench_ret_col", uploaded_file))

asset_name = st.sidebar.text_input("Display name: Asset",
                                   value=default_name_asset,
                                   key=file_scope_key("asset_name", uploaded_file))
bench_name_default = "Benchmark" if bench_price_sel != "<None>" else ""
bench_name = st.sidebar.text_input("Display name: Benchmark",
                                   value=bench_name_default,
                                   key=file_scope_key("bench_name", uploaded_file))

# Resolve selections
date_col = resolve_col(date_sel, raw_cols)
asset_price_col = resolve_col(asset_price_sel, raw_cols)
benchmark_price_col = None if bench_price_sel == "<None>" else resolve_col(bench_price_sel, raw_cols)
asset_ret_col = None if asset_ret_sel == "<None>" else resolve_col(asset_ret_sel, raw_cols)
benchmark_ret_col = None if bench_ret_sel == "<None>" else resolve_col(bench_ret_sel, raw_cols)

# ---------- Build indexed frame ----------
assert_in_df(raw, date_col, "Date")
assert_in_df(raw, asset_price_col, "Asset Price")
if benchmark_price_col is not None: assert_in_df(raw, benchmark_price_col, "Benchmark Price")
if asset_ret_col is not None:      assert_in_df(raw, asset_ret_col, "Asset Return")
if benchmark_ret_col is not None:  assert_in_df(raw, benchmark_ret_col, "Benchmark Return")

df = raw.copy()
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True).set_index(date_col)

# ---------- Filters & Params ----------
st.sidebar.header("Filters")
min_d, max_d = df.index.min(), df.index.max()
date_range = st.sidebar.date_input("Select Date Range", [min_d, max_d],
                                   key=file_scope_key("daterange", uploaded_file))
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df.loc[start_d:end_d]
else:
    st.warning("Invalid date range; using full data.")
if df.empty:
    st.warning("No data in the selected range. Adjust filters."); st.stop()

st.sidebar.subheader("Global Parameters")
risk_free = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0,
                              key=file_scope_key("rf", uploaded_file)) / 100.0
time_horizon = st.sidebar.slider("Time Horizon (Years) [for terminal MC]", 0.1, 5.0, 1.0,
                                 key=file_scope_key("horizon", uploaded_file))
conf = st.sidebar.select_slider("VaR Confidence Level",
                                options=[0.90, 0.95, 0.99], value=0.95,
                                key=file_scope_key("var_conf", uploaded_file),
                                format_func=lambda x: f"{int(x*100)}%")

# ---------- Working frame ----------
ndf = pd.DataFrame(index=df.index)
ndf["Asset_Price"] = pd.to_numeric(df[asset_price_col], errors="coerce")
ndf["Asset_Return"] = (pd.to_numeric(df[asset_ret_col], errors="coerce")
                       if asset_ret_col is not None else ndf["Asset_Price"].pct_change())

auto_pct = detect_return_scale(ndf["Asset_Return"])
with st.sidebar.expander("Return Scaling (Asset)"):
    st.checkbox("Treat Asset Return column as % (divide by 100)",
                value=auto_pct, key=file_scope_key("asset_pct_flag", uploaded_file))
if st.session_state.get(file_scope_key("asset_pct_flag", uploaded_file), False):
    ndf["Asset_Return"] = ndf["Asset_Return"] / 100.0

if benchmark_price_col is not None:
    ndf["Bench_Price"] = pd.to_numeric(df[benchmark_price_col], errors="coerce")
    ndf["Bench_Return"] = (pd.to_numeric(df[benchmark_ret_col], errors="coerce")
                           if benchmark_ret_col is not None else ndf["Bench_Price"].pct_change())
    auto_pct_b = detect_return_scale(ndf["Bench_Return"])
    with st.sidebar.expander("Return Scaling (Benchmark)"):
        st.checkbox("Treat Benchmark Return column as % (divide by 100)",
                    value=auto_pct_b, key=file_scope_key("bench_pct_flag", uploaded_file))
    if st.session_state.get(file_scope_key("bench_pct_flag", uploaded_file), False):
        ndf["Bench_Return"] = ndf["Bench_Return"] / 100.0
else:
    ndf["Bench_Price"] = np.nan
    ndf["Bench_Return"] = np.nan

# ======================================
# 1) Performance
# ======================================
st.header("1️⃣ Performance Analysis")

df_reset = reset_with_date(ndf)
price_cols = ["Asset_Price"] + (["Bench_Price"] if ndf["Bench_Price"].notna().any() else [])
rename_map = {"Asset_Price": asset_name}
if "Bench_Price" in price_cols: rename_map["Bench_Price"] = bench_name or "Benchmark"

fig1 = px.line(df_reset, x="Date", y=price_cols,
               title=f"{asset_name}" + (f" vs {bench_name}" if "Bench_Price" in price_cols else "") + " — Prices")
for tr in fig1.data:
    tr.name = rename_map.get(tr.name, tr.name)
st.plotly_chart(fig1, use_container_width=True)

cum_df = pd.DataFrame({"Date": df_reset["Date"]})
cum_df[f"{asset_name}_CumRet"] = (1 + ndf["Asset_Return"].fillna(0)).cumprod().values - 1
if ndf["Bench_Return"].notna().any():
    cum_df[f"{bench_name or 'Benchmark'}_CumRet"] = (1 + ndf["Bench_Return"].fillna(0)).cumprod().values - 1
fig_cum = px.line(cum_df, x="Date", y=[c for c in cum_df.columns if c.endswith("_CumRet")], title="Cumulative Returns")
st.plotly_chart(fig_cum, use_container_width=True)

# ==== Return diagnostics (after scaling) for BOTH ====
st.subheader("Return diagnostics (after scaling) — Asset & Benchmark")
diag_asset = pd.DataFrame({
    "Metric": ["Mean (daily)", "Std (daily)", "Min", "Max"],
    "Value (%)": [
        safe_mean(ndf["Asset_Return"])*100.0,
        safe_std(ndf["Asset_Return"])*100.0,
        pd.Series(ndf["Asset_Return"]).min()*100.0,
        pd.Series(ndf["Asset_Return"]).max()*100.0
    ]
})
cols = st.columns(2)
with cols[0]:
    st.markdown(f"**{asset_name}**")
    st.dataframe(diag_asset.style.format({"Value (%)":"{:.4f}"}), use_container_width=True)

if ndf["Bench_Return"].notna().sum() > 0:
    diag_bench = pd.DataFrame({
        "Metric": ["Mean (daily)", "Std (daily)", "Min", "Max"],
        "Value (%)": [
            safe_mean(ndf["Bench_Return"])*100.0,
            safe_std(ndf["Bench_Return"])*100.0,
            pd.Series(ndf["Bench_Return"]).min()*100.0,
            pd.Series(ndf["Bench_Return"]).max()*100.0
        ]
    })
    with cols[1]:
        st.markdown(f"**{bench_name or 'Benchmark'}**")
        st.dataframe(diag_bench.style.format({"Value (%)":"{:.4f}"}), use_container_width=True)

# ======================================
# 2) Risk–Return
# ======================================
st.header("2️⃣ Risk-Return Analysis")

rf_daily = risk_free / 252.0
excess = (ndf["Asset_Return"] - rf_daily).dropna().values
downside = (ndf["Asset_Return"][ndf["Asset_Return"] < 0] - rf_daily).dropna().values
sharpe = np.sqrt(252.0) * safe_div(safe_mean(excess), safe_std(excess))
sortino = np.sqrt(252.0) * safe_div(safe_mean(excess), safe_std(downside))
st.write(f"**Sharpe Ratio ({asset_name}):** {sharpe:.3f} | **Sortino Ratio:** {sortino:.3f}")

if ndf["Bench_Return"].notna().sum() > 5:
    reg_df = ndf[["Asset_Return", "Bench_Return"]].dropna()
    if not reg_df.empty and reg_df["Bench_Return"].nunique() > 1:
        X = add_constant(reg_df["Bench_Return"].values.astype(float))
        y = reg_df["Asset_Return"].values.astype(float)
        try:
            model = OLS(y, X).fit()
            alpha, beta = float(model.params[0]), float(model.params[1])
            st.write(f"**Alpha ({asset_name} vs {bench_name or 'Benchmark'}):** {alpha:.6f} | **Beta:** {beta:.4f}")
            fig2 = px.scatter(reg_df.reset_index(), x="Bench_Return", y="Asset_Return", trendline="ols",
                              title=f"Regression: {asset_name} vs {bench_name or 'Benchmark'} (daily returns)")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning(f"Regression failed: {e}")
    else:
        st.info("Insufficient variability in benchmark returns for regression.")
else:
    st.info("No benchmark selected/provided → skipping Alpha/Beta.")

# ======================================
# 3) VaR — Daily Loss % (Historical | Analytical | Monte Carlo) — BOTH
# ======================================
st.header("3️⃣ Value at Risk (VaR) — Daily Loss % (Historical | Analytical | Monte Carlo)")
var_sims = st.slider("Monte Carlo simulations (for VaR)", 1_000, 200_000, 20_000, step=1_000,
                     key=file_scope_key("var_mc_sims", uploaded_file))
seed_on = st.checkbox("Fix random seed (VaR MC)", value=True, key=file_scope_key("var_mc_seed_on", uploaded_file))
seed_val = 123 if seed_on else None
k = k_from_conf(conf); z_used = z_from_conf(conf)

def var_panel(title, series):
    colA, colB, colC = st.columns(3)
    hv = historical_var_pct_daily(series, conf)
    av = analytical_var_pct_daily(series, conf)
    mv = mc_var_pct_daily(series, conf, n_sims=var_sims, seed=seed_val)
    with colA: st.markdown("**Historical VaR**"); st.write(f"Loss VaR: {hv:.2%}")
    with colB: st.markdown("**Analytical VaR (Z×SD)**"); st.write(f"Loss VaR: {av:.2%}  \n(Z≈{z_used:.2f})")
    with colC: st.markdown("**Monte Carlo VaR (Normal μ,σ)**"); st.write(f"Loss VaR: {mv:.2%}")
    st.caption(f"{title}: With **{int(conf*100)}% confidence** (k={int(k*100)}%), "
               f"there’s a {int(k*100)}% chance daily loss ≥ VaR.")

st.subheader(f"{asset_name}")
var_panel(asset_name, ndf["Asset_Return"].dropna())

if ndf["Bench_Return"].notna().sum() > 5:
    st.subheader(f"{bench_name or 'Benchmark'}")
    var_panel(bench_name or "Benchmark", ndf["Bench_Return"].dropna())

# ======================================
# 4) Black–Scholes Calculation
# ======================================
st.header("4️⃣ Black–Scholes Calculation")
col1, col2 = st.columns(2)
with col1:
    last_price = ndf["Asset_Price"].dropna().iloc[-1] if ndf["Asset_Price"].dropna().size else 100.0
    S = st.number_input("Spot Price", min_value=0.0, value=float(last_price),
                        key=file_scope_key("bs_S", uploaded_file))
    K = st.number_input("Strike Price", min_value=0.0, value=float(last_price),
                        key=file_scope_key("bs_K", uploaded_file))
with col2:
    sigma_ui = st.number_input("Volatility (σ, decimal)", min_value=0.0, value=0.2,
                               key=file_scope_key("bs_sigma", uploaded_file))
    T = st.number_input("Time to Maturity (Years)", min_value=0.0, value=1.0,
                        key=file_scope_key("bs_T", uploaded_file))
opt_type = st.selectbox("Option Type", ["call", "put"], key=file_scope_key("bs_type", uploaded_file))
price, delta, gamma, theta, vega, rho = black_scholes(S, K, risk_free, sigma_ui, T, option_type=opt_type)
st.write(f"**Price:** {price:.4f} | **Delta:** {delta:.4f} | **Gamma:** {gamma:.6f} | "
         f"**Theta:** {theta:.4f} | **Vega:** {vega:.4f} | **Rho:** {rho:.4f}")

# ======================================
# 5) Asset Liability Management (ALM)
# ======================================
st.header("5️⃣ Asset Liability Management (ALM)")

tab_direct, tab_csv = st.tabs(["🧮 Direct Input (Equity Shock)", "📤 Upload CSV (RSA/RSL + Auto-Durations)"])

with tab_direct:
    colA, colB, colC = st.columns(3)
    with colA:
        A = st.number_input("Total Rate-Sensitive Assets A", min_value=0.0,
                            value=st.session_state.get("alm_A", 1_000_000_000.0),
                            step=1_000_000.0, format="%.2f", key=file_scope_key("alm_A", uploaded_file))
        DA = st.number_input("Duration of Assets DA (years)", min_value=0.0,
                             value=st.session_state.get("alm_DA", 2.0),
                             step=0.1, format="%.2f", key=file_scope_key("alm_DA", uploaded_file))
        CA = st.number_input("Convexity of Assets CA (optional)", min_value=0.0,
                             value=st.session_state.get("alm_CA", 0.0),
                             step=0.1, format="%.4f", key=file_scope_key("alm_CA", uploaded_file))
    with colB:
        L = st.number_input("Total Rate-Sensitive Liabilities L", min_value=0.0,
                            value=st.session_state.get("alm_L", 900_000_000.0),
                            step=1_000_000.0, format="%.2f", key=file_scope_key("alm_L", uploaded_file))
        DL = st.number_input("Duration of Liabilities DL (years)", min_value=0.0,
                             value=st.session_state.get("alm_DL", 1.5),
                             step=0.1, format="%.2f", key=file_scope_key("alm_DL", uploaded_file))
        CL = st.number_input("Convexity of Liabilities CL (optional)", min_value=0.0,
                             value=st.session_state.get("alm_CL", 0.0),
                             step=0.1, format="%.4f", key=file_scope_key("alm_CL", uploaded_file))
    with colC:
        shock_pct = st.number_input("Parallel Yield Shock (Percent, e.g., 1.00 for 1%)",
                                    value=1.00, step=0.25, format="%.2f",
                                    key=file_scope_key("alm_shock_pct", uploaded_file))
        dy = float(shock_pct) / 100.0  # use percent input directly

    E = A - L
    A_safe = A if A > 0 else np.nan
    L_over_A = (L / A_safe) if (A_safe and not np.isnan(A_safe)) else np.nan
    DG = DA - (DL * L_over_A) if not np.isnan(L_over_A) else np.nan

    dE_linear = - DG * A * dy if not (np.isnan(DG) or np.isnan(dy)) else np.nan
    dE_conv = 0.5 * ((CA * A) - (CL * L)) * (dy**2) if (CA > 0 or CL > 0) else 0.0
    dE_total = (0.0 if np.isnan(dE_linear) else dE_linear) + dE_conv
    shock_equity_pct = (dE_total / E * 100.0) if E != 0 else np.nan

    st.markdown("DG = DA − DL × (L/A);  ΔE ≈ − DG × A × Δy  (+ convexity: 0.5 × (CA×A − CL×L) × Δy²)")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Equity E = A − L", f"{E:,.2f}"); st.metric("DA (years)", f"{DA:.2f}")
    with c2: st.metric("DL (years)", f"{DL:.2f}");       st.metric("Duration Gap (DG)", f"{DG:.4f}" if not np.isnan(DG) else "—")
    with c3: st.metric("Δy (Percent)", f"{dy*100:.2f}%"); st.metric("Equity Shock (%)", f"{(shock_equity_pct if not np.isnan(shock_equity_pct) else 0):.2f}%")
    st.write(f"**ΔE (amount):** {dE_total:,.2f} | **Linear:** {0.0 if np.isnan(dE_linear) else dE_linear:,.2f} | **Convexity:** {dE_conv:,.2f}")

    # ---- ΔEVE Sensitivity: now in PERCENT shocks ----
    st.subheader("ΔEVE Sensitivity: Equity Shock % vs Yield Shift (−3.00% … +3.00%)")
    step_pct = st.select_slider("Shock grid resolution (percent)", options=[0.10, 0.25, 0.50, 1.00], value=0.25,
                                key=file_scope_key("alm_step_pct", uploaded_file))
    # Build grid in percent, convert to decimal Δy
    shocks_pct = np.round(np.arange(-3.00, 3.00 + step_pct, step_pct), 2)
    dy_vec = shocks_pct / 100.0
    if not np.isnan(DG) and E != 0 and not np.isnan(E):
        dE_linear_vec = -DG * A * dy_vec
        dE_conv_vec = 0.5 * ((CA * A) - (CL * L)) * (dy_vec ** 2) if (CA > 0 or CL > 0) else 0.0
        dE_total_vec = dE_linear_vec + dE_conv_vec
        eq_shock_pct_vec = (dE_total_vec / E) * 100.0
        sens_df = pd.DataFrame({
            "Shock_%": shocks_pct,
            "Delta_y": dy_vec,
            "EquityShock_%": eq_shock_pct_vec
        })
        st.dataframe(sens_df.style.format({"Shock_%": "{:.2f}", "Delta_y": "{:.4%}", "EquityShock_%": "{:.2f}"}),
                     use_container_width=True)
        fig_sens = px.line(sens_df, x="Shock_%", y="EquityShock_%",
                           title="Equity Shock % vs Parallel Yield Shift (%)")
        fig_sens.update_traces(mode="lines+markers")
        st.plotly_chart(fig_sens, use_container_width=True)
    else:
        st.info("Provide valid A, L, DA, DL (and ensure A ≠ 0) to run the sensitivity sweep.")

with tab_csv:
    st.caption("Minimum columns: **Type** ∈ {Asset, Liability}, **Amount** (numeric). Optional: **Midpoint_Years** for duration.")
    alm_file = st.file_uploader("Upload CSV", type=["csv"], key=file_scope_key("alm_csv", uploaded_file))
    if alm_file:
        alm_df = pd.read_csv(alm_file)
        st.dataframe(alm_df, use_container_width=True)
        cols_norm = {c.strip().lower(): c for c in alm_df.columns}
        if "type" in cols_norm and "amount" in cols_norm:
            type_col = cols_norm["type"]; amt_col = cols_norm["amount"]
            assets = pd.to_numeric(alm_df.loc[alm_df[type_col].str.lower()=="asset", amt_col], errors="coerce")
            liabs  = pd.to_numeric(alm_df.loc[alm_df[type_col].str.lower()=="liability", amt_col], errors="coerce")
            A_csv = float(assets.sum(skipna=True)) if not assets.empty else np.nan
            L_csv = float(liabs.sum(skipna=True)) if not liabs.empty else np.nan
            DA_csv = DL_csv = np.nan
            if "midpoint_years" in cols_norm:
                t_col = cols_norm["midpoint_years"]
                a_df = alm_df.loc[alm_df[type_col].str.lower()=="asset", [t_col, amt_col]].copy()
                a_df[amt_col] = pd.to_numeric(a_df[amt_col], errors="coerce")
                a_df[t_col]   = pd.to_numeric(a_df[t_col], errors="coerce")
                if a_df[amt_col].sum(skipna=True) > 0:
                    DA_csv = float((a_df[t_col] * a_df[amt_col]).sum(skipna=True) / a_df[amt_col].sum(skipna=True))
                l_df = alm_df.loc[alm_df[type_col].str.lower()=="liability", [t_col, amt_col]].copy()
                l_df[amt_col] = pd.to_numeric(l_df[amt_col], errors="coerce")
                l_df[t_col]   = pd.to_numeric(l_df[t_col], errors="coerce")
                if l_df[amt_col].sum(skipna=True) > 0:
                    DL_csv = float((l_df[t_col] * l_df[amt_col]).sum(skipna=True) / l_df[amt_col].sum(skipna=True))
            st.write(f"**RSA (A):** {A_csv:,.2f} | **RSL (L):** {L_csv:,.2f}")
            st.write(f"**Estimated DA (years):** {DA_csv:.4f}" if not np.isnan(DA_csv) else "**Estimated DA:** —")
            st.write(f"**Estimated DL (years):** {DL_csv:.4f}" if not np.isnan(DL_csv) else "**Estimated DL:** —")
            if st.button("Use these in Direct Input", type="primary", key=file_scope_key("alm_use_btn", uploaded_file)):
                if not np.isnan(A_csv): st.session_state["alm_A"] = A_csv
                if not np.isnan(L_csv): st.session_state["alm_L"] = L_csv
                if not np.isnan(DA_csv): st.session_state["alm_DA"] = DA_csv
                if not np.isnan(DL_csv): st.session_state["alm_DL"] = DL_csv
                st.success("Loaded into Direct Input. Go to the '🧮 Direct Input (Equity Shock)' tab.")
        else:
            st.info("CSV must include columns: Type, Amount. (Optional: Midpoint_Years).")

# ======================================
# 6) Portfolio Simulation — BOTH (terminal horizon MC on log-returns)
# ======================================
st.header("6️⃣ Portfolio Simulation")

tabs = st.tabs(([f"{asset_name}"] + ([f"{bench_name or 'Benchmark'}"] if ndf["Bench_Return"].notna().sum() > 5 else [])))

def terminal_mc_panel(series, label):
    log_r = np.log1p(series.dropna().values.astype(float))
    if log_r.size < 5 or np.isnan(log_r).all():
        st.info(f"Insufficient data to simulate returns for {label}."); return
    n_sims  = st.slider(f"Number of simulations ({label})", 1000, 100_000, 10_000, step=1000,
                        key=file_scope_key(f"sim_n_{label}", uploaded_file))
    steps_y = int(np.ceil(252 * time_horizon))
    seed_on = st.checkbox(f"Set random seed (reproducible) — {label}", value=False,
                          key=file_scope_key(f"sim_seed_on_{label}", uploaded_file))
    if seed_on:
        seed_val = st.number_input(f"Seed — {label}", min_value=0, value=42, step=1,
                                   key=file_scope_key(f"sim_seed_{label}", uploaded_file))
        np.random.seed(int(seed_val))
    mu_log_d = float(np.nanmean(log_r)); sig_log_d = float(np.nanstd(log_r, ddof=1))
    term = np.exp(mu_log_d * steps_y + sig_log_d * np.sqrt(steps_y) * np.random.randn(n_sims)) - 1.0
    var_left = np.percentile(term, (1 - conf) * 100.0)
    prob_loss = float((term < 0).mean())
    mean_ret = float(np.mean(term)); p5, p50, p95 = np.percentile(term, [5, 50, 95])
    sim_df = pd.DataFrame({"Terminal Return": term})
    fig = px.histogram(sim_df, x="Terminal Return", nbins=60,
                       title=f"{label}: Monte Carlo Terminal Return (Horizon = {time_horizon:.2f} yrs, {n_sims} paths)")
    fig.add_vline(x=float(var_left), line_dash="dash", line_color="red",
                  annotation_text=f"Left-tail {(1 - conf):.0%}", annotation_position="top right")
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"**Probability of Loss:** {prob_loss:.2%}")
    st.write(f"**Mean:** {mean_ret:.2%} | **Median:** {p50:.2%} | **5th pct:** {p5:.2%} | **95th pct:** {p95:.2%}")

with tabs[0]:
    terminal_mc_panel(ndf["Asset_Return"], asset_name)
if len(tabs) > 1:
    with tabs[1]:
        terminal_mc_panel(ndf["Bench_Return"], bench_name or "Benchmark")

# ======================================
# 7) Download
# ======================================
st.header("7️⃣ Download Options")
output = io.BytesIO()
export_df = ndf.copy()
export_df.reset_index().rename(columns={ndf.index.name: "Date"}).to_excel(output, index=False)
st.download_button(
    "Download Processed Data (Excel)",
    data=output.getvalue(),
    file_name="processed_output.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
