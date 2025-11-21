import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import date, datetime, timedelta

COINGECKO_URL = "https://api.coingecko.com/api/v3"


# ---------- DATA FROM COINGECKO ----------

def fetch_top_markets(vs_currency="usd", per_page=100):
    """Pide las primeras N por market cap (N=per_page)."""
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": 1,
        "price_change_percentage": "1h,24h,7d,30d",
    }
    resp = requests.get(f"{COINGECKO_URL}/coins/markets", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return pd.DataFrame(data)


def fetch_fundamentals_for_ids(ids):
    rows = []
    for cid in ids:
        try:
            params = {
                "localization": "false",
                "tickers": "false",
                "market_data": "false",
                "community_data": "true",
                "developer_data": "true",
                "sparkline": "false",
            }
            resp = requests.get(
                f"{COINGECKO_URL}/coins/{cid}", params=params, timeout=30
            )
            resp.raise_for_status()
            j = resp.json()
            rows.append(
                {
                    "id": cid,
                    "developer_score": j.get("developer_score"),
                    "community_score": j.get("community_score"),
                    "coingecko_score": j.get("coingecko_score"),
                    "sentiment_votes_up_percentage": j.get(
                        "sentiment_votes_up_percentage"
                    ),
                    "genesis_date": j.get("genesis_date"),
                }
            )
        except Exception:
            rows.append(
                {
                    "id": cid,
                    "developer_score": None,
                    "community_score": None,
                    "coingecko_score": None,
                    "sentiment_votes_up_percentage": None,
                    "genesis_date": None,
                }
            )
    return pd.DataFrame(rows)


# ---------- HELPERS ----------

def scale(x, xmin, xmax):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.5
    if pd.isna(v) or xmax == xmin:
        return 0.5
    v = (v - xmin) / (xmax - xmin)
    return float(np.clip(v, 0.0, 1.0))


def years_since(genesis_str):
    if not genesis_str or not isinstance(genesis_str, str):
        return 0.0
    try:
        parts = genesis_str.split("-")
        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        g = date(y, m, d)
        return max((date.today() - g).days / 365.25, 0.0)
    except Exception:
        return 0.0


def is_stable_or_wrapped(row):
    """Marca stablecoins y wrapped/staked para excluirlas del radar."""
    sym = str(row.get("symbol", "")).lower()
    name = str(row.get("name", "")).lower()

    # Stablecoins típicas
    stable_syms = {
        "usdt", "usdc", "busd", "dai", "tusd", "usdd", "usde",
        "gusd", "usdp", "fdusd", "eusd", "eurt", "euroc", "susd",
        "lusd", "frax", "pai", "usdx",
    }
    if sym in stable_syms:
        return True

    # Wrapped / staked más comunes
    if "wrapped" in name:
        return True
    if "staked ether" in name or "staked eth" in name:
        return True

    wrapped_syms = {"wbtc", "weth", "wsteth", "wsol", "wavax"}
    if sym in wrapped_syms:
        return True

    return False


# ---------- SCORES ----------

def score_market(row):
    rank = row.get("market_cap_rank")
    if pd.isna(rank):
        rank = 200
    rank_score = scale(200 - rank, 0, 199)

    mc = row.get("market_cap")
    vol = row.get("total_volume")
    if pd.isna(mc) or mc == 0:
        liq_ratio = 0
    else:
        liq_ratio = (vol or 0) / mc
    liq_score = scale(liq_ratio, 0.01, 0.30)

    ath_chg = row.get("ath_change_percentage")
    if pd.isna(ath_chg):
        ath_score = 0.5
    else:
        dd = -ath_chg
        if dd < 10:
            ath_score = 0.2
        elif dd > 80:
            ath_score = 0.3
        else:
            ath_score = 0.2 + 0.8 * (dd - 10) / 70

    market_score_0_1 = 0.4 * rank_score + 0.3 * liq_score + 0.3 * ath_score
    return 25 * market_score_0_1


def score_tokenomics(row):
    circ = row.get("circulating_supply")
    max_supply = row.get("max_supply")

    if pd.isna(max_supply) or not max_supply or max_supply == 0:
        circ_score = 0.4
    else:
        try:
            ratio = float(circ) / float(max_supply)
        except (TypeError, ValueError, ZeroDivisionError):
            ratio = 0.5
        if ratio < 0.3:
            circ_score = 0.2
        elif ratio > 0.95:
            circ_score = 0.7
        else:
            circ_score = 0.2 + 0.8 * (ratio - 0.3) / (0.95 - 0.3)

    mc = row.get("market_cap")
    fdv = row.get("fully_diluted_valuation")
    if pd.isna(mc) or mc == 0 or pd.isna(fdv) or not fdv or fdv <= 0:
        dilution_score = 0.5
    else:
        try:
            d_ratio = float(mc) / float(fdv)
        except (TypeError, ValueError, ZeroDivisionError):
            d_ratio = 0.5

        if d_ratio >= 0.7:
            dilution_score = 1.0
        elif d_ratio <= 0.3:
            dilution_score = 0.3
        else:
            dilution_score = 0.3 + 0.7 * (d_ratio - 0.3) / (0.7 - 0.3)

    inflation_score = 0.6

    tokenomics_0_1 = 0.4 * circ_score + 0.4 * dilution_score + 0.2 * inflation_score
    return 25 * tokenomics_0_1


def score_momentum_narr(row):
    p30 = row.get("price_change_percentage_30d_in_currency")
    p7 = row.get("price_change_percentage_7d_in_currency")

    mom30 = scale(p30, -20, 60)
    mom7 = scale(p7, -15, 40)
    narrative_score = 0.6

    momentum_0_1 = 0.4 * mom30 + 0.3 * mom7 + 0.3 * narrative_score
    return 20 * momentum_0_1


def score_whales_deriv(_row):
    return 10.0  # neutro en v1


def score_personal_fundamental(row):
    dev = row.get("developer_score")
    comm = row.get("community_score")
    sent = row.get("sentiment_votes_up_percentage")
    genesis = row.get("genesis_date")

    dev_norm = scale(dev, 0, 100)
    comm_norm = scale(comm, 0, 100)
    sent_norm = scale(sent, 50, 90)

    years = years_since(genesis)
    years_capped = min(years, 10)
    age_norm = scale(years_capped, 0, 10)

    personal_0_1 = (
        0.35 * dev_norm
        + 0.25 * comm_norm
        + 0.25 * sent_norm
        + 0.15 * age_norm
    )
    return 10 * personal_0_1


# ---------- STREAMLIT APP ----------

st.set_page_config(page_title="Holder Radar v1 – Top 40 (sin stables/wrapped)", layout="wide")

st.title("📡 HOLDER RADAR v1 – Top 40 (sin stables ni wrapped)")

st.write(
    "Dashboard automático usando CoinGecko, **excluyendo stablecoins y wrapped/staked tokens**.\n\n"
    "- Mercado (0–25)\n"
    "- Tokenomics (0–25)\n"
    "- Momentum (0–20)\n"
    "- Ballenas & Derivados (0–20, neutro por ahora)\n"
    "- Criterio Personal / Fundamental auto (0–10)\n"
)


@st.cache_data(ttl=900)
def load_data():
    # 1) Traemos top 100
    markets_all = fetch_top_markets("usd", per_page=100)

    # 2) Hora de última actualización global (máximo last_updated)
    last_updated_utc = None
    if "last_updated" in markets_all.columns:
        lu = pd.to_datetime(markets_all["last_updated"], errors="coerce", utc=True)
        if lu.notna().any():
            last_updated_utc = lu.max().to_pydatetime()

    # 3) Quitamos stables/wrapped y nos quedamos con las 40 mejores por market_cap_rank
    markets = markets_all[~markets_all.apply(is_stable_or_wrapped, axis=1)].copy()
    markets = markets.sort_values("market_cap_rank").head(40).reset_index(drop=True)

    # 4) Fundamentals
    fundamentals = fetch_fundamentals_for_ids(markets["id"].tolist())
    df = markets.merge(fundamentals, on="id", how="left")

    # 5) Scores
    df["score_market"] = df.apply(score_market, axis=1)
    df["score_tokenomics"] = df.apply(score_tokenomics, axis=1)
    df["score_momentum_narr"] = df.apply(score_momentum_narr, axis=1)
    df["score_whales_deriv"] = df.apply(score_whales_deriv, axis=1)
    df["score_personal"] = df.apply(score_personal_fundamental, axis=1)

    df["holder_score_100"] = (
        df["score_market"]
        + df["score_tokenomics"]
        + df["score_momentum_narr"]
        + df["score_whales_deriv"]
        + df["score_personal"]
    )
    return df, last_updated_utc


df, last_updated_utc = load_data()

# ---------- HORAS (UTC y COLOMBIA) ----------
now_utc = datetime.utcnow()
now_col = now_utc - timedelta(hours=5)  # Colombia UTC-5 todo el año

if last_updated_utc is None:
    last_updated_utc = now_utc
last_updated_col = last_updated_utc - timedelta(hours=5)

st.markdown(
    f"**Hora actual (UTC):** {now_utc.strftime('%Y-%m-%d %H:%M:%S')}  \n"
    f"**Hora actual Colombia (UTC-5):** {now_col.strftime('%Y-%m-%d %H:%M:%S')}  \n"
    f"**Última actualización datos (UTC):** {last_updated_utc.strftime('%Y-%m-%d %H:%M:%S')}  \n"
    f"**Última actualización datos (Colombia):** {last_updated_col.strftime('%Y-%m-%d %H:%M:%S')}"
)

# ---- Filtros ----
st.sidebar.header("Filtros")
min_holder = st.sidebar.slider(
    "HOLDER SCORE mínimo (0–100)", 0, 100, 60, 1
)

# Filtramos y ORDENAMOS por score descendente
df_filtered = df[df["holder_score_100"] >= min_holder].copy()
df_filtered = df_filtered.sort_values("holder_score_100", ascending=False).reset_index(drop=True)

# ---- Tabla ----
st.subheader("Tabla con HOLDER SCORE (0–100) — sin stables ni wrapped")

# Métrica rápida: cuántas pasan el filtro
st.write(
    f"Monedas con HOLDER SCORE ≥ {min_holder}: **{len(df_filtered)}** de {len(df)} analizadas."
)

# Estilo para resaltar TOP 3
def highlight_top3(row):
    if row.name == 0:
        # Puesto 1
        return ['background-color: #14532d; color: white; font-weight: bold'] * len(row)
    elif row.name in (1, 2):
        # Puestos 2 y 3
        return ['background-color: #166534; color: white'] * len(row)
    else:
        return [''] * len(row)

cols_table = [
    "market_cap_rank",
    "symbol",
    "name",
    "current_price",
    "market_cap",
    "total_volume",
    "score_market",
    "score_tokenomics",
    "score_momentum_narr",
    "score_whales_deriv",
    "score_personal",
    "holder_score_100",
]

st.dataframe(
    df_filtered[cols_table].style.apply(highlight_top3, axis=1),
    use_container_width=True,
)

# ---- FICHA DE DETALLE ----
st.subheader("Detalle de una cripto")

if not df_filtered.empty:
    # Como df_filtered ya está ordenado, el primero es el mejor score
    default_symbol = df_filtered["symbol"].iloc[0]
    symbol_list = df_filtered["symbol"].tolist()

    symbol_sel = st.selectbox(
        "Selecciona símbolo",
        symbol_list,
        index=symbol_list.index(default_symbol),
    )

    row = df_filtered[df_filtered["symbol"] == symbol_sel].iloc[0]

    c1, c2 = st.columns(2)

    with c1:
        st.metric("HOLDER SCORE", f"{row['holder_score_100']:.1f}")
        st.write(f"**Nombre:** {row['name']}")
        st.write(f"**Símbolo:** {row['symbol'].upper()}")
        st.write(f"**Precio actual:** ${row['current_price']:.4f}")
        st.write(f"**Market Cap Rank:** {int(row['market_cap_rank'])}")
        st.write(f"**Market Cap:** {row['market_cap']:,}")
        st.write(f"**Volumen 24h:** {row['total_volume']:,}")

    with c2:
        st.write("**Desglose de scores:**")
        st.write(f"- Mercado: {row['score_market']:.1f} / 25")
        st.write(f"- Tokenomics: {row['score_tokenomics']:.1f} / 25")
        st.write(f"- Momentum: {row['score_momentum_narr']:.1f} / 20")
        st.write(f"- Ballenas & Derivados (neutro): {row['score_whales_deriv']:.1f} / 20")
        st.write(f"- Fundamental auto: {row['score_personal']:.1f} / 10")

        st.write("**Fundamentales CoinGecko:**")
        st.write(f"- Developer score: {row['developer_score']}")
        st.write(f"- Community score: {row['community_score']}")
        st.write(f"- Sentiment ↑: {row['sentiment_votes_up_percentage']}%")
        st.write(f"- Genesis date: {row['genesis_date']}")
