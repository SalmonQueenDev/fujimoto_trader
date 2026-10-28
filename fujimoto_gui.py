"""후지모토 시게루식 1:2:6 분할 매매 판단 GUI

한국·미국 종목명(예: 삼성전자, 애플), 6자리 종목코드 또는 미국 티커(예: AAPL)를 입력하면 일봉 데이터를 받아
(한국: pykrx, 미국: yfinance — 미국 종목의 매수 수량은 원/달러 환율로 환산)
후지모토 본인이 저서·인터뷰에서 밝힌 원칙을 일봉 규칙으로 옮겨 1:2:6 분할 매수/매도 단계를 판단한다.

후지모토 본인이 밝힌 원칙 (『87歳、現役トレーダー シゲルさんの教え』, プレジデント·マネーポスト·楽天証券 인터뷰)
  - 매수 전제: 増収·増益·増配 (매출·이익·배당 증가). 이익은 경상/순이익을 보고 영업이익은 잘 안 본다.
  - RSI 30 이하는 매수, 70 이상은 매도. 단, 지표 하나로만 판단하지 않는다 (MACD 도 본다).
  - 1:2:6: 좋아 보이면 1 → 역시 좋으면 2 → 성장하기 시작하면 6. 팔 때도 1:2:6.
  - 오르면 팔고 내려가면 산다. 오른다고 사지 않는다. 떨어졌을 때 파는 것이 최악의 수.
  - 물타기는 무섭지 않다 — 단, 실적이 계속 좋을 종목만, 냉정하게.
  - '얼마까지 떨어지면 판다'는 손절 규칙은 정하지 않는다.

일봉 규칙으로 옮긴 것 (저서·인터뷰에 구체적 조건이 없어 해석한 부분 포함)
  매수  1차 +1 : RSI ≤ 30 (과매도 — 打診買い)
        2차 +2 : RSI 가 30 위로 반등 (やっぱりよさそう)  또는  물타기: 직전 매수가 대비 -10% & RSI ≤ 30
        3차 +6 : MACD 골든크로스 & 종가가 평단 위 (成長し始めた)  또는  물타기: 위와 같은 조건
  매도  (평단보다 위에 있을 때만 — 손실 상태에서는 팔지 않음)
        1차 -1 : RSI ≥ 70 (과열)
        2차 -2 : RSI 가 70 아래로 꺾임
        3차 잔량 : MACD 데드크로스
  실적 조건(증수·증익)을 못 맞추면 매수 신호가 나와도 '관망', 보유 중이면 '정리 검토'로 표시한다.
  손절은 기본 꺼짐 (PARAMS["stop_loss_pct"] 로 켤 수 있음).

'종목 검색' 탭: 코스피·코스닥 전체 / S&P500 에서 최근 N거래일 안에 매수 신호가 나왔고
아직 보유 중인 종목을 찾은 뒤, 증수·증익·증배로 매수/매도 점수를 매긴다.
받은 시세·실적은 cache/ 에 날짜별로 저장해 같은 날 재검색은 빠르다.
"""
import math
import os
import pickle
import re
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from tkinter import ttk, messagebox

import FinanceDataReader as fdr
import matplotlib
import numpy as np
import matplotlib.ticker
import requests
import yfinance as yf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from pykrx import stock
from pykrx.website.comm import webio as krx_webio


class _TimeoutRequests:
    """pykrx 는 requests 를 timeout=None 으로 불러 서버가 응답을 멈추면 무한 대기한다 → 기본 10초 타임아웃 주입."""

    def __getattr__(self, name):
        return getattr(requests, name)

    def get(self, *args, **kw):
        return requests.get(*args, timeout=kw.pop("timeout", 10), **kw)

    def post(self, *args, **kw):
        return requests.post(*args, timeout=kw.pop("timeout", 10), **kw)


krx_webio.requests = _TimeoutRequests()

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

PARAMS = {
    "rsi_period": 14,
    "rsi_oversold": 30,      # 이하면 매수 (후지모토: "30以下は買い")
    "rsi_overbought": 70,    # 이상이면 매도 (후지모토: "70以上は売り")
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "units": (1, 2, 6),      # 1:2:6 분할 비중
    "nanpin": True,          # 물타기 허용 (후지모토: "ナンピンは怖くない")
    "nanpin_drop": 0.10,     # 직전 매수가 대비 이만큼 더 떨어지고 RSI ≤ 30 이면 다음 단계를 물타기로 매수 (해석값)
    "stop_loss_pct": None,   # 후지모토는 손절 규칙을 두지 않음. 예: 20 으로 두면 평단 대비 -20% 종가에서 전량 정리
    # 1차 매수(회차 시작) 필터 — 신호 성적표에서 효과가 확인된 조건. None 이면 끔 (화면 상단 체크박스로도 바꿈)
    "entry_market_drop": -0.05,   # 지수 20일 수익률이 이 값 이하(시장 전체 급락)일 때만 1차 매수
    "entry_max_rel_drop": -0.15,  # 종목 20일 수익률 - 지수 20일 수익률이 이 값 이하(혼자 과도하게 하락)면 1차 매수 안 함
    "capital": 10_000_000,   # 종목당 투자 한도(원) — 1:2:6 으로 나눠 씀
}
MIN_BARS = 120  # 판단에 필요한 최소 거래일 수
BACKTEST_DAYS = 1825  # 종목 분석 탭은 과거 손익 계산을 위해 5년치 일봉을 받음 (검색 탭은 2년)

SCAN = {
    "recent_days": 3,        # 최근 며칠(거래일) 안에 나온 매수 신호까지 포함
    "min_amount_KR": 1e9,    # 한국: 최근 20일 평균 거래대금 하한(원)
    "min_amount_US": 2e7,    # 미국: 최근 20일 평균 거래대금 하한(달러)
    "chunk_US": 100,         # 미국 시세 한 번에 요청할 종목 수
}
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def fmt_price(v, market):
    return f"${v:,.2f}" if market == "US" else f"{v:,.0f}원"


def resolve_ticker(query):
    """종목명/코드/티커를 (코드, 이름, 시장)으로 변환. 시장은 'KR' 또는 'US'."""
    query = query.strip()
    if query.isdigit() and len(query) == 6:
        name = stock.get_market_ticker_name(query)
        if not isinstance(name, str) or not name:
            raise ValueError(f"종목코드 {query}를 찾을 수 없습니다.")
        return query, name, "KR"
    r = requests.get(
        "https://ac.stock.naver.com/ac",
        params={"q": query, "target": "stock"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    r.raise_for_status()
    items = [i for i in r.json().get("items", []) if i.get("nationCode") in ("KOR", "USA")]
    if items:
        exact = [i for i in items if i["name"] == query or i["code"].upper() == query.upper()]
        item = (exact or items)[0]
        if item["nationCode"] == "KOR":
            return item["code"], item["name"], "KR"
        # 네이버의 BRK B / BRK.B → yfinance 표기 BRK-B
        return item["code"].replace(" ", "-").replace(".", "-"), item["name"], "US"
    # 네이버 검색에 없는 영문 티커는 미국 종목으로 보고 시세 조회에서 존재 여부 확인
    if query.replace(".", "").replace("-", "").isalpha():
        return query.upper().replace(".", "-"), query.upper(), "US"
    raise ValueError(f"'{query}' 종목을 찾을 수 없습니다.")


def fetch_usdkrw():
    fx = yf.Ticker("KRW=X").history(period="5d")["Close"].dropna()
    if fx.empty:
        raise ValueError("원/달러 환율을 가져오지 못했습니다.")
    return float(fx.iloc[-1])


def us_frame(raw):
    """yfinance 일봉을 pykrx 와 같은 한글 컬럼으로 변환."""
    df = raw.rename(columns={"Open": "시가", "High": "고가", "Low": "저가", "Close": "종가", "Volume": "거래량"})
    df = df[["시가", "고가", "저가", "종가", "거래량"]].dropna()  # 장 마감 직후 오늘 행은 값이 비어 있을 수 있음
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def fetch_ohlcv(code, market, days=730):
    end = datetime.now()
    start = end - timedelta(days=days)
    if market == "US":
        raw = yf.Ticker(code).history(start=start.strftime("%Y-%m-%d"),
                                      end=(end + timedelta(days=1)).strftime("%Y-%m-%d"), auto_adjust=True)
        if raw.empty:
            raise ValueError(f"미국 종목 '{code}'의 시세를 찾을 수 없습니다.")
        df = us_frame(raw)
    else:
        df = stock.get_market_ohlcv(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), code)
    df = df[df["거래량"] > 0].copy()
    if len(df) < MIN_BARS:
        raise ValueError("판단에 필요한 거래일 데이터가 부족합니다.")
    return df


def add_indicators(df):
    p = PARAMS
    close = df["종가"]

    # RSI (Wilder 평활)
    diff = close.diff()
    a = 1 / p["rsi_period"]
    gain = diff.clip(lower=0).ewm(alpha=a, adjust=False).mean()
    loss = (-diff.clip(upper=0)).ewm(alpha=a, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)

    # MACD
    ema = lambda n: close.ewm(span=n, adjust=False).mean()
    df["MACD"] = ema(p["macd_fast"]) - ema(p["macd_slow"])
    df["MACD_SIG"] = df["MACD"].ewm(span=p["macd_signal"], adjust=False).mean()
    return df


def simulate(df, entry=None):
    """1:2:6 매수/매도 상태 머신을 과거부터 오늘까지 돌린다.
    entry: (지수 급락 기준, 시장 대비 낙폭 한도) 로 PARAMS 의 1차 매수 필터를 덮어씀 (필터 비교용).
    필터에는 df["IDX20"](같은 날짜의 지수 20일 수익률)이 필요하며, 없거나 비어 있는 날은 필터를 통과하지 못함.
    반환: (이벤트 목록, 오늘 기준 상태 dict)."""
    p = PARAMS
    mkt_lim, rel_lim = entry if entry is not None else (p["entry_market_drop"], p["entry_max_rel_drop"])
    I = df["IDX20"].to_numpy() if "IDX20" in df else np.full(len(df), np.nan)
    rel = df["종가"].pct_change(20).to_numpy() - I

    def entry_ok(i):
        if mkt_lim is not None and not I[i] <= mkt_lim:
            return False
        return rel_lim is None or rel[i] > rel_lim
    u1, u2, u3 = p["units"]
    full = u1 + u2 + u3
    lo, hi, drop = p["rsi_oversold"], p["rsi_overbought"], p["nanpin_drop"]
    R, M, S = df["RSI"].to_numpy(), df["MACD"].to_numpy(), df["MACD_SIG"].to_numpy()
    C = df["종가"].to_numpy()
    n = len(df)

    events = []
    held, cost, stage, sell_step, last_buy = 0, 0.0, 0, 0, 0.0

    def trade(i, kind, units, label, why):
        nonlocal held, cost
        if kind == "buy":
            cost = (cost * held + C[i] * units) / (held + units)
            held += units
        else:
            held -= units
        events.append({"i": i, "date": df.index[i], "kind": kind, "units": units, "price": C[i],
                       "label": label, "why": why, "held": held, "cost": cost,
                       "pnl": C[i] / cost - 1 if kind == "sell" else None})

    for i in range(p["macd_slow"] + p["macd_signal"], n):
        gc = M[i - 1] <= S[i - 1] and M[i] > S[i]
        dc = M[i - 1] >= S[i - 1] and M[i] < S[i]
        nanpin_ok = p["nanpin"] and R[i] <= lo and C[i] <= last_buy * (1 - drop)

        if held and p["stop_loss_pct"] and C[i] < cost * (1 - p["stop_loss_pct"] / 100):
            trade(i, "sell", held, "손절", f"평단 대비 -{p['stop_loss_pct']}% (선택 규칙)")
        elif held and C[i] > cost and sell_step == 0 and R[i] >= hi:
            trade(i, "sell", min(held, u1), "1차 매도", f"RSI {R[i]:.0f} ≥ {hi} (과열)")
            sell_step = 1
        elif held and C[i] > cost and sell_step == 1 and R[i - 1] >= hi > R[i]:
            trade(i, "sell", min(held, u2), "2차 매도", f"RSI {hi} 아래로 꺾임")
            sell_step = 2
        elif held and C[i] > cost and sell_step == 2 and dc:
            trade(i, "sell", held, "3차 매도", "MACD 데드크로스")
        elif sell_step == 0:  # 매도를 시작한 뒤에는 추가 매수하지 않음
            if stage == 0 and R[i] <= lo and entry_ok(i):
                trade(i, "buy", u1, "1차 매수", f"RSI {R[i]:.0f} ≤ {lo} (과매도)"
                      + (f" · 지수 20일 {I[i]:+.1%}" if mkt_lim is not None or rel_lim is not None else ""))
                stage, last_buy = 1, C[i]
            elif stage == 1 and R[i - 1] <= lo < R[i]:
                trade(i, "buy", u2, "2차 매수", f"RSI {lo} 위로 반등")
                stage, last_buy = 2, C[i]
            elif stage == 2 and gc and C[i] > cost:
                trade(i, "buy", u3, "3차 매수", "MACD 골든크로스 & 평단 위 (상승 시작)")
                stage, last_buy = 3, C[i]
            elif stage in (1, 2) and nanpin_ok:
                u = u2 if stage == 1 else u3
                trade(i, "buy", u, f"{stage + 1}차 물타기", f"직전 매수가 대비 -{drop:.0%} 이상 & RSI {R[i]:.0f}")
                stage, last_buy = stage + 1, C[i]
        elif R[i] <= lo:
            sell_step = 0  # 매도 도중 다시 과매도로 내려오면 매도 사이클 종료 → 다시 '내려가면 산다'

        if held == 0 and stage:
            stage = sell_step = 0
            cost = 0.0

    pnl = C[-1] / cost - 1 if held else None
    if held == 0:
        conds = [f"RSI {R[-1]:.0f} → {lo} 이하"]
        if mkt_lim is not None:
            conds.append(f"지수 20일 {'-' if np.isnan(I[-1]) else f'{I[-1]:+.1%}'} → {mkt_lim:.0%} 이하")
        if rel_lim is not None:
            conds.append(f"시장 대비 {'-' if np.isnan(rel[-1]) else f'{rel[-1] * 100:+.0f}%p'} → {rel_lim * 100:.0f}%p 초과")
        nxt = f"1차 매수(+{u1}/{full}) 조건: " + " · ".join(conds) + " · 오른다고 사지 않는다"
    elif sell_step == 1:
        nxt = f"RSI 가 {hi} 아래로 꺾이면 2차 매도 (-{u2}) · 평단 위일 때만"
    elif sell_step == 2:
        nxt = "MACD 데드크로스 나오면 잔량 매도 · 평단 위일 때만"
    elif stage == 3:
        nxt = f"평단 위에서 RSI {hi} 이상이면 1차 매도 (-{u1}) · 손실 중에는 팔지 않음"
    else:
        nxt = (f"RSI {lo} 위로 반등하면 2차 매수 (+{u2})" if stage == 1
               else f"MACD 골든크로스 & 평단 위면 3차 매수 (+{u3})")
        if p["nanpin"]:
            nxt += f" · 직전 매수가 -{drop:.0%} & RSI ≤ {lo} 면 물타기 (+{u2 if stage == 1 else u3})"

    state = {"held": held, "full": full, "stage": stage, "sell_step": sell_step,
             "cost": cost, "pnl": pnl, "next": nxt}
    return events, state


def backtest(df, events, capital, fx=1.0):
    """신호대로 사고팔았을 때의 손익.
    1단위 = 종목당 한도의 1/9 을 그날 종가에 정수 주식 수로 매수, 매도는 보유 단위 비율만큼 주식을 판다.
    회차 = 보유 0 에서 첫 매수 ~ 다시 보유 0 이 될 때까지. 원화 환산은 현재 환율, 수수료·세금은 미반영."""
    unit_krw = capital / sum(PARAMS["units"])
    cycles, cur = [], None
    shares, cost = 0, 0.0  # 보유 주식 수, 보유분 매입금액(종목 통화)
    for e in events:
        if e["kind"] == "buy":
            if cur is None:
                cur = {"start": e["date"], "start_price": e["price"], "invested": 0.0, "pnl": 0.0, "buys": 0}
            q = int(unit_krw * e["units"] // (e["price"] * fx))
            shares += q
            cost += q * e["price"]
            cur["invested"] += q * e["price"]
            cur["buys"] += 1
            continue
        q = shares if e["held"] == 0 else round(shares * e["units"] / (e["held"] + e["units"]))
        avg = cost / shares if shares else 0.0
        cur["pnl"] += q * (e["price"] - avg)
        shares -= q
        cost -= q * avg
        if e["held"] == 0:
            cycles.append({**cur, "end": e["date"]})
            cur, shares, cost = None, 0, 0.0

    close = df["종가"].iloc[-1]
    for c in cycles:
        c["days"] = (c["end"] - c["start"]).days
    if cur:  # 아직 보유 중인 회차는 오늘 종가로 평가
        cycles.append({**cur, "end": None, "days": (df.index[-1] - cur["start"]).days,
                       "unreal": shares * close - cost})
    done = [c for c in cycles if c["end"] is not None]
    for c in cycles:
        c["ret"] = (c["pnl"] + c.get("unreal", 0.0)) / c["invested"] if c["invested"] else 0.0
        for k in ("invested", "pnl", "unreal"):
            if k in c:
                c[k] *= fx  # 원화 환산

    first = cycles[0] if cycles else None
    invested = sum(c["invested"] for c in done)
    return {
        "cycles": cycles,
        "start": df.index[0], "end": df.index[-1],
        "n": len(done), "wins": sum(1 for c in done if c["pnl"] > 0),
        "pnl": sum(c["pnl"] for c in done), "ret": sum(c["pnl"] for c in done) / invested if invested else 0.0,
        "avg_days": sum(c["days"] for c in done) / len(done) if done else 0,
        "open": cycles[-1] if cycles and cycles[-1]["end"] is None else None,
        "hold_ret": close / first["start_price"] - 1 if first else None,  # 첫 매수일에 사서 지금까지 보유했다면
        "hold_from": first["start"] if first else None,
    }


def position_plan(price, capital, fx=1.0):
    """종목당 한도를 1:2:6 으로 나눈 금액(원)과 수량. fx: 종목 통화 1단위의 원화 가격."""
    units = PARAMS["units"]
    parts = [capital * u / sum(units) for u in units]
    return {"parts": parts, "shares": [int(a // (price * fx)) for a in parts]}


def evaluate(df, market, fund=None):
    add_indicators(df)
    events, state = simulate(df)
    last = df.iloc[-1]
    today = [e for e in events if e["i"] == len(df) - 1]
    gate = fund.get("gate") if fund else None

    held, full = state["held"], state["full"]
    if today:
        e = today[-1]
        verdict = f"오늘 {e['label']} 신호 ({'+' if e['kind'] == 'buy' else '-'}{e['units']}) — {e['why']}"
        if e["kind"] == "buy" and gate is False:
            verdict += " · 실적 조건 미충족 → 관망"
    elif held:
        verdict = f"보유 중 {held}/{full} · 평단 대비 {state['pnl']:+.1%}"
        if gate is False:
            verdict += " · 실적 악화 → 정리 검토"
    else:
        verdict = "보유 없음 · 대기"

    p = PARAMS
    rsi, lo, hi = last["RSI"], p["rsi_oversold"], p["rsi_overbought"]
    ov_sold = df.index[df["RSI"] <= lo]
    ov_bought = df.index[df["RSI"] >= hi]
    macd_gc = df.index[(df["MACD"].shift() <= df["MACD_SIG"].shift()) & (df["MACD"] > df["MACD_SIG"])]
    macd_dc = df.index[(df["MACD"].shift() >= df["MACD_SIG"].shift()) & (df["MACD"] < df["MACD_SIG"])]
    fmt = lambda idx: idx[-1].strftime("%Y-%m-%d") if len(idx) else "-"
    checks = [
        (f"① RSI({p['rsi_period']}) — {lo} 이하 매수 / {hi} 이상 매도",
         "과매도 (매수권)" if rsi <= lo else "과열 (매도권)" if rsi >= hi else "중립",
         f"{rsi:.1f} · 최근 {lo} 이하 {fmt(ov_sold)} · 최근 {hi} 이상 {fmt(ov_bought)}"),
        (f"② MACD({p['macd_fast']}/{p['macd_slow']}/{p['macd_signal']}) — 보조 확인",
         "골든 (상승)" if last["MACD"] > last["MACD_SIG"] else "데드 (하락)",
         f"MACD {last['MACD']:,.2f} / 시그널 {last['MACD_SIG']:,.2f} · 최근 GC {fmt(macd_gc)} · DC {fmt(macd_dc)}"),
        ("③ 증수·증익·증배 — 매수 전제",
         fund["verdict"] if fund else "조회 실패",
         fund_summary(fund, market) if fund else "실적 데이터를 가져오지 못했습니다."),
    ]
    info = {"close": last["종가"], "date": df.index[-1].strftime("%Y-%m-%d"), "market": market, **state}
    return verdict, checks, events, info


def load_universe(market):
    """검색 대상 {코드: 이름}. 한국은 코넥스·스팩·우선주·관리종목·투자주의환기종목 제외 (ETF는 목록에 없음)."""
    if market == "KR":
        kl = fdr.StockListing("KRX")
        kl = kl[kl["Market"].isin(["KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"])
                & kl["Code"].str.endswith("0")     # 우선주는 코드 끝자리가 5·7·9·K 등
                & ~kl["Dept"].fillna("").str.contains("SPAC|관리종목|투자주의환기")]
        return dict(zip(kl["Code"], kl["Name"]))
    sp = fdr.StockListing("S&P500")
    return {s.replace(".", "-"): n for s, n in zip(sp["Symbol"], sp["Name"])}


def _fetch_kr(codes, progress, cancel, days=730):
    # pykrx 는 여러 스레드에서 동시에 부르면 멈추므로 순차 조회 (종목당 약 0.02초)
    data = {}
    for n, c in enumerate(codes, 1):
        if cancel.is_set():
            return None
        for _ in range(2):
            try:
                data[c] = fetch_ohlcv(c, "KR", days)
                break
            except ValueError:
                break     # 신규 상장 등 데이터 부족 종목은 건너뜀
            except Exception:
                continue  # 응답 지연(타임아웃)·연결 끊김은 한 번 더 시도
        if n % 20 == 0 or n == len(codes):
            progress(n, len(codes))
    return data


def _fetch_us(codes, progress, cancel, days=730):
    data, step = {}, SCAN["chunk_US"]
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    for k in range(0, len(codes), step):
        if cancel.is_set():
            return None
        chunk = codes[k:k + step]
        raw = yf.download(chunk, start=start, group_by="ticker", auto_adjust=True, progress=False, threads=True)
        for c in chunk:
            if c in raw.columns.get_level_values(0):
                df = us_frame(raw[c])
                df = df[df["거래량"] > 0].copy()
                if len(df) >= MIN_BARS:
                    data[c] = df
        progress(min(k + step, len(codes)), len(codes))
    return data


def _cache_path(prefix):
    return os.path.join(CACHE_DIR, f"{prefix}_{datetime.now():%Y%m%d}.pkl")


def _load_cache(prefix):
    path = _cache_path(prefix)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_cache(prefix, obj):
    path = _cache_path(prefix)
    os.makedirs(CACHE_DIR, exist_ok=True)
    for old in os.listdir(CACHE_DIR):  # 지난 날짜 캐시(한국 시세는 하루 약 60MB)는 삭제 — KR 과 KR_5y 는 서로 지우지 않음
        if re.fullmatch(rf"{re.escape(prefix)}_\d{{8}}\.pkl", old) and old != os.path.basename(path):
            os.remove(os.path.join(CACHE_DIR, old))
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_market_data(market, refresh, progress, cancel, days=730):
    """시장 전체 일봉 {코드: df} 와 {코드: 이름}. 오늘 받은 캐시가 있으면 재사용. 취소 시 None.
    days 가 기본(2년)이 아니면 별도 캐시(예: KR_5y)에 저장."""
    prefix = market if days == 730 else f"{market}_{days // 365}y"
    cached = None if refresh else _load_cache(prefix)
    if cached:
        return cached["data"], cached["names"]
    names = load_universe(market)
    fetch = _fetch_us if market == "US" else _fetch_kr
    data = fetch(list(names), progress, cancel, days)
    if data is None:
        return None, None
    _save_cache(prefix, {"data": data, "names": names})
    return data, names


def scan_signals(data, names, market, recent_days, meta):
    """최근 recent_days 거래일 안에 마지막 이벤트가 매수(물타기 포함)이고 아직 보유 중인 종목."""
    rows = []
    for code, df in data.items():
        amount = (df["종가"] * df["거래량"]).iloc[-20:].mean()
        if amount < SCAN[f"min_amount_{market}"]:
            continue
        add_indicators(df)
        attach_index(df, code, meta)
        events, state = simulate(df)
        if not events:
            continue
        e = events[-1]
        if e["kind"] != "buy" or e["i"] < len(df) - recent_days:
            continue
        rows.append({"market": market, "code": code, "name": names.get(code, code), "label": e["label"],
                     "units": e["units"], "date": e["date"], "price": e["price"], "close": df["종가"].iloc[-1],
                     "held": state["held"], "full": state["full"], "pnl": state["pnl"],
                     "next": state["next"], "amount": amount})
    rows.sort(key=lambda r: (r["date"], r["amount"]), reverse=True)
    return rows


def fmt_amount(v, market):
    return f"${v / 1e6:,.1f}M" if market == "US" else f"{v / 1e8:,.0f}억"


# ── 신호 성적표: 과거 매수 신호가 실제로 맞았는지, 어떤 종목·상황에서 잘 맞는지 ─────
HORIZONS = (20, 60, 120)  # 신호 후 몇 거래일 뒤 수익률을 볼지
RISK_WINDOW = 60          # 신호 후 이 기간 안의 최대 낙폭/상승

# 나눠 보기 기준 — 모두 신호 당일까지의 정보만 사용 (미래 정보 사용 방지)
TREND, SLOPE, MKT, REL, SIZE, SECTOR = ("추세 (200일선)", "200일선 기울기", "시장 상황 (지수 20일)",
                                        "시장 대비 낙폭 (20일)", "규모 (20일 평균 거래대금)", "업종")
FEATURE_LABELS = {
    TREND: ("200일선 위", "200일선 아래"),
    SLOPE: ("200일선 상승 중", "200일선 하락 중"),
    MKT: ("지수 급락 (-5% 이하)", "지수 약세 (-5~0%)", "지수 상승"),
    REL: ("시장보다 15%p 이상 더 하락", "시장보다 5~15%p 더 하락", "시장과 비슷하거나 덜 하락"),
    SIZE: {"KR": ("50억 미만", "50~300억", "300억 이상"), "US": ("$100M 미만", "$100~500M", "$500M 이상")},
}
SIZE_CUTS = {"KR": (5e9, 3e10), "US": (1e8, 5e8)}
FEATURES = (TREND, SLOPE, MKT, REL, SIZE, SECTOR)


def load_stats_meta(market):
    """나눠 보기에 필요한 종목별 지수 20일 수익률과 업종. 반환: {"index": {코드: Series}, "sector": {코드: 업종}}."""
    start = (datetime.now() - timedelta(days=365 * 7)).strftime("%Y-%m-%d")
    if market == "KR":
        desc = fdr.StockListing("KRX-DESC")
        idx = {m: fdr.DataReader(sym, start)["Close"].pct_change(20) for m, sym in (("KOSPI", "KS11"), ("KOSDAQ", "KQ11"))}
        return {"index": {c: idx["KOSPI" if m == "KOSPI" else "KOSDAQ"] for c, m in zip(desc["Code"], desc["Market"])},
                "default_index": idx["KOSPI"],
                "sector": dict(zip(desc["Code"], desc["Industry"].fillna("미분류")))}
    sp = yf.Ticker("^GSPC").history(start=start)["Close"]
    sp.index = sp.index.tz_localize(None)
    listing = fdr.StockListing("S&P500")
    return {"index": {}, "default_index": sp.pct_change(20),
            "sector": {s.replace(".", "-"): sec for s, sec in zip(listing["Symbol"], listing["Sector"])}}


_META = {}


def get_meta(market):
    """load_stats_meta 결과를 실행 중 한 번만 받아 재사용."""
    if market not in _META:
        _META[market] = load_stats_meta(market)
    return _META[market]


def attach_index(df, code, meta):
    """df["IDX20"] = 같은 날짜의 소속 지수 20일 수익률 (지수에 아직 오늘 값이 없으면 직전 값)."""
    idx = meta["index"].get(code, meta.get("default_index"))
    df["IDX20"] = idx.reindex(df.index, method="ffill", limit=3).to_numpy() if idx is not None else np.nan
    return idx


def day_feature_codes(df, market, index_ret20):
    """날짜별 나눠 보기 구간 번호 {기준: int 배열} (-1 = 판단 불가). 업종은 종목마다 하나라 따로 처리."""
    C, amt = df["종가"], (df["종가"] * df["거래량"]).rolling(20).mean()
    ma200 = C.rolling(200).mean()
    slope = ma200 - ma200.shift(20)
    lo, hi = SIZE_CUTS[market]

    def codes(conds, valid):
        return np.where(valid, np.select(conds, list(range(len(conds))), -1), -1)

    feats = {
        TREND: codes([C >= ma200, C < ma200], ma200.notna()),
        SLOPE: codes([slope > 0, slope <= 0], slope.notna()),
        SIZE: codes([amt < lo, (amt >= lo) & (amt < hi), amt >= hi], amt.notna()),
    }
    if index_ret20 is not None:
        mret = index_ret20.reindex(df.index).to_numpy()
        rel = C.pct_change(20).to_numpy() - mret
        feats[MKT] = codes([mret <= -0.05, (mret > -0.05) & (mret <= 0), mret > 0], ~np.isnan(mret))
        feats[REL] = codes([rel <= -0.15, (rel > -0.15) & (rel <= -0.05), rel > -0.05], ~np.isnan(rel))
    return feats


def _labels(feature, market):
    labels = FEATURE_LABELS[feature]
    return labels[market] if isinstance(labels, dict) else labels


def _order(feature, keys):
    """구간 표시 순서: 정해진 구간은 정해진 순서, 업종 등은 건수 많은 순 (keys 는 {구간: 목록})."""
    by_count = sorted(keys, key=lambda k: -len(keys[k]))
    if feature == SECTOR:
        return by_count
    labels = FEATURE_LABELS[feature]
    order = [l for ls in (labels.values() if isinstance(labels, dict) else [labels]) for l in ls]
    return [l for l in order if l in keys] + [l for l in by_count if l not in order]


ENTRY_COMBOS = (("필터 없음", (None, None)), ("지수 급락 시만", (-0.05, None)),
                ("과도 낙폭 제외", (None, -0.15)), ("둘 다", (-0.05, -0.15)))


def signal_stats(data, market, meta, progress=None, cancel=None):
    """시장 전체 과거 매수 신호의 이후 성과를 모은다. 취소 시 None.
    반환: (신호 기록 목록, 기준 {(나눠 보기, 구간): {기간: [수익률 합, 건수, 상승 건수]}}, 1:2:6 회차 목록,
           필터 비교 {조합 이름: {"sig": 1차 매수 60일 수익률 목록, "cycles": 회차 목록}})
    신호·회차는 현재 PARAMS 의 1차 매수 필터를 적용한 결과, 필터 비교는 ENTRY_COMBOS 각각을 적용한 결과.
    거래대금 필터는 신호 당일 기준 20일 평균. 기준은 같은 종목·같은 필터·같은 구간의 '아무 날' 매수."""
    recs, cycles, base, compare = [], [], {}, {}
    min_amt = SCAN[f"min_amount_{market}"]
    start = PARAMS["macd_slow"] + PARAMS["macd_signal"]
    items = list(data.items())

    def add_base(key, h, fwd):
        b = base.setdefault(key, {k: [0.0, 0, 0] for k in HORIZONS})[h]
        b[0] += fwd.sum()
        b[1] += len(fwd)
        b[2] += int((fwd > 0).sum())

    for n, (code, df) in enumerate(items, 1):
        if cancel and cancel.is_set():
            return None
        add_indicators(df)
        idx = attach_index(df, code, meta)
        events, _ = simulate(df)
        C, L, H = df["종가"].to_numpy(), df["저가"].to_numpy(), df["고가"].to_numpy()
        liquid = ((df["종가"] * df["거래량"]).rolling(20).mean() >= min_amt).to_numpy()
        feats = day_feature_codes(df, market, idx)
        sector = meta["sector"].get(code, "미분류")

        def feat_at(i):
            out = {f: (_labels(f, market)[c[i]] if c[i] >= 0 else None) for f, c in feats.items()}
            out[SECTOR] = sector
            return out

        for h in HORIZONS:
            if len(C) <= start + h:
                continue
            fwd = C[start + h:] / C[start:-h] - 1
            ok = liquid[start:-h]
            add_base(("", "전체"), h, fwd[ok])
            add_base((SECTOR, sector), h, fwd[ok])
            for f, c in feats.items():
                cs = c[start:-h]
                for k, label in enumerate(_labels(f, market)):
                    add_base((f, label), h, fwd[ok & (cs == k)])

        for e in events:
            i = e["i"]
            if e["kind"] != "buy" or not liquid[i]:
                continue
            r = {"market": market, "code": code, "label": e["label"], "date": e["date"], "feat": feat_at(i)}
            for h in HORIZONS:
                r[h] = C[i + h] / C[i] - 1 if i + h < len(C) else None
            full = i + RISK_WINDOW < len(C)
            r["mdd"] = L[i + 1:i + 1 + RISK_WINDOW].min() / C[i] - 1 if full else None
            r["mfu"] = H[i + 1:i + 1 + RISK_WINDOW].max() / C[i] - 1 if full else None
            recs.append(r)

        for c in backtest(df, events, PARAMS["capital"])["cycles"]:  # 수익률만 쓰므로 환율 무관
            i = df.index.get_loc(c["start"])
            if liquid[i]:
                cycles.append({"market": market, "feat": feat_at(i), **c})

        for name, combo in ENTRY_COMBOS:
            ev, _ = simulate(df, combo)
            comp = compare.setdefault(name, {"sig": [], "cycles": []})
            comp["sig"] += [C[e["i"] + 60] / C[e["i"]] - 1 if e["i"] + 60 < len(C) else None
                            for e in ev if e["label"] == "1차 매수" and liquid[e["i"]]]
            comp["cycles"] += [c for c in backtest(df, ev, PARAMS["capital"])["cycles"]
                               if liquid[df.index.get_loc(c["start"])]]
        if progress and (n % 50 == 0 or n == len(items)):
            progress(n, len(items))
    return recs, base, cycles, compare


def summarize_compare(compare):
    """필터 조합별 1차 매수 60일 성과와 회차 성과 (물린 비율 등)."""
    rows = []
    for name, _ in ENTRY_COMBOS:
        comp = compare.get(name, {"sig": [], "cycles": []})
        s = _summ(comp["sig"])
        rows.append({"name": name, "n": len(comp["sig"]), "r60": s[0] if s else None, "w60": s[2] if s else None,
                     **summarize_cycles(comp["cycles"])[0]})
    return rows


def _summ(vals):
    """수익률 목록 → (평균, 중앙값, 승률). 비어 있으면 None."""
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    mid = len(v) // 2
    median = v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2
    return sum(v) / len(v), median, sum(1 for x in v if x > 0) / len(v)


SIGNAL_GROUPS = ("1차 매수", "2차 매수", "3차 매수", "2차 물타기", "3차 물타기", "매수 전체")


def _signal_row(name, rs, b):
    row = {"group": name, "n": len(rs), "mdd": _summ(r["mdd"] for r in rs), "mfu": _summ(r["mfu"] for r in rs)}
    for h in HORIZONS:
        row[h] = _summ(r[h] for r in rs)
        s, cnt, _ = b[h] if b else (0, 0, 0)
        row[f"base{h}"] = s / cnt if cnt else None
    return row


def summarize_signals(recs, base, feature=None, group="1차 매수", min_n=20):
    """feature 가 없으면 신호 종류별 행 + 기준 행.
    있으면 group 신호를 feature 구간별로 나눈 행 (각 행의 base{h} 는 같은 구간의 '아무 날' 평균)."""
    overall = base.get(("", "전체"))
    if feature is None:
        rows = []
        for g in SIGNAL_GROUPS:
            rs = recs if g == "매수 전체" else [r for r in recs if r["label"] == g]
            if rs:
                rows.append(_signal_row(g, rs, overall))
        base_row = {"group": "기준: 아무 날 매수", "n": overall[HORIZONS[0]][1] if overall else 0,
                    "mdd": None, "mfu": None, "is_base": True}
        for h in HORIZONS:
            s, cnt, up = overall[h] if overall else (0, 0, 0)
            base_row[h] = (s / cnt, None, up / cnt) if cnt else None
        return rows + [base_row]

    rs_all = recs if group == "매수 전체" else [r for r in recs if r["label"] == group]
    by = {}
    for r in rs_all:
        by.setdefault(r["feat"].get(feature) or "판단 불가", []).append(r)
    return [_signal_row(l, by[l], base.get((feature, l))) for l in _order(feature, by) if len(by[l]) >= min_n]


def summarize_cycles(cycles, feature=None, min_n=20):
    """1:2:6 회차(첫 매수 ~ 전량 매도) 성과. feature 가 있으면 첫 매수일의 구간별로 나눈 행 목록.
    stuck = 손실 상태로 아직 보유 중인 회차 비율 (떨어지는 칼날에 물린 비율)."""
    def one(name, cs):
        done = [c for c in cs if c["end"] is not None]
        open_ = [c for c in cs if c["end"] is None]
        s = _summ(c["ret"] for c in done)
        stuck = sum(1 for c in open_ if c["ret"] < 0)
        return {"group": name, "n": len(cs), "done": len(done), "win": s[2] if s else None,
                "avg": s[0] if s else None, "median": s[1] if s else None,
                "days": sum(c["days"] for c in done) / len(done) if done else None,
                "open": len(open_), "open_loss": stuck, "stuck": stuck / len(cs) if cs else None,
                "open_avg": sum(c["ret"] for c in open_) / len(open_) if open_ else None,
                "worst": min((c["ret"] for c in open_), default=None)}

    if feature is None:
        return [one("전체", cycles)]
    by = {}
    for c in cycles:
        by.setdefault(c["feat"].get(feature) or "판단 불가", []).append(c)
    return [one(l, by[l]) for l in _order(feature, by) if len(by[l]) >= min_n]


# ── 펀더멘털: 증수·증익·증배 ─────────────────────────────────────────────────
# 공통 지표: rev_g/ni_g(최근 연간 매출/이익 전년 대비 %), dps_p→dps(직전→최근 연간 주당배당)
# rev_q/ni_q(최근 분기 매출/이익 전년 동기 대비 %), ni(최근 연간 이익), div(배당수익률 %), profit(이익 항목 이름)
# 이익은 후지모토 기준대로 경상/순이익 기준: 한국은 네이버 '당기순이익', 미국은 'Pretax Income'(세전이익, 경상이익에 해당)

def num(s):
    """'9.54배', '22,292원', '0.78%' → float. '-'·빈값·NaN·숫자 아님 → None."""
    if isinstance(s, (int, float)):
        return None if isinstance(s, float) and math.isnan(s) else float(s)
    m = re.search(r"-?\d+(\.\d+)?", str(s or "").replace(",", ""))
    return float(m.group()) if m else None


def _naver_json(code, path):
    r = requests.get(f"https://m.stock.naver.com/api/stock/{code}/{path}",
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    r.raise_for_status()
    return r.json()


def _naver_rows(fin):
    """네이버 재무표 → {항목: [실적값 (오래된→최근)]}. 컨센서스(추정치) 열은 제외."""
    info = fin["financeInfo"]
    keys = [t["key"] for t in info["trTitleList"] if t["isConsensus"] == "N"]
    return {r["title"]: [num(r["columns"].get(k, {}).get("value")) for k in keys] for r in info["rowList"]}


def _pair(vals):
    """값 목록(오래된→최근)에서 (직전, 최근) 실적. 비어 있는 값은 건너뜀."""
    v = [x for x in vals if x is not None]
    return (v[-2] if len(v) >= 2 else None), (v[-1] if v else None)


def _growth(prev, cur):
    if prev is None or cur is None or prev == 0:
        return None
    return (cur - prev) / abs(prev) * 100


def _yoy(vals):
    """분기 실적 목록(오래된→최근)에서 최근 분기의 전년 동기 대비 증가율(%)."""
    v = [x for x in vals if x is not None]
    return _growth(v[-5], v[-1]) if len(v) >= 5 else None


def fundamentals_kr(code):
    total = {i["code"]: num(i.get("value")) for i in _naver_json(code, "integration")["totalInfos"]}
    ann = _naver_rows(_naver_json(code, "finance/annual"))
    qtr = _naver_rows(_naver_json(code, "finance/quarter"))
    rev_p, rev = _pair(ann.get("매출액", []))
    ni_p, ni = _pair(ann.get("당기순이익", []))
    dps = [0.0 if v is None else v for v in ann.get("주당배당금", [])]  # '-' 는 무배당
    return {"rev_g": _growth(rev_p, rev), "ni_g": _growth(ni_p, ni), "ni": ni,
            "dps_p": dps[-2] if len(dps) >= 2 else None, "dps": dps[-1] if dps else None,
            "rev_q": _yoy(qtr.get("매출액", [])), "ni_q": _yoy(qtr.get("당기순이익", [])),
            "div": total.get("dividendYieldRatio"), "profit": "순이익"}


def fundamentals_us(code):
    t = yf.Ticker(code)

    def series(df, row):  # yfinance 재무표 행 → 오래된→최근 값 목록
        if df is None or df.empty or row not in df.index:
            return []
        return [num(v) for v in df.loc[row].values][::-1]

    inc, q = t.income_stmt, t.quarterly_income_stmt
    profit_row = "Pretax Income" if "Pretax Income" in inc.index else "Net Income"
    rev_p, rev = _pair(series(inc, "Total Revenue"))
    ni_p, ni = _pair(series(inc, profit_row))
    d = t.dividends
    by_year = d.groupby(d.index.year).sum() if len(d) else {}
    y = datetime.now().year  # 올해는 아직 다 지급되지 않았으므로 직전 두 해를 비교
    return {"rev_g": _growth(rev_p, rev), "ni_g": _growth(ni_p, ni), "ni": ni,
            "dps_p": float(by_year.get(y - 2, 0.0)), "dps": float(by_year.get(y - 1, 0.0)),
            "rev_q": _yoy(series(q, "Total Revenue")), "ni_q": _yoy(series(q, profit_row)),
            "div": num(t.info.get("dividendYield")),
            "profit": "세전이익" if profit_row == "Pretax Income" else "순이익"}


def fundamental_score(f):
    """증수·증익·증배 판정과 매수 점수(0~100, 매력도) · 매도 점수(0~100, 위험도) · 근거.
    매수 점수는 값이 있는 항목 만점 대비 비율, 매도 점수는 해당하는 위험 항목 점수의 합."""
    up = lambda v: None if v is None else v > 0
    dps_up = None if f["dps"] is None else f["dps"] > (f["dps_p"] or 0)
    flags = {"증수": up(f["rev_g"]), "증익": up(f["ni_g"]), "증배": dps_up}

    got, avail, why_b = 0, 0, []

    def buy(label, ok, pts):
        nonlocal got, avail
        if ok is None:
            return
        avail += pts
        if ok:
            got += pts
            why_b.append(f"{label}(+{pts})")

    pct = lambda v: f" {v:+.0f}%" if v is not None else ""
    buy(f"증수{pct(f['rev_g'])}", flags["증수"], 25)
    buy(f"증익{pct(f['ni_g'])}", flags["증익"], 25)
    buy("증배", flags["증배"], 15)
    buy(f"최근분기 매출{pct(f['rev_q'])}", up(f["rev_q"]), 10)
    buy(f"최근분기 이익{pct(f['ni_q'])}", up(f["ni_q"]), 15)
    if f["div"] is not None:  # 후지모토: 고배당보다 성장성, 배당수익률은 3% 정도면 좋은 편
        avail += 10
        pts = 10 if f["div"] >= 3 else 5 if f["div"] >= 1.5 else 0
        if pts:
            got += pts
            why_b.append(f"배당수익률 {f['div']:.1f}%(+{pts})")
    buy_score = round(got / avail * 100) if avail >= 50 else None  # 지표가 너무 적으면 판단 보류

    sell, why_s = 0, []

    def risk(label, cond, pts):
        nonlocal sell
        if cond:
            sell += pts
            why_s.append(f"{label}(+{pts})")

    risk(f"감수{pct(f['rev_g'])}", f["rev_g"] is not None and f["rev_g"] < 0, 20)
    risk(f"감익{pct(f['ni_g'])}", f["ni_g"] is not None and f["ni_g"] < 0, 20)
    risk("감배", f["dps"] is not None and f["dps_p"] is not None and f["dps"] < f["dps_p"], 15)
    risk(f"최근분기 이익 감소{pct(f['ni_q'])}", f["ni_q"] is not None and f["ni_q"] < 0, 20)
    risk(f"최근분기 매출 감소{pct(f['rev_q'])}", f["rev_q"] is not None and f["rev_q"] < 0, 10)
    risk("적자", f["ni"] is not None and f["ni"] < 0, 15)

    gate = bool(flags["증수"] and flags["증익"])
    if flags["증수"] is None or flags["증익"] is None:
        verdict = "데이터 부족"
    elif gate:
        verdict = "증수·증익·증배" if flags["증배"] else "증수·증익"
    else:
        verdict = "미충족"
    return {"buy": buy_score, "sell": sell, "gate": gate, "verdict": verdict, "flags": flags,
            "why_buy": why_b, "why_sell": why_s}


def fund_summary(f, market):
    g = lambda v: "-" if v is None else f"{v:+.0f}%"
    dps = lambda v: "-" if v is None else (f"${v:,.2f}" if market == "US" else f"{v:,.0f}원")
    return (f"연간 매출 {g(f['rev_g'])} · {f['profit']} {g(f['ni_g'])} · 주당배당 {dps(f['dps_p'])}→{dps(f['dps'])}"
            f" · 최근분기 매출 {g(f['rev_q'])} / {f['profit']} {g(f['ni_q'])}"
            f" · 배당수익률 {'-' if f['div'] is None else f'{f['div']:.1f}%'}")


def load_fundamentals(keys, refresh, progress):
    """[(시장, 코드)] → {(시장, 코드): {지표..., 점수...} 또는 None}. 같은 날 받은 값은 캐시 재사용."""
    cache = ({} if refresh else _load_cache("GROWTH")) or {}
    todo = [k for k in keys if k not in cache]

    def one(key):
        mk, code = key
        try:
            f = fundamentals_us(code) if mk == "US" else fundamentals_kr(code)
            return key, {**f, **fundamental_score(f)}
        except Exception:
            return key, None

    with ThreadPoolExecutor(8) as ex:
        for n, (key, val) in enumerate(ex.map(one, todo), 1):
            cache[key] = val
            progress(n, len(todo))
    if todo:
        _save_cache("GROWTH", cache)
    return {k: cache.get(k) for k in keys}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("후지모토 1:2:6 매매 판단기")
        self.geometry("1300x900")

        fbar = ttk.Frame(self, padding=(10, 6, 10, 0))
        fbar.pack(fill="x")
        ttk.Label(fbar, text="1차 매수 필터:", font=("Malgun Gothic", 10, "bold")).pack(side="left")
        self.f_mkt = tk.BooleanVar(value=PARAMS["entry_market_drop"] is not None)
        self.f_rel = tk.BooleanVar(value=PARAMS["entry_max_rel_drop"] is not None)
        ttk.Checkbutton(fbar, text="지수 급락 시에만 (지수 20일 -5% 이하)", variable=self.f_mkt,
                        command=self._apply_filters).pack(side="left", padx=6)
        ttk.Checkbutton(fbar, text="시장보다 15%p 이상 더 빠진 종목 제외", variable=self.f_rel,
                        command=self._apply_filters).pack(side="left", padx=6)
        ttk.Label(fbar, foreground="#666", text=(
            "— 신호 성적표에서 효과가 확인된 조건 · 바꾸면 분석·검색·계산을 다시 실행하세요")).pack(side="left", padx=6)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)
        self.tab_analyze = ttk.Frame(self.nb)
        self.tab_scan = ttk.Frame(self.nb)
        self.nb.add(self.tab_analyze, text="  종목 분석  ")
        self.nb.add(self.tab_scan, text="  종목 검색  ")
        self.tab_stats = ttk.Frame(self.nb)
        self.nb.add(self.tab_stats, text="  신호 성적표  ")
        self._build_analyze(self.tab_analyze)
        self._build_scan(self.tab_scan)
        self._build_stats(self.tab_stats)

    def _apply_filters(self):
        mkt, rel = ENTRY_COMBOS[-1][1]
        PARAMS["entry_market_drop"] = mkt if self.f_mkt.get() else None
        PARAMS["entry_max_rel_drop"] = rel if self.f_rel.get() else None

    def _build_analyze(self, tab):
        top = ttk.Frame(tab, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="종목명 / 코드:").pack(side="left")
        self.entry = ttk.Entry(top, width=20, font=("Malgun Gothic", 11))
        self.entry.pack(side="left", padx=6)
        self.entry.bind("<Return>", lambda e: self.run())
        ttk.Label(top, text="종목당 한도(원):").pack(side="left", padx=(10, 0))
        self.capital = ttk.Entry(top, width=13)
        self.capital.insert(0, f"{PARAMS['capital']:,}")
        self.capital.pack(side="left", padx=4)
        self.btn = ttk.Button(top, text="분석", command=self.run)
        self.btn.pack(side="left", padx=6)
        self.status = ttk.Label(top, text="예: 삼성전자, 005930, 애플, TSLA")
        self.status.pack(side="left", padx=8)

        self.verdict = tk.Label(tab, text="", font=("Malgun Gothic", 17, "bold"))
        self.verdict.pack(pady=(2, 0))
        self.next = ttk.Label(tab, text="", font=("Malgun Gothic", 11), wraplength=1250, justify="center")
        self.next.pack()
        self.plan = ttk.Label(tab, text="", font=("Malgun Gothic", 10), wraplength=1250, justify="center")
        self.plan.pack()

        self.tree = ttk.Treeview(tab, columns=("result", "value"), height=3)
        self.tree.heading("#0", text="판단 근거")
        self.tree.heading("result", text="현재")
        self.tree.heading("value", text="값")
        self.tree.column("#0", width=300)
        self.tree.column("result", width=120, anchor="center")
        self.tree.column("value", width=520)
        self.tree.pack(fill="x", padx=10, pady=(6, 2))

        self.perf = tk.Label(tab, text="", font=("Malgun Gothic", 10, "bold"), justify="left", anchor="w", wraplength=1260)
        self.perf.pack(fill="x", padx=12, pady=(4, 0))

        tables = ttk.Frame(tab)
        tables.pack(fill="x", padx=10, pady=(2, 4))
        cols = ("action", "price", "held", "why")
        self.log = ttk.Treeview(tables, columns=cols, height=6)
        for col, text, w in (("#0", "날짜", 85), ("action", "신호", 100), ("price", "가격", 85),
                             ("held", "보유 · 평단/수익", 170), ("why", "근거", 260)):
            self.log.heading(col, text=text)
            self.log.column(col, width=w, anchor="w" if col in ("#0", "why") else "center")
        sb = ttk.Scrollbar(tables, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)

        cols = ("end", "days", "buys", "invested", "pnl", "ret")
        self.cycles = ttk.Treeview(tables, columns=cols, height=6)
        for col, text, w in (("#0", "회차 시작", 85), ("end", "종료", 85), ("days", "보유일", 50),
                             ("buys", "매수", 40), ("invested", "투입", 100), ("pnl", "손익", 105),
                             ("ret", "수익률", 65)):
            self.cycles.heading(col, text=text)
            self.cycles.column(col, width=w, anchor="w" if col == "#0" else "center")
        self.cycles.tag_configure("win", foreground="#c62828")
        self.cycles.tag_configure("loss", foreground="#1565c0")
        sb2 = ttk.Scrollbar(tables, orient="vertical", command=self.cycles.yview)
        self.cycles.configure(yscrollcommand=sb2.set)
        # 회차 표를 먼저 오른쪽에 붙여 항상 보이게 하고, 신호 표가 남는 폭을 쓰도록 (화면 배율이 커도 잘리지 않게)
        sb2.pack(side="right", fill="y")
        self.cycles.pack(side="right", padx=(10, 0))
        self.log.pack(side="left", fill="x", expand=True)
        sb.pack(side="left", fill="y")

        # 긴 안내 문구는 창 폭에 맞춰 줄바꿈
        tab.bind("<Configure>", lambda e: [w.config(wraplength=max(400, e.width - 40))
                                           for w in (self.verdict, self.next, self.plan, self.perf)])

        self.fig = Figure(figsize=(10, 5.4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _build_scan(self, tab):
        opt = ttk.Frame(tab, padding=10)
        opt.pack(fill="x")
        ttk.Label(opt, text="시장:").pack(side="left")
        self.scan_kr = tk.BooleanVar(value=True)
        self.scan_us = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="한국(코스피·코스닥)", variable=self.scan_kr).pack(side="left", padx=4)
        ttk.Checkbutton(opt, text="미국(S&P500)", variable=self.scan_us).pack(side="left", padx=4)

        ttk.Label(opt, text="  최근").pack(side="left")
        self.scan_days = ttk.Spinbox(opt, from_=1, to=20, width=4)
        self.scan_days.set(SCAN["recent_days"])
        self.scan_days.pack(side="left", padx=2)
        ttk.Label(opt, text="거래일 이내 매수 신호").pack(side="left")

        self.scan_gate = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="증수·증익 종목만", variable=self.scan_gate).pack(side="left", padx=(12, 4))
        self.scan_refresh = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="시세·실적 새로 받기", variable=self.scan_refresh).pack(side="left", padx=4)
        self.scan_btn = ttk.Button(opt, text="검색", command=self.scan)
        self.scan_btn.pack(side="left", padx=4)
        self.cancel_btn = ttk.Button(opt, text="취소", command=lambda: self.scan_cancel.set(), state="disabled")
        self.cancel_btn.pack(side="left")
        self.scan_cancel = threading.Event()

        prog = ttk.Frame(tab, padding=(10, 0))
        prog.pack(fill="x")
        self.scan_bar = ttk.Progressbar(prog, length=300)
        self.scan_bar.pack(side="left")
        self.scan_status = ttk.Label(prog, text=(
            f"최근 N거래일 안에 1:2:6 매수(물타기 포함) 신호가 나왔고 아직 보유 중인 종목을 찾습니다. "
            f"(스팩·우선주·관리종목 및 20일 평균 거래대금 "
            f"{fmt_amount(SCAN['min_amount_KR'], 'KR')} / {fmt_amount(SCAN['min_amount_US'], 'US')} 미만 제외)"))
        self.scan_status.pack(side="left", padx=10)

        cols = ("market", "code", "signal", "date", "price", "close", "held", "pnl", "amount",
                "buy", "sell", "fund", "rev_g", "ni_g", "next")
        frame = ttk.Frame(tab, padding=10)
        frame.pack(fill="both", expand=True)
        self.result = ttk.Treeview(frame, columns=cols)
        for col, text, w in (("#0", "종목명", 140), ("market", "시장", 40), ("code", "코드", 60),
                             ("signal", "신호", 80), ("date", "신호일", 85), ("price", "신호가", 80),
                             ("close", "현재가", 80), ("held", "보유", 40), ("pnl", "평단대비", 65),
                             ("amount", "거래대금", 70), ("buy", "매수점수", 60), ("sell", "매도점수", 60),
                             ("fund", "실적", 95), ("rev_g", "매출%", 55), ("ni_g", "이익%", 55),
                             ("next", "다음 단계", 300)):
            self.result.heading(col, text=text, command=lambda c=col: self._sort_result(c))
            self.result.column(col, width=w, anchor="w" if col in ("#0", "next") else "center")
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.result.yview)
        self.result.configure(yscrollcommand=sb.set)
        self.result.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.result.tag_configure("good", foreground="#c62828")
        self.result.tag_configure("bad", foreground="#1565c0")
        self.result.bind("<Double-1>", self._open_result)
        self.result.bind("<<TreeviewSelect>>", self._show_reason)
        self.scan_rows, self.sort_desc = {}, {}
        self.reason = ttk.Label(tab, text="", wraplength=1250, padding=(10, 0, 10, 2), font=("Malgun Gothic", 10))
        self.reason.pack(anchor="w")
        ttk.Label(tab, text=("결과를 선택하면 점수 근거를, 더블클릭하면 '종목 분석' 탭에서 차트와 함께 분석합니다. "
                             "열 제목을 누르면 정렬됩니다.  매수점수: 증수·증익·증배·최근분기 성장·배당 (높을수록 매력)  "
                             "/ 매도점수: 감수·감익·감배·최근분기 감소·적자 (높을수록 위험)"),
                  wraplength=1250, padding=(10, 0, 10, 8), foreground="#666").pack(anchor="w")

    def scan(self):
        markets = [m for m, v in (("KR", self.scan_kr), ("US", self.scan_us)) if v.get()]
        try:
            days = int(self.scan_days.get())
        except ValueError:
            days = SCAN["recent_days"]
        if not markets:
            messagebox.showerror("오류", "시장을 하나 이상 선택하세요.")
            return
        self.scan_cancel.clear()
        self.scan_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.result.delete(*self.result.get_children())
        self.reason.config(text="")
        threading.Thread(target=self._scan_work,
                         args=(markets, days, self.scan_gate.get(), self.scan_refresh.get()), daemon=True).start()

    def _scan_work(self, markets, days, gate_only, refresh):
        try:
            rows = []
            for mk in markets:
                label = "한국" if mk == "KR" else "미국"
                self.after(0, self._scan_progress, f"{label} 종목 목록·시세 불러오는 중...", 0, 1)
                progress = lambda n, total, label=label: self.after(
                    0, self._scan_progress, f"{label} 시세 받는 중 {n:,}/{total:,}", n, total)
                data, names = load_market_data(mk, refresh, progress, self.scan_cancel)
                if data is None:
                    self.after(0, self._scan_done, None)
                    return
                self.after(0, self._scan_progress, f"{label} {len(data):,}종목 신호 계산 중...", 1, 1)
                rows += scan_signals(data, names, mk, days, get_meta(mk))
            fund = load_fundamentals(
                [(r["market"], r["code"]) for r in rows], refresh,
                lambda n, total: self.after(0, self._scan_progress, f"실적(증수·증익·증배) 확인 중 {n}/{total}", n, total))
            for r in rows:
                r["fund"] = fund.get((r["market"], r["code"]))
            total = len(rows)
            if gate_only:
                rows = [r for r in rows if r["fund"] and r["fund"]["gate"]]
            self.after(0, self._scan_done, rows, None, total)
        except Exception as e:
            self.after(0, self._scan_done, None, str(e))

    def _scan_progress(self, text, n, total):
        self.scan_status.config(text=text)
        self.scan_bar.config(maximum=total, value=n)

    def _scan_done(self, rows, error=None, total=0):
        self.scan_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        if error:
            self.scan_status.config(text="")
            messagebox.showerror("오류", error)
            return
        if rows is None:
            self.scan_status.config(text="검색을 취소했습니다.")
            return
        self.scan_rows.clear()
        opt = lambda v, f="{:+.0f}": "-" if v is None else f.format(v)
        for r in rows:
            mk, fd = r["market"], r["fund"] or {}
            iid = f"{mk}:{r['code']}"  # 코드는 iid 에 보관 (values 에서 읽으면 '005930' 이 숫자 5930 으로 바뀜)
            self.scan_rows[iid] = r
            verdict = fd.get("verdict", "조회 실패")
            tag = "good" if fd.get("gate") else "bad" if verdict == "미충족" else ""
            self.result.insert("", "end", iid=iid, text=r["name"], tags=(tag,), values=(
                "한국" if mk == "KR" else "미국", r["code"], f"{r['label']} +{r['units']}",
                r["date"].strftime("%Y-%m-%d"), fmt_price(r["price"], mk), fmt_price(r["close"], mk),
                f"{r['held']}/{r['full']}", opt(r["pnl"] * 100 if r["pnl"] is not None else None, "{:+.1f}%"),
                fmt_amount(r["amount"], mk), opt(fd.get("buy"), "{}"), opt(fd.get("sell"), "{}") if fd else "-",
                verdict, opt(fd.get("rev_g")), opt(fd.get("ni_g")), r["next"]))
        hidden = f", 실적 미충족 {total - len(rows)}종목 숨김" if total > len(rows) else ""
        self.scan_status.config(text=f"검색 완료: {len(rows)}종목{hidden} ({datetime.now():%H:%M} 기준)")

    def _sort_result(self, col):
        """열 제목 클릭 정렬 (첫 클릭은 내림차순). 숫자 열은 화면 문자열이 아닌 원래 값으로, 값 없는 행은 항상 아래로."""
        desc = self.sort_desc[col] = not self.sort_desc.get(col, False)

        def value(iid):
            r = self.scan_rows[iid]
            if col in ("buy", "sell", "rev_g", "ni_g"):
                return (r["fund"] or {}).get(col)
            if col in ("price", "close", "held", "pnl"):
                return r[col]
            if col == "amount":
                return r["market"], r["amount"]  # 원·달러가 섞이므로 시장별로 묶어 정렬
            return self.result.item(iid)["text"] if col == "#0" else self.result.set(iid, col)

        items = self.result.get_children()
        has = [i for i in items if value(i) is not None]
        ordered = sorted(has, key=value, reverse=desc) + [i for i in items if value(i) is None]
        for pos, iid in enumerate(ordered):
            self.result.move(iid, "", pos)

    def _show_reason(self, _event):
        sel = self.result.selection()
        if not sel or sel[0] not in self.scan_rows:
            return
        r = self.scan_rows[sel[0]]
        fd = r["fund"]
        if not fd:
            self.reason.config(text=f"{r['name']}: 실적 데이터를 가져오지 못했습니다.")
            return
        self.reason.config(text=(
            f"{r['name']}  ▶ {fund_summary(fd, r['market'])}\n"
            f"매수 {fd['buy'] if fd['buy'] is not None else '-'}점: {', '.join(fd['why_buy']) or '해당 없음'}"
            f"   ▶ 매도 {fd['sell']}점: {', '.join(fd['why_sell']) or '해당 없음'}"))

    def _open_result(self, event):
        item = self.result.identify_row(event.y)
        if not item:
            return
        self.entry.delete(0, "end")
        self.entry.insert(0, item.split(":", 1)[1])
        self.nb.select(self.tab_analyze)
        self.run()

    def _build_stats(self, tab):
        opt = ttk.Frame(tab, padding=(10, 10, 10, 4))
        opt.pack(fill="x")
        ttk.Label(opt, text="시장:").pack(side="left")
        self.stats_kr = tk.BooleanVar(value=True)
        self.stats_us = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="한국(코스피·코스닥)", variable=self.stats_kr).pack(side="left", padx=4)
        ttk.Checkbutton(opt, text="미국(S&P500)", variable=self.stats_us).pack(side="left", padx=4)
        ttk.Label(opt, text="   기간:").pack(side="left")
        self.stats_days = tk.IntVar(value=730)
        ttk.Radiobutton(opt, text="2년 (검색 탭 시세 재사용)", value=730, variable=self.stats_days).pack(side="left", padx=4)
        ttk.Radiobutton(opt, text="5년 (처음엔 몇 분 소요)", value=1825, variable=self.stats_days).pack(side="left", padx=4)
        self.stats_btn = ttk.Button(opt, text="계산", command=self.calc_stats)
        self.stats_btn.pack(side="left", padx=(12, 4))
        self.stats_cancel_btn = ttk.Button(opt, text="취소", command=lambda: self.stats_cancel.set(), state="disabled")
        self.stats_cancel_btn.pack(side="left")
        self.stats_cancel = threading.Event()

        view = ttk.Frame(tab, padding=(10, 0, 10, 4))
        view.pack(fill="x")
        ttk.Label(view, text="나눠 보기:").pack(side="left")
        self.stats_feature = ttk.Combobox(view, values=("없음 (신호 종류별)",) + FEATURES, state="readonly", width=26)
        self.stats_feature.current(0)
        self.stats_feature.pack(side="left", padx=4)
        ttk.Label(view, text="   대상 신호:").pack(side="left")
        self.stats_group = ttk.Combobox(view, values=SIGNAL_GROUPS, state="readonly", width=12)
        self.stats_group.current(0)
        self.stats_group.pack(side="left", padx=4)
        for cb in (self.stats_feature, self.stats_group):
            cb.bind("<<ComboboxSelected>>", lambda e: self._render_stats())
        ttk.Label(view, foreground="#666", text=(
            "  ← 계산 후 바꾸면 바로 다시 보여 줍니다. 구간은 모두 신호 당일 기준이며, '기준'은 같은 구간에서 아무 날 샀을 때입니다.")
        ).pack(side="left")

        prog = ttk.Frame(tab, padding=(10, 0))
        prog.pack(fill="x")
        self.stats_bar = ttk.Progressbar(prog, length=300)
        self.stats_bar.pack(side="left")
        self.stats_status = ttk.Label(prog, text=(
            "검색 대상 전 종목에서 과거 매수 신호를 모두 찾아, 신호 뒤 실제로 올랐는지를 '아무 날 매수'와 비교합니다."))
        self.stats_status.pack(side="left", padx=10)

        self.stats_title = ttk.Label(tab, font=("Malgun Gothic", 10, "bold"), padding=(10, 8, 10, 2))
        self.stats_title.pack(anchor="w")
        cols = ("group", "n", "r20", "w20", "r60", "m60", "w60", "b60", "x60", "r120", "w120", "mdd", "mfu")
        frame = ttk.Frame(tab)
        frame.pack(fill="x", padx=10)
        self.stats_tree = ttk.Treeview(frame, columns=cols, height=9)
        self.stats_tree.heading("#0", text="시장")
        self.stats_tree.column("#0", width=50, anchor="center")
        for col, text, w in (("group", "신호 / 구간", 190), ("n", "건수", 60), ("r20", "20일 평균", 70),
                             ("w20", "20일 승률", 65), ("r60", "60일 평균", 70), ("m60", "60일 중앙값", 75),
                             ("w60", "60일 승률", 65), ("b60", "60일 기준", 70), ("x60", "기준 대비", 75),
                             ("r120", "120일 평균", 75), ("w120", "120일 승률", 70),
                             ("mdd", f"{RISK_WINDOW}일 내 최대낙폭", 105), ("mfu", f"{RISK_WINDOW}일 내 최대상승", 105)):
            self.stats_tree.heading(col, text=text)
            self.stats_tree.column(col, width=w, anchor="w" if col == "group" else "center")
        self.stats_tree.tag_configure("good", foreground="#c62828")
        self.stats_tree.tag_configure("bad", foreground="#1565c0")
        self.stats_tree.tag_configure("base", foreground="#666", background="#f2f2f2")
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.stats_tree.yview)
        self.stats_tree.configure(yscrollcommand=sb.set)
        self.stats_tree.pack(side="left", fill="x", expand=True)
        sb.pack(side="left", fill="y")

        ttk.Label(tab, text=("1:2:6 회차 성과 (첫 매수 ~ 전량 매도를 한 회차로, 보유 중 회차는 오늘 종가 평가) — "
                             "'물린 비율' = 손실 상태로 아직 못 판 회차 비율 (낮을수록 좋음)"),
                  font=("Malgun Gothic", 10, "bold"), padding=(10, 8, 10, 2)).pack(anchor="w")
        cols = ("group", "n", "done", "win", "avg", "days", "open", "stuck", "open_avg", "worst")
        frame = ttk.Frame(tab)
        frame.pack(fill="x", padx=10)
        self.cycle_tree = ttk.Treeview(frame, columns=cols, height=5)
        self.cycle_tree.heading("#0", text="시장")
        self.cycle_tree.column("#0", width=50, anchor="center")
        for col, text, w in (("group", "구간", 190), ("n", "회차", 60), ("done", "완료", 60), ("win", "완료 승률", 70),
                             ("avg", "완료 평균", 75), ("days", "평균 보유일", 80), ("open", "보유 중", 60),
                             ("stuck", "물린 비율", 75), ("open_avg", "보유 중 평균", 85), ("worst", "최악", 70)):
            self.cycle_tree.heading(col, text=text)
            self.cycle_tree.column(col, width=w, anchor="w" if col == "group" else "center")
        self.cycle_tree.tag_configure("good", foreground="#c62828")
        self.cycle_tree.tag_configure("bad", foreground="#1565c0")
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.cycle_tree.yview)
        self.cycle_tree.configure(yscrollcommand=sb.set)
        self.cycle_tree.pack(side="left", fill="x", expand=True)
        sb.pack(side="left", fill="y")

        ttk.Label(tab, text=("1차 매수 필터 비교 (상단 체크박스와 무관하게 네 가지 조합을 모두 계산) — "
                             "물린 비율이 줄고 60일 성과가 좋아지는지 확인"),
                  font=("Malgun Gothic", 10, "bold"), padding=(10, 8, 10, 2)).pack(anchor="w")
        cols = ("name", "n", "r60", "w60", "cyc", "done_avg", "stuck", "open_avg", "worst")
        frame = ttk.Frame(tab)
        frame.pack(fill="x", padx=10)
        self.compare_tree = ttk.Treeview(frame, columns=cols, height=5)
        self.compare_tree.heading("#0", text="시장")
        self.compare_tree.column("#0", width=50, anchor="center")
        for col, text, w in (("name", "필터", 190), ("n", "1차 매수 건수", 95), ("r60", "60일 평균", 75),
                             ("w60", "60일 승률", 70), ("cyc", "회차", 60), ("done_avg", "완료 평균", 75),
                             ("stuck", "물린 비율", 75), ("open_avg", "보유 중 평균", 85), ("worst", "최악", 70)):
            self.compare_tree.heading(col, text=text)
            self.compare_tree.column(col, width=w, anchor="w" if col == "name" else "center")
        self.compare_tree.tag_configure("good", foreground="#c62828")
        self.compare_tree.tag_configure("bad", foreground="#1565c0")
        self.compare_tree.tag_configure("base", foreground="#666", background="#f2f2f2")
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.compare_tree.yview)
        self.compare_tree.configure(yscrollcommand=sb.set)
        self.compare_tree.pack(side="left", fill="x", expand=True)
        sb.pack(side="left", fill="y")

        self.stats_fig = Figure(figsize=(10, 2.2), dpi=100)
        self.stats_canvas = FigureCanvasTkAgg(self.stats_fig, master=tab)
        self.stats_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(6, 0))
        self.stats_note = ttk.Label(tab, foreground="#666", padding=(10, 2, 10, 8), text=(
            "주의: ① 지금 상장된 종목만 대상이라 상장폐지 종목이 빠져 결과가 실제보다 좋게 나옵니다(생존 편향). "
            "② 같은 종목의 신호 구간이 겹쳐 서로 독립적이지 않습니다. ③ 기간이 짧으면 그 기간의 상승장/하락장에 크게 좌우됩니다. "
            "④ 수수료·세금 미반영, 실적(증수·증익) 조건은 과거 시점에 적용하지 않았습니다. "
            "⑤ 완료 회차 승률이 100%인 것은 손실 중에는 팔지 않는 규칙 때문이며, 지는 거래는 '보유 중'에 쌓입니다. "
            "⑥ 200일선은 데이터 시작 후 200거래일이 지나야 계산되어 2년 기간에선 '판단 불가'가 많습니다 — 5년을 권장합니다."))
        self.stats_note.pack(anchor="w")
        tab.bind("<Configure>", lambda e: self.stats_note.config(wraplength=max(400, e.width - 40)))
        self.stats_out = None

    def calc_stats(self):
        markets = [m for m, v in (("KR", self.stats_kr), ("US", self.stats_us)) if v.get()]
        if not markets:
            messagebox.showerror("오류", "시장을 하나 이상 선택하세요.")
            return
        self.stats_cancel.clear()
        self.stats_btn.config(state="disabled")
        self.stats_cancel_btn.config(state="normal")
        threading.Thread(target=self._stats_work, args=(markets, self.stats_days.get()), daemon=True).start()

    def _stats_work(self, markets, days):
        try:
            out = {}
            for mk in markets:
                label = "한국" if mk == "KR" else "미국"
                prog = lambda text: (lambda n, total: self.after(
                    0, self._stats_progress, f"{text} {n:,}/{total:,}", n, total))
                self.after(0, self._stats_progress, f"{label} 시세·지수·업종 불러오는 중...", 0, 1)
                data, _ = load_market_data(mk, False, prog(f"{label} 시세 받는 중"), self.stats_cancel, days)
                if data is None:
                    self.after(0, self._stats_done, None)
                    return
                res = signal_stats(data, mk, get_meta(mk), prog(f"{label} 신호 계산 중"), self.stats_cancel)
                if res is None:
                    self.after(0, self._stats_done, None)
                    return
                out[mk] = res
            if len(out) > 1:  # 시장 합계
                base = {}
                for _, b, _, _ in out.values():
                    for key, hs in b.items():
                        tgt = base.setdefault(key, {h: [0.0, 0, 0] for h in HORIZONS})
                        for h in HORIZONS:
                            tgt[h] = [x + y for x, y in zip(tgt[h], hs[h])]
                compare = {name: {"sig": [x for v in out.values() for x in v[3][name]["sig"]],
                                  "cycles": [c for v in out.values() for c in v[3][name]["cycles"]]}
                           for name, _ in ENTRY_COMBOS}
                out["ALL"] = ([r for v in out.values() for r in v[0]], base,
                              [c for v in out.values() for c in v[2]], compare)
            self.after(0, self._stats_done, out, None, days)
        except Exception as e:
            self.after(0, self._stats_done, None, str(e))

    def _stats_progress(self, text, n, total):
        self.stats_status.config(text=text)
        self.stats_bar.config(maximum=total, value=n)

    def _stats_done(self, out, error=None, days=730):
        self.stats_btn.config(state="normal")
        self.stats_cancel_btn.config(state="disabled")
        if error:
            self.stats_status.config(text="")
            messagebox.showerror("오류", error)
            return
        if out is None:
            self.stats_status.config(text="계산을 취소했습니다.")
            return
        self.stats_out = out
        self._render_stats()
        n_sig = sum(len(v[0]) for k, v in out.items() if k != "ALL")
        self.stats_status.config(text=f"계산 완료: 신호 {n_sig:,}건 · 기간 {days // 365}년 ({datetime.now():%H:%M} 기준)")

    def _render_stats(self):
        if not self.stats_out:
            return
        feature = self.stats_feature.get()
        feature = None if feature.startswith("없음") else feature
        group = self.stats_group.get()
        name = {"KR": "한국", "US": "미국", "ALL": "합계"}
        pct = lambda v: "-" if v is None else f"{v:+.1%}"
        rate = lambda v: "-" if v is None else f"{v:.0%}"
        self.stats_title.config(text=(
            "신호별 성과 (평균 수익률 / 승률 = 신호 당일 종가에 샀다면 N거래일 뒤 오른 비율)" if feature is None
            else f"'{group}' 신호를 [{feature}] 구간별로 — 기준 = 같은 구간에서 아무 날 샀을 때 (20건 미만 구간은 생략)"))
        self.stats_tree.delete(*self.stats_tree.get_children())
        self.cycle_tree.delete(*self.cycle_tree.get_children())
        sig_rows, cyc_rows = {}, {}
        self.compare_tree.delete(*self.compare_tree.get_children())
        for mk, (recs, base, cycles, compare) in self.stats_out.items():
            crows = summarize_compare(compare)
            ref = crows[0]["stuck"]
            for c in crows:
                tag = "base" if c is crows[0] else "" if c["stuck"] is None or ref is None else                     "good" if c["stuck"] < ref else "bad"
                self.compare_tree.insert("", "end", text=name[mk], tags=(tag,), values=(
                    c["name"], f"{c['n']:,}", pct(c["r60"]), rate(c["w60"]), f"{c['done'] + c['open']:,}",
                    pct(c["avg"]), rate(c["stuck"]), pct(c["open_avg"]), pct(c["worst"])))
            rows = sig_rows[mk] = summarize_signals(recs, base, feature, group)
            for r in rows:
                s20, s60, s120 = r[20], r[60], r[120]
                is_base = r.get("is_base", False)
                b60 = r.get("base60") if feature is not None else (rows[-1][60][0] if rows[-1][60] else None)
                x60 = None if is_base or not s60 or b60 is None else s60[0] - b60
                tag = "base" if is_base else "" if x60 is None else "good" if x60 > 0 else "bad"
                self.stats_tree.insert("", "end", text=name[mk], tags=(tag,), values=(
                    r["group"], f"{r['n']:,}",
                    pct(s20 and s20[0]), rate(s20 and s20[2]),
                    pct(s60 and s60[0]), pct(s60 and s60[1]), rate(s60 and s60[2]),
                    "" if is_base else pct(b60), "" if is_base else (f"{x60:+.1%}p" if x60 is not None else "-"),
                    pct(s120 and s120[0]), rate(s120 and s120[2]),
                    pct(r["mdd"] and r["mdd"][0]), pct(r["mfu"] and r["mfu"][0])))
            crows = cyc_rows[mk] = summarize_cycles(cycles, feature)
            overall_stuck = summarize_cycles(cycles)[0]["stuck"]
            for c in crows:
                tag = "" if feature is None or c["stuck"] is None else "good" if c["stuck"] < overall_stuck else "bad"
                self.cycle_tree.insert("", "end", text=name[mk], tags=(tag,), values=(
                    c["group"], f"{c['n']:,}", f"{c['done']:,}", rate(c["win"]), pct(c["avg"]),
                    "-" if c["days"] is None else f"{c['days']:.0f}일", f"{c['open']:,}",
                    rate(c["stuck"]), pct(c["open_avg"]), pct(c["worst"])))
        self._draw_stats(sig_rows, cyc_rows, name, feature)

    def _draw_stats(self, sig_rows, cyc_rows, name, feature):
        """나눠 보기 없음: 신호 종류별 20/60/120일 평균 수익률 막대.
        나눠 보기: 구간별 60일 평균(빨강) vs 같은 구간 아무 날(회색) 가로 막대 + 물린 비율 표시."""
        self.stats_fig.clear()
        markets = list(sig_rows)
        colors = {"1차 매수": "#ef9a9a", "2차 매수": "#e57373", "3차 매수": "#c62828",
                  "2차 물타기": "#90caf9", "3차 물타기": "#1565c0"}
        for k, mk in enumerate(markets):
            ax = self.stats_fig.add_subplot(1, len(markets), k + 1)
            if feature is None:
                rows = [r for r in sig_rows[mk] if r["group"] != "매수 전체"]
                width = 0.8 / len(rows)
                for j, r in enumerate(rows):
                    vals = [r[h][0] * 100 if r[h] else 0 for h in HORIZONS]
                    ax.bar([x + j * width for x in range(len(HORIZONS))], vals, width,
                           color=colors.get(r["group"], "#9e9e9e"), label=r["group"].replace("기준: ", ""))
                ax.set_xticks([x + 0.4 - width / 2 for x in range(len(HORIZONS))])
                ax.set_xticklabels([f"{h}일 뒤" for h in HORIZONS])
                ax.axhline(0, color="#999", lw=0.8)
                ax.set_title(f"{name[mk]} — 평균 수익률(%)", fontsize=9)
                ax.grid(axis="y", alpha=0.3)
            else:
                rows = [r for r in sig_rows[mk] if r["group"] != "판단 불가"][:8]
                stuck = {c["group"]: c["stuck"] for c in cyc_rows[mk]}
                ys = list(range(len(rows)))[::-1]
                sig = [r[60][0] * 100 if r[60] else 0 for r in rows]
                base = [r["base60"] * 100 if r["base60"] is not None else 0 for r in rows]
                ax.barh([y + 0.2 for y in ys], sig, 0.4, color="#e57373", label="신호 60일 평균")
                ax.barh([y - 0.2 for y in ys], base, 0.4, color="#bdbdbd", label="같은 구간 아무 날")
                ax.set_yticks(ys)
                ax.set_yticklabels([r["group"][:20] for r in rows], fontsize=7)
                for y, r, v in zip(ys, rows, sig):
                    s = stuck.get(r["group"])
                    if s is not None:
                        ax.annotate(f"물림 {s:.0%}", (max(v, 0), y + 0.2), xytext=(3, 0), textcoords="offset points",
                                    va="center", fontsize=7, color="#1565c0")
                ax.axvline(0, color="#999", lw=0.8)
                ax.set_title(f"{name[mk]} — 60일 평균 수익률(%)", fontsize=9)
                ax.grid(axis="x", alpha=0.3)
            ax.tick_params(labelsize=7)
        if self.stats_fig.axes:
            self.stats_fig.axes[0].legend(fontsize=7, loc="best")
        self.stats_fig.tight_layout()
        self.stats_canvas.draw()

    def run(self):
        query = self.entry.get()
        if not query.strip():
            return
        try:
            capital = float(self.capital.get().replace(",", ""))
        except ValueError:
            messagebox.showerror("오류", "종목당 한도는 숫자로 입력하세요.")
            return
        self.btn.config(state="disabled")
        self.status.config(text="데이터 조회 중...")
        threading.Thread(target=self._work, args=(query, capital), daemon=True).start()

    def _work(self, query, capital):
        try:
            code, name, market = resolve_ticker(query)
            df = fetch_ohlcv(code, market, days=BACKTEST_DAYS)
            fund = load_fundamentals([(market, code)], False, lambda n, total: None)[(market, code)]
            attach_index(df, code, get_meta(market))
            result = evaluate(df, market, fund)
            fx = fetch_usdkrw() if market == "US" else 1.0
            self.after(0, self._show, code, name, df, result, capital, fx)
        except Exception as e:
            self.after(0, self._error, str(e))

    def _error(self, msg):
        self.btn.config(state="normal")
        self.status.config(text="")
        messagebox.showerror("오류", msg)

    def _show(self, code, name, df, result, capital, fx):
        verdict, checks, events, info = result
        market = info["market"]
        self.btn.config(state="normal")
        self.status.config(text=f"{name} ({code} · {'미국' if market == 'US' else '한국'}) · 기준일 {info['date']}")

        color = "#ef6c00" if "관망" in verdict or "정리 검토" in verdict else \
            "#c62828" if "매수" in verdict else "#1565c0" if "매도" in verdict else \
            "#ef6c00" if info["held"] else "#555"
        self.verdict.config(text=f"{name}: {verdict}", fg=color)
        self.next.config(text=f"다음 단계 ▶ {info['next']}")

        plan = position_plan(info["close"], capital, fx)
        s, parts = plan["shares"], plan["parts"]
        held = (f" · 평단 {fmt_price(info['cost'], market)} ({info['pnl']:+.1%})" if info["held"] else "")
        self.plan.config(text=(
            f"종목당 한도 {capital:,.0f}원 → 1차 {s[0]:,}주({parts[0]:,.0f}원) / 2차 {s[1]:,}주({parts[1]:,.0f}원) / "
            f"3차 {s[2]:,}주({parts[2]:,.0f}원)" + (f"  (환율 {fx:,.1f}원/$)" if market == "US" else "") + held
            + ("  · 손절 규칙 없음" if not PARAMS["stop_loss_pct"] else "")))

        self.tree.delete(*self.tree.get_children())
        for label, status, value in checks:
            self.tree.insert("", "end", text=label, values=(status, value))

        self.log.delete(*self.log.get_children())
        for e in reversed(events):
            buy = e["kind"] == "buy"
            pos = (f"{e['held']}/{info['full']} · 평단 {fmt_price(e['cost'], market)}" if buy
                   else f"{e['held']}/{info['full']} · 수익 {e['pnl']:+.1%}")
            self.log.insert("", "end", text=e["date"].strftime("%Y-%m-%d"), values=(
                f"{e['label']} {'+' if buy else '-'}{e['units']}", fmt_price(e["price"], market), pos, e["why"]))

        self._show_backtest(backtest(df, events, capital, fx))
        self._draw(name, df, events, market)

    def _show_backtest(self, bt):
        won = lambda v: f"{v:+,.0f}원"
        period = f"{bt['start']:%Y-%m-%d} ~ {bt['end']:%Y-%m-%d}"
        if not bt["cycles"]:
            self.perf.config(text=f"과거 손익 ({period}): 이 기간에 매수 신호가 없었습니다.", fg="#555")
        else:
            text = (f"과거 손익 ({period}) · 완료 {bt['n']}회 (이익 {bt['wins']} / 손실 {bt['n'] - bt['wins']}) · "
                    f"실현손익 {won(bt['pnl'])} (투입 합계 대비 {bt['ret']:+.1%}) · 평균 보유 {bt['avg_days']:.0f}일")
            if bt["open"]:
                o = bt["open"]
                text += f" · 보유 중 평가손익 {won(o['pnl'] + o['unreal'])} ({o['ret']:+.1%})"
            text += f"\n비교: {bt['hold_from']:%Y-%m-%d} 첫 매수일에 사서 지금까지 그냥 보유했다면 {bt['hold_ret']:+.1%}"
            text += "  (수수료·세금 미반영, 실적 조건은 과거 시점에 적용하지 않음)"
            total = bt["pnl"] + (bt["open"]["pnl"] + bt["open"]["unreal"] if bt["open"] else 0)
            self.perf.config(text=text, fg="#c62828" if total > 0 else "#1565c0" if total < 0 else "#555")

        self.cycles.delete(*self.cycles.get_children())
        for c in reversed(bt["cycles"]):
            total = c["pnl"] + c.get("unreal", 0.0)
            self.cycles.insert("", "end", text=f"{c['start']:%Y-%m-%d}", tags=("win" if total > 0 else "loss",),
                               values=(f"{c['end']:%Y-%m-%d}" if c["end"] is not None else "보유 중",
                                       c["days"], c["buys"], f"{c['invested']:,.0f}원", won(total), f"{c['ret']:+.1%}"))

    def _draw(self, name, df, events, market, bars=180):
        off = max(0, len(df) - bars)
        df = df.iloc[off:]
        self.fig.clear()
        ax1 = self.fig.add_axes([0.07, 0.42, 0.9, 0.53])
        ax2 = self.fig.add_axes([0.07, 0.25, 0.9, 0.15], sharex=ax1)
        ax3 = self.fig.add_axes([0.07, 0.06, 0.9, 0.17], sharex=ax1)
        x = range(len(df))

        ax1.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter(
            "${x:,.0f}" if market == "US" else "{x:,.0f}"))
        ax1.plot(x, df["종가"], color="#333", lw=1.2, label="종가")
        for e in events:
            if e["i"] < off:
                continue
            xi, buy = e["i"] - off, e["kind"] == "buy"
            color = "#d32f2f" if buy else "#1565c0"
            ax1.scatter(xi, e["price"], marker="^" if buy else "v", s=70, zorder=5, color=color)
            # 매수는 마커 아래, 매도는 마커 위에 비중과 날짜 표시 (물타기는 '물' 표시)
            mark = "물" if "물타기" in e["label"] else ""
            ax1.annotate(f"{'+' if buy else '-'}{e['units']}{mark}\n{e['date']:%m-%d}", (xi, e["price"]),
                         textcoords="offset points", xytext=(0, -12 if buy else 12),
                         ha="center", va="top" if buy else "bottom", fontsize=7, color=color)
            for ax in (ax2, ax3):
                ax.axvline(xi, color=color, ls=":", lw=0.8, alpha=0.6)
        ax1.margins(y=0.15)  # 위·아래 끝의 신호 라벨이 잘리지 않도록
        ax1.set_title(f"{name} 일봉 · 1:2:6 신호 (최근 {len(df)}거래일)")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(alpha=0.3)

        ax2.plot(x, df["RSI"], color="#8e24aa", lw=1)
        ax2.axhline(PARAMS["rsi_oversold"], color="#d32f2f", ls="--", lw=0.8)
        ax2.axhline(PARAMS["rsi_overbought"], color="#1565c0", ls="--", lw=0.8)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("RSI")
        ax2.grid(alpha=0.3)

        hist = df["MACD"] - df["MACD_SIG"]
        ax3.bar(x, hist, color=["#e53935" if h >= 0 else "#1e88e5" for h in hist], width=0.8, alpha=0.5)
        ax3.plot(x, df["MACD"], color="#e53935", lw=1, label="MACD")
        ax3.plot(x, df["MACD_SIG"], color="#1e88e5", lw=1, label="시그널")
        ax3.axhline(0, color="#999", lw=0.8)
        ax3.set_ylabel("MACD")
        ax3.legend(loc="upper left", fontsize=8)
        ax3.grid(alpha=0.3)

        ticks = list(range(0, len(df), max(1, len(df) // 8)))
        ax3.set_xticks(ticks)
        ax3.set_xticklabels([df.index[i].strftime("%y-%m-%d") for i in ticks])
        ax1.tick_params(labelbottom=False)
        ax2.tick_params(labelbottom=False)
        self.canvas.draw()


if __name__ == "__main__":
    App().mainloop()
