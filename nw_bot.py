"""NW + HH/LL pullback breakout bot - KuCoin USDT futures, 1-minute candles, LONG and SHORT.

Python port of the TradingView script "NW + HH/LL Pullback Breakout Bot".
LONG (ENABLE_LONGS)
  1. Nadaraya-Watson (non-repainting, rational-quadratic kernel) + Higher High / Lower Low pivots
  2. NW middle line must be GREEN (rising)
  3. price must be ABOVE the green line
  4. price pulls back into the line (touches it); no candle may CLOSE below the green line
  5. after the pullback a candle closes INSIDE the red (upper) cloud AND above the current Higher High
  6. enter long; stop below the pullback low; take-profit = RR x stop distance (default 1.5)
SHORT (ENABLE_SHORTS) - the exact mirror
  2. NW middle line must be RED (falling)
  3. price must be BELOW the red line
  4. price pulls back up into the line (touches it); no candle may CLOSE above the red line
  5. after the pullback a candle closes INSIDE the green (lower) cloud AND below the current Lower Low
  6. enter short; stop above the pullback high; take-profit = RR x stop distance

Runs one decision per closed candle, a few seconds after the close. Stop and take-profit are placed as
orders on KuCoin, so they fill between candles. MODE=paper (default) simulates everything.
Kill switch: create a file named STOP next to this script.
"""
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from fkucoin import STEP_SECONDS, FuturesLive, fetch_futures_candles, to_contract
from futures_rules import FUT_FEE, MAX_LEVERAGE, MAX_MARGIN_FRAC, MAX_TOTAL_LEVERAGE, choose_leverage_and_notional


def _f(name, default):
    return float(os.getenv(name, default))


def _i(name, default):
    return int(os.getenv(name, default))


# ---------------------------------------------------------------- settings (environment variables)
PAIRS = [p.strip() for p in os.getenv("PAIRS", "ETH-USDT").split(",") if p.strip()]
ENABLE_LONGS = os.getenv("ENABLE_LONGS", "true").lower() in ("1", "true", "yes")
ENABLE_SHORTS = os.getenv("ENABLE_SHORTS", "true").lower() in ("1", "true", "yes")
MODE = os.getenv("MODE", "paper").lower()
INTERVAL = os.getenv("INTERVAL", "1min")
CANDLES = _i("CANDLES", 600)                   # history fetched per coin each cycle

NW_H = _f("NW_H", 8.0)                         # Lookback Window
NW_ALPHA = _f("NW_ALPHA", 8.0)                 # Relative Weighting
NW_X0 = _i("NW_X0", 25)                        # Start Regression at Bar
NW_ATR_LEN = _i("NW_ATR_LEN", 60)              # Envelope ATR Length
NEAR_F = _f("NEAR_F", 1.5)                     # red cloud starts this many kernel-ATRs above the line
FAR_F = _f("FAR_F", 8.0)                       # ...and ends here
PIVOT_LEFT = _i("PIVOT_LEFT", 5)
PIVOT_RIGHT = _i("PIVOT_RIGHT", 5)

RR = _f("RR", 1.5)                             # take-profit = RR x stop distance (only when TP_PCT=0)
SL_MODE = os.getenv("SL_MODE", "pct").lower()        # pct | pullback | line | atr
SL_PCT = _f("SL_PCT", 0.00382)                 # pct mode: stop this far from the entry (0.382%)
TP_PCT = _f("TP_PCT", 0.00466)                 # take-profit this far from the entry (0.466%); 0 = use RR
SL_BUF = _f("SL_BUF", 0.1)                     # stop buffer in ATR(14) (pullback / line modes)
SL_ATR_MULT = _f("SL_ATR_MULT", 1.5)           # stop = close -/+ this x ATR(14) (atr mode)
CONFIRM = os.getenv("CONFIRM", "color").lower()  # color | break | off  (see signals())
MIN_STOP_PCT = _f("MIN_STOP_PCT", 0.0)         # skip signals whose stop is closer than this (0 = off)

RISK_FRAC = _f("RISK_FRAC", 0.01)              # equity lost if the stop is hit (before the caps below)
MAX_POSITIONS = _i("MAX_POSITIONS", 3)
DAILY_LOSS_LIMIT = _f("DAILY_LOSS_LIMIT", 0.03)
START_CASH = _f("START_CASH", 1000)            # paper mode only
SLIPPAGE = _f("SLIPPAGE", 0.0005)              # paper mode only
MIN_NOTIONAL = 5.0
STATE_PATH = os.getenv("STATE_PATH", "/var/data/nw_state.json")
CLOSE_DELAY = _i("CLOSE_DELAY", 3)             # seconds after the candle close before fetching

STEP = STEP_SECONDS[INTERVAL]


def log(msg):
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


# ---------------------------------------------------------------- indicators (match the Pine script)
def kernel_regression(src, h=NW_H, alpha=NW_ALPHA, x0=NW_X0):
    """Rational-quadratic kernel, endpoint (non-repainting): sum over i = 0..x0+1 of src[t-i] * w_i."""
    src = np.asarray(src, dtype=float)
    n = len(src)
    out = np.full(n, np.nan)
    i = np.arange(x0 + 2)
    w = (1 + (i ** 2) / (h ** 2 * 2 * alpha)) ** (-alpha)
    for t in range(x0 + 1, n):
        window = src[t - i]                    # src[t], src[t-1], ...
        out[t] = (window * w).sum() / w.sum()
    return out


def rma(x, length):
    """Pine ta.rma: seeded with the SMA of the first `length` valid values, then Wilder smoothing."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    valid = np.where(~np.isnan(x))[0]
    if len(valid) < length:
        return out
    f = valid[0]
    if np.isnan(x[f:f + length]).any():
        return out
    out[f + length - 1] = x[f:f + length].mean()
    a = 1.0 / length
    for t in range(f + length, len(x)):
        out[t] = out[t - 1] if np.isnan(x[t]) else a * x[t] + (1 - a) * out[t - 1]
    return out


def atr(df, length=14):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.r_[np.nan, c[:-1]]
    tr = np.where(np.isnan(pc), h - l, np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)]))
    return rma(tr, length)


def pivots(high, low, left=PIVOT_LEFT, right=PIVOT_RIGHT):
    """Pivot high/low known at bar t (confirmed `right` bars after the swing), like ta.pivothigh/low."""
    n = len(high)
    ph, pl = np.full(n, np.nan), np.full(n, np.nan)
    for t in range(left + right, n):
        c = t - right
        hl, hr = high[c - left:c], high[c + 1:t + 1]
        if high[c] > hl.max() and high[c] >= hr.max():
            ph[t] = high[c]
        ll, lr = low[c - left:c], low[c + 1:t + 1]
        if low[c] < ll.min() and low[c] <= lr.min():
            pl[t] = low[c]
    return ph, pl


def signals(df):
    """Replay the strategy over the candles. Returns a DataFrame with indicator columns and
    `signal` (+1 = long, -1 = short, 0 = none on the candle whose close triggers it) plus `stop`."""
    d = df.reset_index(drop=True).copy()
    hi, lo, cl = d["high"].values.astype(float), d["low"].values.astype(float), d["close"].values.astype(float)
    yhat, yh, yl = kernel_regression(cl), kernel_regression(hi), kernel_regression(lo)
    prev = np.r_[np.nan, yhat[:-1]]
    trk = np.where(np.isnan(prev), yh - yl, np.maximum.reduce([yh - yl, np.abs(yh - prev), np.abs(yl - prev)]))
    katr = rma(trk, NW_ATR_LEN)
    up_near, up_far = yhat + NEAR_F * katr, yhat + FAR_F * katr        # red cloud
    dn_near, dn_far = yhat - NEAR_F * katr, yhat - FAR_F * katr        # green cloud
    green, red = yhat > prev, yhat < prev
    a14 = atr(d, 14)
    ph, pl = pivots(hi, lo)

    n = len(d)
    last_ph = last_hh = last_pl = last_ll = np.nan
    sl_state, pb_low = 0, np.nan       # long setup:  0 idle | 1 above green line | 2 pullback done
    ss_state, pb_high = 0, np.nan      # short setup: 0 idle | 1 below red line   | 2 pullback done
    o_sl, o_ss = np.zeros(n, int), np.zeros(n, int)
    o_hh, o_ll = np.full(n, np.nan), np.full(n, np.nan)
    o_sig, o_stop = np.zeros(n, int), np.full(n, np.nan)
    o_pend = np.zeros(n, int)
    pending = None                     # setup signal waiting for its confirmation candle
    for t in range(n):
        if not np.isnan(ph[t]):
            if not np.isnan(last_ph) and ph[t] > last_ph:
                last_hh = ph[t]
            last_ph = ph[t]
        if not np.isnan(pl[t]):
            if not np.isnan(last_pl) and pl[t] < last_pl:
                last_ll = pl[t]
            last_pl = pl[t]
        o_hh[t], o_ll[t] = last_hh, last_ll
        if np.isnan(yhat[t]) or np.isnan(prev[t]) or np.isnan(katr[t]):
            continue
        g, r, y = bool(green[t]), bool(red[t]), yhat[t]
        buf = SL_BUF * a14[t] if not np.isnan(a14[t]) else 0.0

        # ---- confirmation candle: the candle right after the setup signal
        if pending is not None:
            pd_, level, sig_hi, sig_lo, basis = pending
            pending = None
            if pd_ == 1:
                ok = cl[t] > d["open"].values[t] and cl[t] > level and (CONFIRM != "break" or cl[t] > sig_hi)
            else:
                ok = cl[t] < d["open"].values[t] and cl[t] < level and (CONFIRM != "break" or cl[t] < sig_lo)
            if ok:
                if SL_MODE == "pct":
                    stop = cl[t] * (1 - pd_ * SL_PCT)
                elif SL_MODE == "atr":
                    stop = cl[t] - pd_ * SL_ATR_MULT * a14[t]
                else:
                    stop = basis                        # pullback / line stop from the setup candle
                o_sig[t], o_stop[t] = pd_, stop

        # ---- long setup
        if sl_state == 0:
            if g and cl[t] > y and lo[t] > y:
                sl_state = 1
        elif sl_state == 1:
            if not g or cl[t] < y:
                sl_state = 0
            elif lo[t] <= y:                           # pulled back into the line, closed above it
                sl_state, pb_low = 2, lo[t]
        elif sl_state == 2:
            pb_low = min(pb_low, lo[t])
            if cl[t] < y or not g:                     # a close below the green line kills the setup
                sl_state, pb_low = 0, np.nan
        long_sig = (ENABLE_LONGS and sl_state == 2 and g and up_near[t] <= cl[t] <= up_far[t]
                    and not np.isnan(last_hh) and cl[t] > last_hh)

        # ---- short setup (mirror)
        if ss_state == 0:
            if r and cl[t] < y and hi[t] < y:
                ss_state = 1
        elif ss_state == 1:
            if not r or cl[t] > y:
                ss_state = 0
            elif hi[t] >= y:                           # pulled back up into the line, closed below it
                ss_state, pb_high = 2, hi[t]
        elif ss_state == 2:
            pb_high = max(pb_high, hi[t])
            if cl[t] > y or not r:                     # a close above the red line kills the setup
                ss_state, pb_high = 0, np.nan
        short_sig = (ENABLE_SHORTS and ss_state == 2 and r and dn_far[t] <= cl[t] <= dn_near[t]
                     and not np.isnan(last_ll) and cl[t] < last_ll)

        o_sl[t], o_ss[t] = sl_state, ss_state
        if o_sig[t]:
            long_sig = short_sig = False                # just entered on a confirmation candle
        if long_sig:
            stop = (cl[t] * (1 - SL_PCT) if SL_MODE == "pct" else y - buf if SL_MODE == "line"
                    else cl[t] - SL_ATR_MULT * a14[t] if SL_MODE == "atr" else pb_low - buf)
            if CONFIRM == "off":
                o_sig[t], o_stop[t] = 1, stop
            else:
                pending, o_pend[t] = (1, last_hh, hi[t], lo[t], stop), 1
            sl_state, pb_low = 0, np.nan               # the setup resets after a signal
        elif short_sig:
            stop = (cl[t] * (1 + SL_PCT) if SL_MODE == "pct" else y + buf if SL_MODE == "line"
                    else cl[t] + SL_ATR_MULT * a14[t] if SL_MODE == "atr" else pb_high + buf)
            if CONFIRM == "off":
                o_sig[t], o_stop[t] = -1, stop
            else:
                pending, o_pend[t] = (-1, last_ll, hi[t], lo[t], stop), -1
            ss_state, pb_high = 0, np.nan
    d["yhat"], d["green"], d["red"] = yhat, green, red
    d["red_lo"], d["red_hi"], d["grn_hi"], d["grn_lo"] = up_near, up_far, dn_near, dn_far
    d["hh"], d["ll"], d["state"], d["state_s"] = o_hh, o_ll, o_sl, o_ss
    d["signal"], d["stop"], d["pending"] = o_sig, o_stop, o_pend
    return d


# ---------------------------------------------------------------- state
def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"cash": START_CASH, "positions": {}, "trades": [], "day": None, "day_start_equity": None, "seen": {}}


def save_state(state):
    os.makedirs(os.path.dirname(os.path.abspath(STATE_PATH)), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, STATE_PATH)


def levels(d, entry, stop):
    """Stop and take-profit at the actual fill price. pct mode re-anchors the stop on the fill."""
    if SL_MODE == "pct":
        stop = entry * (1 - d * SL_PCT)
    take = entry * (1 + d * TP_PCT) if TP_PCT > 0 else entry + d * RR * abs(entry - stop)
    return stop, take


def side_name(d):
    return "LONG" if d == 1 else "SHORT"


def record(state, sym, pos, exit_price, reason):
    d = pos["dir"]
    pnl = d * (exit_price / pos["entry"] - 1) - 2 * FUT_FEE
    state["trades"].append({"symbol": sym, "dir": d, "entry": pos["entry"], "exit": exit_price, "stop": pos["stop"],
                            "take": pos["take"], "lev": pos["lev"], "pnl_pct": pnl, "reason": reason,
                            "closed": datetime.now(timezone.utc).isoformat()})
    state["trades"] = state["trades"][-500:]
    return pnl


# ---------------------------------------------------------------- brokers
class Paper:
    live = False

    def equity(self, state, prices):
        return state["cash"] + sum(p["dir"] * p["qty"] * (prices.get(s, p["entry"]) - p["entry"])
                                   for s, p in state["positions"].items())

    def open(self, state, sym, d, price, stop, notional, lev, candle_t):
        fill = price * (1 + d * SLIPPAGE)
        stop, take = levels(d, fill, stop)
        if d * (fill - stop) <= 0:
            raise RuntimeError("price already beyond the stop")
        state["cash"] -= notional * FUT_FEE
        state["positions"][sym] = {"dir": d, "qty": notional / fill, "entry": fill, "stop": stop, "take": take,
                                   "lev": lev, "opened": candle_t}
        return state["positions"][sym]

    def check(self, state, sym, row):
        """Stop / take-profit against the newest closed candle (stop first if both were touched)."""
        pos = state["positions"].get(sym)
        if not pos or int(row["t"]) <= pos["opened"]:
            return
        d = pos["dir"]
        o, h, l = float(row["open"]), float(row["high"]), float(row["low"])
        stop_hit = l <= pos["stop"] if d == 1 else h >= pos["stop"]
        take_hit = h >= pos["take"] if d == 1 else l <= pos["take"]
        if stop_hit:
            self.close(state, sym, min(pos["stop"], o) if d == 1 else max(pos["stop"], o), "stop")
        elif take_hit:
            self.close(state, sym, max(pos["take"], o) if d == 1 else min(pos["take"], o), "take-profit")

    def close(self, state, sym, price, reason):
        pos = state["positions"].pop(sym)
        d = pos["dir"]
        fill = price * (1 - d * SLIPPAGE) if reason == "stop" else price
        state["cash"] += d * pos["qty"] * (fill - pos["entry"]) - pos["qty"] * fill * FUT_FEE
        pnl = record(state, sym, pos, fill, reason)
        log(f"{sym}: {side_name(d)} {reason.upper()} @ {fill:.6g} ({pnl:+.2%})")


class Live:
    live = True

    def __init__(self):
        self.api = FuturesLive()

    def equity(self, state, prices):
        return self.api.overview()[0]

    def open(self, state, sym, d, price, stop, notional, lev, candle_t):
        buy, sell = ("buy", "sell") if d == 1 else ("sell", "buy")
        lots = self.api.lots_for(sym, notional, price)
        if lots < 1:
            raise RuntimeError(f"{notional:.2f} USDT is below one contract")
        self.api.prepare(sym, lev)
        self.api.market(sym, buy, lots, lev)
        time.sleep(2)
        ex = self.api.positions().get(sym)
        if not ex:
            raise RuntimeError("order sent but no position visible - check KuCoin manually!")
        entry, open_lots = float(ex["avgEntryPrice"]), abs(float(ex["currentQty"]))
        stop, take = levels(d, entry, stop)
        if d * (entry - stop) <= 0:                               # filled through the stop: get out
            self.api.market(sym, sell, open_lots, lev, reduce_only=True)
            raise RuntimeError(f"filled @ {entry} beyond the stop {stop}, position closed again")
        try:
            stop_id = self.api.exchange_stop(sym, d, open_lots, lev, stop)["orderId"]
        except Exception as e:                                    # never hold a leveraged position unprotected
            self.api.market(sym, sell, open_lots, lev, reduce_only=True)
            raise RuntimeError(f"stop order failed, position closed again: {e}")
        try:
            take_id = self.api.exchange_take(sym, d, open_lots, lev, take)["orderId"]
        except Exception as e:
            take_id = None
            log(f"{sym}: take-profit order failed ({e}) - the bot will close at the take-profit itself")
        mult = self.api.info(sym)["multiplier"]
        state["positions"][sym] = {"dir": d, "qty": open_lots * mult, "lots": open_lots, "entry": entry,
                                   "stop": stop, "take": take, "lev": lev, "opened": candle_t,
                                   "stop_id": stop_id, "take_id": take_id}
        return state["positions"][sym]

    def tidy(self, sym, pos):
        for k in ("stop_id", "take_id"):
            if pos.get(k):
                try:
                    self.api.cancel(pos[k])
                except Exception:
                    pass
        try:
            self.api.cancel_stop_orders(sym)
        except Exception as e:
            log(f"{sym}: could not clear leftover stop orders: {e}")

    def sync(self, state, last_close):
        """Exchange is the truth: positions that are gone were stopped / took profit on KuCoin."""
        ex = self.api.positions()
        for sym in list(state["positions"]):
            if sym in ex:
                continue
            pos = state["positions"].pop(sym)
            self.tidy(sym, pos)
            won = pos["dir"] * (last_close.get(sym, pos["entry"]) - pos["entry"]) > 0
            pnl = record(state, sym, pos, pos["take"] if won else pos["stop"],
                         "take-profit on exchange" if won else "stop on exchange")
            log(f"{sym}: {side_name(pos['dir'])} closed on KuCoin ({'take-profit' if won else 'stop'}, ~{pnl:+.2%})")
        return [s for s in ex if s not in state["positions"]]

    def check(self, state, sym, row):
        """Only needed if the take-profit order could not be placed."""
        pos = state["positions"].get(sym)
        if not pos or pos.get("take_id") or int(row["t"]) <= pos["opened"]:
            return
        d = pos["dir"]
        if (d == 1 and float(row["high"]) >= pos["take"]) or (d == -1 and float(row["low"]) <= pos["take"]):
            self.api.market(sym, "sell" if d == 1 else "buy", pos["lots"], pos["lev"], reduce_only=True)
            state["positions"].pop(sym)
            self.tidy(sym, pos)
            pnl = record(state, sym, pos, pos["take"], "take-profit")
            log(f"{sym}: {side_name(d)} TAKE-PROFIT, closed (~{pnl:+.2%})")


# ---------------------------------------------------------------- one cycle
def fetch(sym):
    try:
        return fetch_futures_candles(sym, INTERVAL, n=CANDLES)
    except RuntimeError as e:
        if "stale" not in str(e):
            raise
        time.sleep(4)                                             # KuCoin can lag a few seconds
        return fetch_futures_candles(sym, INTERVAL, n=CANDLES)


PHASE_L = {0: "", 1: "long: above green line", 2: "long: pullback done, waiting for breakout"}
PHASE_S = {0: "", 1: "short: below red line", 2: "short: pullback done, waiting for breakdown"}


def run_cycle(state, broker, fetcher=fetch):
    data = {}
    for pair in PAIRS:
        sym = to_contract(pair)
        try:
            c = fetcher(sym)
            epoch = pd.Timestamp("1970-01-01")             # unit-safe (pandas may store ns or us)
            c = c.assign(t=((pd.to_datetime(c["timestamps"]) - epoch) // pd.Timedelta(seconds=1)).astype("int64"))
            data[sym] = c
        except Exception as e:
            log(f"{sym}: data error: {e}")
    last_close = {s: float(c["close"].iloc[-1]) for s, c in data.items()}

    untracked = broker.sync(state, last_close) if broker.live else []
    for sym, c in data.items():
        broker.check(state, sym, c.iloc[-1])

    equity = broker.equity(state, last_close)
    today = datetime.now(timezone.utc).date().isoformat()
    if state["day"] != today:
        state["day"], state["day_start_equity"] = today, equity
    halted = equity < state["day_start_equity"] * (1 - DAILY_LOSS_LIMIT)

    for sym, c in data.items():
        t_last = int(c["t"].iloc[-1])
        if state["seen"].get(sym) == t_last:
            continue                                              # this candle was already handled
        state["seen"][sym] = t_last
        row = signals(c).iloc[-1]
        d = int(row["signal"])
        if int(row["pending"]):
            log(f"{sym}: {side_name(int(row['pending']))} setup triggered @ {row['close']:.6g} "
                f"- waiting for the confirmation candle")
        if not d:
            phase = " | ".join(p for p in (PHASE_L[int(row["state"])], PHASE_S[int(row["state_s"])]) if p)
            if "pullback" in phase or sym in state["positions"]:
                log(f"{sym}: close {row['close']:.6g} | line {row['yhat']:.6g} "
                    f"{'green' if row['green'] else 'red' if row['red'] else 'flat'} | HH {row['hh']:.6g} "
                    f"LL {row['ll']:.6g} | {phase or 'waiting'}"
                    + (f" | in {side_name(state['positions'][sym]['dir'])}" if sym in state["positions"] else ""))
            continue
        price, stop = float(row["close"]), float(row["stop"])
        stop_pct = d * (price - stop) / price
        why = None
        if sym in state["positions"]:
            why = "already in a trade"
        elif sym in untracked:
            why = "position on KuCoin that this bot did not open"
        elif halted:
            why = f"daily loss limit ({DAILY_LOSS_LIMIT:.0%}) reached"
        elif len(state["positions"]) >= MAX_POSITIONS:
            why = f"max {MAX_POSITIONS} positions"
        elif stop_pct <= 0:
            why = "stop is on the wrong side of the price"
        elif stop_pct < MIN_STOP_PCT:
            why = f"stop only {stop_pct:.2%} away (MIN_STOP_PCT)"
        if why:
            log(f"{sym}: {side_name(d)} signal @ {price:.6g} skipped: {why}")
            continue
        exposure = sum(p["qty"] * last_close.get(x, p["entry"]) for x, p in state["positions"].items())
        plan = choose_leverage_and_notional(equity, exposure, 1.0, 1.0, stop_pct, RISK_FRAC)
        if not plan or plan["notional"] < MIN_NOTIONAL:
            log(f"{sym}: {side_name(d)} signal skipped: no room under the exposure cap")
            continue
        try:
            pos = broker.open(state, sym, d, price, stop, plan["notional"], plan["leverage"], t_last)
            dist = abs(pos["entry"] - pos["stop"])
            risk = pos["qty"] * dist
            log(f"{sym}: OPENED {side_name(d)} {pos['qty'] * pos['entry']:.2f} USDT @ {pos['entry']:.6g} "
                f"| {plan['leverage']}x | stop {pos['stop']:.6g} ({dist / pos['entry']:.2%} away) "
                f"| take {pos['take']:.6g} | risk {risk:.2f} USDT ({risk / equity:.2%} of equity)")
        except Exception as e:
            log(f"{sym}: open failed: {e}")
        save_state(state)

    open_pos = {s: f"{side_name(p['dir'])} @ {p['entry']:.6g}" for s, p in state["positions"].items()}
    if datetime.now(timezone.utc).minute % 15 == 0 or open_pos:
        log(f"equity {equity:.2f} | open {open_pos or 'none'}" + ("  ** DAILY LOSS LIMIT **" if halted else ""))
    save_state(state)


def main():
    if MODE == "live":
        if os.getenv("CONFIRM_LIVE") != "YES_I_ACCEPT_THE_RISK":
            raise SystemExit("MODE=live needs CONFIRM_LIVE=YES_I_ACCEPT_THE_RISK. Run paper mode first.")
        log("*** LIVE FUTURES MODE: real leveraged orders will be placed (CROSS margin) ***")
        broker = Live()
    else:
        log("PAPER MODE: simulated fills only")
        broker = Paper()
    sides = "+".join(s for s, on in (("longs", ENABLE_LONGS), ("shorts", ENABLE_SHORTS)) if on) or "NO SIDES ENABLED"
    log(f"NW+HH bot | {sides} | {INTERVAL} | pairs {', '.join(PAIRS)} | NW h={NW_H} a={NW_ALPHA} x0={NW_X0} "
        f"| cloud {NEAR_F}-{FAR_F} kATR({NW_ATR_LEN}) | pivots {PIVOT_LEFT}/{PIVOT_RIGHT}")
    stop_txt = f"stop {SL_PCT:.3%}" if SL_MODE == "pct" else f"stop {SL_MODE} (buffer {SL_BUF} ATR)"
    take_txt = f"take {TP_PCT:.3%}" if TP_PCT > 0 else f"take {RR}x stop"
    log(f"confirmation candle: {CONFIRM} | {stop_txt} | {take_txt} | risk {RISK_FRAC:.1%}/trade | max lev {MAX_LEVERAGE}x "
        f"| total exposure <= {MAX_TOTAL_LEVERAGE}x | margin/position <= {MAX_MARGIN_FRAC:.0%} "
        f"| max {MAX_POSITIONS} positions | daily loss limit {DAILY_LOSS_LIMIT:.0%} | state {STATE_PATH}")
    state = load_state()
    state.setdefault("seen", {})
    while True:
        if os.path.exists("STOP"):
            log("STOP file found, exiting (open positions keep their stop / take-profit orders on KuCoin).")
            break
        try:
            run_cycle(state, broker)
        except Exception as e:
            log(f"cycle error: {e}")
        time.sleep(STEP - (time.time() % STEP) + CLOSE_DELAY)


if __name__ == "__main__":
    main()
