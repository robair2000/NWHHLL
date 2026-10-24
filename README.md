# NW + HH/LL 1-minute bot (KuCoin futures)

Python version of the TradingView script "NW + HH/LL Pullback Breakout Bot", longs and shorts (mirror rules), 1-minute candles.
Entries wait for a confirmation candle (`CONFIRM=color`). Stop 0.382% / take-profit 0.466% from the entry (`SL_PCT`, `TP_PCT`).
Turn a side off with `ENABLE_LONGS=false` or `ENABLE_SHORTS=false`.

- `nw_bot.py` – the bot (indicators, rules, paper + live trading)
- `fkucoin.py`, `futures_rules.py` – KuCoin futures client and leverage rules (same as the KB bot)
- `render.yaml` – Render background worker with a 1 GB disk at /var/data

Deploy: Render → New → Blueprint → this repo → Apply → paste the KuCoin sub-account keys.
Starts in `MODE=paper`. To trade for real set `MODE=live` and `CONFIRM_LIVE=YES_I_ACCEPT_THE_RISK`.
Kill switch: create a file named `STOP` (Render Shell: `touch STOP`).
