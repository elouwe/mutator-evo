# src/mutator_evo/backtest/backtrader_adapter.py
import backtrader as bt
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
import traceback
import math
import random
import logging
import zlib
import pickle
import hashlib
import json
import time

logger = logging.getLogger(__name__)

class BacktraderAdapter:
    """
    - Divides dataset 70/30 (IS/OOS)
    - Cleans NaN/±∞ values
    - Calculates metrics + overfitting penalty
    - Implements LRU cache for backtest results
    - Compresses market data to save memory
    """

    # ---------- INIT ----------
    def __init__(self, full_data_feed: bt.feeds.PandasData):
        # Compress market data for memory efficiency
        self._compressed_data = self._compress_data(full_data_feed.params.dataname)
        self._init_cache()
        self._cache_hits = 0
        self._cache_misses = 0
        
    def _compress_data(self, df: pd.DataFrame) -> bytes:
        """Compress market data using zlib and pickle"""
        return zlib.compress(pickle.dumps(df), 3)
    
    def _decompress_data(self) -> pd.DataFrame:
        """Decompress market data"""
        return pickle.loads(zlib.decompress(self._compressed_data))
    
    def _init_cache(self):
        """Initialize LRU cache for backtest results"""
        self._cache = {}
        self._cache_order = []
        
    def _generate_strategy_hash(self, emb) -> str:
        """Generate unique hash for strategy features"""
        features_str = json.dumps(emb.features, sort_keys=True)
        return hashlib.sha256(features_str.encode()).hexdigest()
    
    # ---------- PUBLIC METHODS ----------
    def evaluate(self, emb) -> Dict[str, float]:
        # Generate unique hash for this strategy
        strategy_hash = self._generate_strategy_hash(emb)
        
        # Check cache first
        if strategy_hash in self._cache:
            self._cache_hits += 1
            return self._cache[strategy_hash]
        
        self._cache_misses += 1
        
        # Perform backtest if not in cache
        start_time = time.time()
        result = self._perform_backtest(emb)
        duration = time.time() - start_time
        
        # Update cache
        self._cache[strategy_hash] = result
        self._cache_order.append(strategy_hash)
        
        # Maintain LRU cache size
        if len(self._cache_order) > 50:
            oldest_hash = self._cache_order.pop(0)
            del self._cache[oldest_hash]
            
        return result
    
    @property
    def original_df(self) -> pd.DataFrame:
        """Returns a copy of the original data"""
        return self._decompress_data().copy()

    # ------------------------------------------------------------
    #                INTERNAL METHODS
    # ------------------------------------------------------------
    def _split_data(self) -> Tuple[bt.DataBase, bt.DataBase]:
        df = self._decompress_data()
        split = int(len(df) * 0.7)
        df_is = df.iloc[:split].copy()
        df_oos = df.iloc[split:].copy()
        
        # Add small noise to prevent zero price movements
        df_is["close"] = df_is["close"].apply(lambda x: max(x, 0.001))
        df_oos["close"] = df_oos["close"].apply(lambda x: max(x, 0.001))
        df_is["volume"] = df_is["volume"].replace(0, 1)
        df_oos["volume"] = df_oos["volume"].replace(0, 1)

        args = dict(
            datetime=None,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            openinterest=None,
        )

        return bt.feeds.PandasData(dataname=df_is, **args), bt.feeds.PandasData(
            dataname=df_oos, **args
        )

    # ---------- SAFE BACKTEST WRAPPER ----------
    def _safe_backtest(self, emb, feed, tag):
        try:
            return self._run_backtest(emb, feed)
        except Exception as e:
            logger.error(f"{tag} backtest failed: {e}")
            logger.error(traceback.format_exc())
            return self._default_metrics()

    # ------------------------------------------------------------
    #               BACKTEST CORE
    # ------------------------------------------------------------
    def _run_backtest(self, emb, feed):
        # ------ SAFE ADX (avoid /0) ------
        class SafeADX(bt.Indicator):
            lines = ("adx",)
            params = (("period", 14),)

            def __init__(self):
                self.addminperiod(self.p.period + 1)

            def next(self):
                if len(self.data) < 2:
                    return

                up = self.data.high[0] - self.data.high[-1]
                dn = self.data.low[-1] - self.data.low[0]

                plus_dm = up if up > dn and up > 0 else 0
                minus_dm = dn if dn > up and dn > 0 else 0

                # True Range
                tr = max(
                    self.data.high[0] - self.data.low[0],
                    abs(self.data.high[0] - self.data.close[-1]),
                    abs(self.data.low[0] - self.data.close[-1]),
                )

                # Avoid division by zero
                if tr == 0:
                    self.lines.adx[0] = 0
                    return

                plus_di = 100 * plus_dm / tr
                minus_di = 100 * minus_dm / tr
                denom = plus_di + minus_di

                dx = 0 if denom == 0 else abs(plus_di - minus_di) / denom * 100

                # Smoothing similar to Wilder's but simplified
                prev = self.lines.adx[-1] if len(self.lines.adx) > 0 else dx
                self.lines.adx[0] = (prev * (self.p.period - 1) + dx) / self.p.period

        # ------ Custom OBV Indicator ------
        class SafeOBV(bt.Indicator):
            lines = ('obv',)
            
            def __init__(self):
                self.addminperiod(2)
                
            def next(self):
                if len(self.data) < 2:
                    self.lines.obv[0] = self.data.volume[0]
                    return
                    
                # Calculate OBV
                if self.data.close[0] > self.data.close[-1]:
                    # Price up: add volume
                    self.lines.obv[0] = self.lines.obv[-1] + self.data.volume[0]
                elif self.data.close[0] < self.data.close[-1]:
                    # Price down: subtract volume
                    self.lines.obv[0] = self.lines.obv[-1] - self.data.volume[0]
                else:
                    # Price unchanged: carry forward
                    self.lines.obv[0] = self.lines.obv[-1]

        # ------ SAFE RSI ------
        class SafeRSI(bt.Indicator):
            lines = ('rsi',)
            params = (('period', 14),)
            
            def __init__(self):
                self.addminperiod(self.p.period + 1)
                up = bt.ind.UpDay(self.data, subplot=False)
                down = bt.ind.DownDay(self.data, subplot=False)
                
                self.avgup = bt.indicators.EMA(up, period=self.p.period)
                self.avgdown = bt.indicators.EMA(down, period=self.p.period)
            
            def next(self):
                if self.avgdown[0] == 0:
                    if self.avgup[0] == 0:
                        self.lines.rsi[0] = 50.0
                    else:
                        self.lines.rsi[0] = 100.0
                else:
                    r = self.avgup[0] / self.avgdown[0]
                    self.lines.rsi[0] = 100.0 - 100.0 / (1.0 + r)

        # ------ SAFE STOCHASTIC ------
        class SafeStochastic(bt.Indicator):
            lines = ('percK', 'percD')
            params = (
                ('period', 14),
                ('period_dfast', 3),
            )

            def __init__(self):
                self.addminperiod(self.p.period + self.p.period_dfast)
                
                # Calculate highest high and lowest low
                hh = bt.ind.Highest(self.data.high, period=self.p.period)
                ll = bt.ind.Lowest(self.data.low, period=self.p.period)
                
                # Calculate %K
                denom = hh - ll
                denom = bt.If(denom == 0, 1, denom)  # Avoid division by zero
                self.percK = 100 * (self.data.close - ll) / denom
                
                # Calculate %D as SMA of %K
                self.lines.percD = bt.ind.SMA(self.percK, period=self.p.period_dfast)
                self.lines.percK = self.percK

        # ------ STRATEGY CLASS ------
        class SafeStrategy(bt.Strategy):
            params = (("features", emb.features),)

            def __init__(self):
                f = self.p.features
                self.trade_size = self._get_f("trade_size", 0.2)
                self.stop_loss = self._get_f("stop_loss", 0.03)
                self.take_profit = self._get_f("take_profit", 0.06)
                self.trade_count = 0
                self.orders = []

                # periods ≥1
                gi = self._get_i
                self.ema_p = gi("ema_period", 15)
                self.rsi_p = gi("rsi_period", 10)
                self.macd_f = gi("macd_fast", 12)
                self.macd_s = gi("macd_slow", 26)
                self.macd_sig = gi("macd_signal", 9)
                self.stoch_k = gi("stoch_k", 14)
                self.stoch_d = gi("stoch_d", 3)
                self.bb_p = gi("bollinger_period", 20)
                self.adx_p = gi("adx_period", 14)

                self.min_period = max(
                    20,
                    self.ema_p,
                    self.rsi_p,
                    self.macd_f + self.macd_s + self.macd_sig,
                    self.stoch_k + self.stoch_d,
                    self.bb_p,
                    self.adx_p,
                )

                self.ind = {}
                # Initialize indicators with proper parameters
                if f.get("use_ema", False):
                    try:
                        self.ind["ema"] = bt.indicators.ExponentialMovingAverage(
                            self.data.close, period=self.ema_p
                        )
                    except Exception as e:
                        print(f"EMA init error: {e}")
                if f.get("use_rsi", False):
                    try:
                        self.ind["rsi"] = SafeRSI(
                            period=self.rsi_p
                        )
                    except Exception as e:
                        print(f"RSI init error: {e}")
                if f.get("use_macd", False):
                    try:
                        self.ind["macd"] = bt.indicators.MACD(
                            self.data.close,
                            period_me1=self.macd_f,
                            period_me2=self.macd_s,
                            period_signal=self.macd_sig
                        )
                    except Exception as e:
                        print(f"MACD init error: {e}")
                if f.get("use_stoch", False):
                    try:
                        self.ind["stoch"] = SafeStochastic(
                            period=self.stoch_k,
                            period_dfast=self.stoch_d
                        )
                    except Exception as e:
                        print(f"Stochastic init error: {e}")
                if f.get("use_bollinger", False):
                    try:
                        self.ind["boll"] = bt.indicators.BollingerBands(
                            self.data.close, period=self.bb_p
                        )
                    except Exception as e:
                        print(f"BollingerBands init error: {e}")
                if f.get("use_adx", False):
                    try:
                        self.ind["adx"] = SafeADX(period=self.adx_p)
                    except Exception as e:
                        print(f"ADX init error: {e}")
                if f.get("use_obv", False):
                    try:
                        self.ind["obv"] = SafeOBV()  # Use custom OBV
                    except Exception as e:
                        print(f"OBV init error: {e}")

                # Initialize RL agent if present
                if f.get("rl_agent"):
                    self.rl_agent = self._init_rl_agent(f["rl_agent"])
                else:
                    self.rl_agent = None

            # ----- Helper methods -----
            def _get_i(self, k, d):
                try:
                    v = int(self.p.features.get(k, d))
                    return v if v >= 1 else d
                except (TypeError, ValueError):
                    return d

            def _get_f(self, k, d):
                try:
                    v = float(self.p.features.get(k, d))
                    return v
                except (TypeError, ValueError):
                    return d

            def _init_rl_agent(self, config):
                # Placeholder for RL agent initialization
                return {
                    "config": config,
                    "model": None  # Simulated model
                }

            def _rl_predict(self, state):
                # Placeholder: returns random action (-1, 0, 1)
                return random.randint(-1, 1)

            def _get_rl_state(self):
                # Create state vector from market data and indicators
                state = []
                # Add price change
                if len(self.data.close) > 1:
                    state.append(self.data.close[0] / self.data.close[-1] - 1)
                else:
                    state.append(0)

                # Add volume change
                if len(self.data.volume) > 1:
                    state.append(self.data.volume[0] / self.data.volume[-1] - 1)
                else:
                    state.append(0)

                # Add indicator values if available
                for name, ind in self.ind.items():
                    if len(ind) > 0:
                        state.append(ind[0])
                    else:
                        state.append(0)

                return state

            # ----- Data validation -----
            def _data_ok(self):
                l = self.data
                return (
                    None
                    not in (l.open[0], l.high[0], l.low[0], l.close[0], l.volume[0])
                    and l.close[0] > 0
                )

            def _ind_ready(self, ind):
                if ind is None or len(ind) == 0:
                    return False
                if hasattr(ind, "lines"):
                    return all(getattr(ind.lines, n)[0] is not None for n in ind.lines._getlines())
                return ind[0] is not None

            # ----- MAIN TRADING LOGIC -----
            def next(self):
                for o in self.orders:
                    self.cancel(o)
                self.orders.clear()

                if len(self.data) < self.min_period or not self._data_ok():
                    return
                if any(not self._ind_ready(i) for i in self.ind.values()):
                    return

                s = []  # Signal strength components

                if "ema" in self.ind:
                    s.append(1 if self.data.close[0] > self.ind["ema"][0] else -1)

                if "rsi" in self.ind:
                    r = self.ind["rsi"][0]
                    if r < 40:
                        s.append(1)
                    elif r > 60:
                        s.append(-1)

                if "macd" in self.ind:
                    m = self.ind["macd"]
                    s.append(1 if m.macd[0] > m.signal[0] else -1)

                if "stoch" in self.ind:
                    st = self.ind["stoch"]
                    if st.percK[0] < 20 and st.percD[0] < 20:
                        s.append(1)
                    elif st.percK[0] > 80 and st.percD[0] > 80:
                        s.append(-1)

                if "boll" in self.ind:
                    bb = self.ind["boll"]
                    if self.data.close[0] < bb.bot[0]:
                        s.append(1)
                    elif self.data.close[0] > bb.top[0]:
                        s.append(-1)

                if "adx" in self.ind and self.ind["adx"][0] > 25:
                    s = [x * 2 for x in s]  # Amplify signals in trending markets

                if "obv" in self.ind and len(self.ind["obv"]) > 1:
                    obv = self.ind["obv"]
                    s.append(1 if obv[0] > obv[-1] else -1)

                # RL agent signal
                if self.rl_agent is not None:
                    state = self._get_rl_state()
                    action = self._rl_predict(state)
                    s.append(action)

                if not s:
                    return

                strength = sum(s)
                cash = self.broker.getvalue()
                price = self.data.close[0]
                size = cash * self.trade_size / price
                if size < cash * 0.01 / price:
                    return

                if strength > 0 and not self.position:
                    self.buy(size=size)
                    self.trade_count += 1
                    self._exit_orders(price, size)
                elif strength < 0 and self.position:
                    self.close()
                    self.trade_count += 1

            def _exit_orders(self, entry, size):
                sl = entry * (1 - self.stop_loss)
                tp = entry * (1 + self.take_profit)
                self.orders.extend(
                    [
                        self.sell(exectype=bt.Order.Stop, price=sl, size=size),
                        self.sell(exectype=bt.Order.Limit, price=tp, size=size),
                    ]
                )

            def notify_trade(self, trade):
                if trade.status == trade.Closed:
                    for o in self.orders:
                        self.cancel(o)
                    self.orders.clear()

        # ------ SETUP CEREBRO ENGINE ------
        cerebro = bt.Cerebro()
        cerebro.adddata(feed)
        cerebro.addstrategy(SafeStrategy)
        cerebro.broker.setcash(100_000)
        cerebro.broker.setcommission(commission=0.001)
        cerebro.broker.set_coc(True)

        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.0, annualize=True)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

        try:
            # Run without vectorization to avoid division errors
            strat = cerebro.run(stdstats=False, runonce=False)[0]  # type: ignore
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            logger.error(traceback.format_exc())
            return self._default_metrics()

        try:
            trade_analysis = strat.analyzers.trades.get_analysis()
            
            # Handle cases where analyzers return None
            total_trades = trade_analysis.total.total if hasattr(trade_analysis, 'total') else 0
            won_trades = trade_analysis.won.total if hasattr(trade_analysis, 'won') else 0
            win_rate = won_trades / total_trades if total_trades > 0 else 0.0
            
            sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio", 0.0) or 0.0
            max_drawdown = strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 50.0)
        except (AttributeError, KeyError, TypeError) as e:
            logger.error(f"Metrics calculation error: {str(e)}")
            return self._default_metrics()

        return {
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "trade_count": strat.trade_count,
        }

    # ---------- HELPER METHODS ----------
    @staticmethod
    def _default_metrics():
        return dict(sharpe=-1.0, max_drawdown=50.0, win_rate=0.3, trade_count=10)

    def _default_results(self):
        d = self._default_metrics()
        return {
            "is_sharpe": 0.0,
            "is_max_drawdown": 100.0,
            "is_win_rate": 0.0,
            "oos_sharpe": -1.0,
            "oos_max_drawdown": 50.0,
            "oos_win_rate": 0.3,
            "overfitting_penalty": 0.8,
            "trade_count": 10,
        }

    @staticmethod
    def _penalty(is_r, oos_r):
        try:
            s = max(0, np.clip(is_r["sharpe"], -10, 10) - np.clip(oos_r["sharpe"], -10, 10))
            dd = max(0, oos_r["max_drawdown"] - is_r["max_drawdown"])
            wr = max(0, is_r["win_rate"] - oos_r["win_rate"])
            return min(0.6 * s + 0.3 * dd + 0.1 * wr, 1.0)
        except Exception as e:
            logger.error(f"Penalty calculation error: {e}")
            return 1.0

    def _perform_backtest(self, emb) -> Dict[str, float]:
        """Actual backtest execution with error handling"""
        try:
            is_data, oos_data = self._split_data()
        except Exception as e:
            logger.error(f"Data split failed: {e}")
            return self._default_results()

        # Run backtests
        is_res = self._safe_backtest(emb, is_data, "IS")
        oos_res = self._safe_backtest(emb, oos_data, "OOS")
        
        pen = self._penalty(is_res, oos_res)

        return {
            "is_sharpe": is_res["sharpe"],
            "is_max_drawdown": is_res["max_drawdown"],
            "is_win_rate": is_res["win_rate"],
            "oos_sharpe": oos_res["sharpe"],
            "oos_max_drawdown": oos_res["max_drawdown"],
            "oos_win_rate": oos_res["win_rate"],
            "overfitting_penalty": pen,
            "trade_count": is_res["trade_count"] + oos_res["trade_count"],
        }