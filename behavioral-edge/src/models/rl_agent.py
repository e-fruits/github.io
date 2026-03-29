"""
src/models/rl_agent.py

Phase 2 stub — PPO reinforcement learning agent.

This module is intentionally minimal in Phase 1.  The interface is
defined so that BacktestEngine can accept either a MechanicalStrategy
(Phase 1) or an RLStrategy (Phase 2) without code changes.

Phase 2 implementation plan
────────────────────────────
  State space:
    For each watchlist ticker at each 5-min bar:
      - herd_score, herd_direction (encoded), herd_stage (encoded)
      - gap_pct, relative_volume
      - current unrealised PnL (if in position)
      - time-of-day (normalised 0-1)
      - capital_utilisation (open notional / total capital)

  Action space:
    0 = hold / no trade
    1 = enter long (equity)
    2 = enter short (equity)
    3 = enter long (margin)
    4 = enter short (margin)
    5 = close position (if in one)

  Reward:
    Step reward = realised PnL on close (net of costs)
    Penalty     = -0.001 per bar (holding cost encourages decisiveness)
    Terminal    = 0 (episode ends at EOD)

  Training:
    Use stable-baselines3 PPO with a custom gym.Env wrapping the
    event-driven engine.  Train on 2020-2022, validate on 2023.

To activate: set strategy to RLStrategy in run_backtest.py and ensure
torch + stable-baselines3 are installed (see requirements.txt).
"""

from __future__ import annotations

# Phase 2 imports (commented out until dependencies are installed)
# import torch
# import torch.nn as nn
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env

from src.backtest.strategy import BaseStrategy, Signal, WatchlistEntry
from src.backtest.execution import Direction, Order, PositionType
from src.backtest.risk_manager import RiskManager, RiskState
from src.utils.logging import get_logger

log = get_logger(__name__)


class RLStrategy(BaseStrategy):
    """
    Phase 2 placeholder.  Currently delegates to MechanicalStrategy.
    Replace the body of signal_for_entry() with PPO inference once
    the model is trained.
    """

    def __init__(self, model_path: str | None = None) -> None:
        from src.backtest.strategy import MechanicalStrategy
        self._fallback = MechanicalStrategy()
        self._model_path = model_path
        self._model = None

        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        log.warning("rl_model_loading_not_implemented_using_fallback", path=path)
        # Phase 2:
        # from stable_baselines3 import PPO
        # self._model = PPO.load(path)

    def signal_for_entry(
        self,
        entry: WatchlistEntry,
        current_price: float,
        current_time: str,
    ) -> Signal:
        # Phase 2: build observation vector and call self._model.predict()
        return self._fallback.signal_for_entry(entry, current_price, current_time)

    def size_order(
        self,
        signal: Signal,
        state: RiskState,
        risk_manager: RiskManager,
        current_price: float,
    ) -> Order | None:
        return self._fallback.size_order(signal, state, risk_manager, current_price)
