"""
================================================================================
REGIME-ADAPTIVE PORTFOLIO MANAGEMENT SYSTEM v3.0
================================================================================
CRITICAL FIXES IN THIS VERSION:
    ✅ A. Data Leakage Prevention - Strict train/test separation for scaling
    ✅ B. Rolling Window Optimization - Adaptive correlations
    ✅ C. Volatility-Adjusted Targets - Dynamic regime thresholds
    ✅ D. Overfitting Prevention - Regularization, early stopping, purged CV

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  PHASE 1: INITIALIZATION                                                │
    │  ├─► Rolling Window Portfolio Optimization (adaptive correlations)      │
    │  ├─► Train ML with Purged Cross-Validation (no leakage)                │
    │  └─► Volatility-Adjusted Regime Classification                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  PHASE 2: DAILY SIGNAL                                                  │
    │  ├─► Walk-Forward Feature Scaling (fit on past only)                   │
    │  ├─► ML Predicts Regime Probabilities                                   │
    │  ├─► Blend Portfolios (using rolling-optimized weights)                │
    │  └─► Apply Transaction Filters                                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  PHASE 3: RISK MANAGEMENT                                               │
    │  ├─► Monte Carlo with Regime Uncertainty                                │
    │  ├─► Drawdown Monitoring                                                │
    │  └─► Model Health Checks (overfit detection)                           │
    └─────────────────────────────────────────────────────────────────────────┘

Transaction Costs:
    - Trading: 5 bps | Slippage: 2 bps | Management: 10 bps/year
    - Rebalance threshold: 3%

================================================================================
"""

import streamlit as st
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    log_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
from abc import ABC, abstractmethod
import time
import warnings

warnings.filterwarnings('ignore')

# Optional ML libraries
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================

class MarketRegime(Enum):
    """Market regime classifications."""
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"


@dataclass
class SystemConfig:
    """
    Central configuration with all tunable parameters.
    """

    # ──────────────────────────────────────────────────────────────────────
    # DATA SETTINGS
    # ──────────────────────────────────────────────────────────────────────
    TRAIN_START: str = "2006-01-01"
    TEST_START: str = "2018-01-01"

    # ──────────────────────────────────────────────────────────────────────
    # ASSET UNIVERSE
    # ──────────────────────────────────────────────────────────────────────
    CORE_EQUITIES: Tuple[str, ...] = ("SPY", "QQQ")
    INTL_EQUITIES: Tuple[str, ...] = ("VEA",)
    BONDS: Tuple[str, ...] = ("TLT", "IEF")
    ALTERNATIVES: Tuple[str, ...] = ("GLD",)
    VOLATILITY_INDEX: str = "^VIX"

    # ──────────────────────────────────────────────────────────────────────
    # TRANSACTION COSTS
    # ──────────────────────────────────────────────────────────────────────
    TRADING_COST_BPS: float = 5.0
    SLIPPAGE_BPS: float = 2.0
    MANAGEMENT_FEE_ANNUAL_BPS: float = 10.0
    REBALANCE_THRESHOLD: float = 0.03

    # ──────────────────────────────────────────────────────────────────────
    # RISK PARAMETERS
    # ──────────────────────────────────────────────────────────────────────
    RISK_FREE_RATE: float = 0.045
    BULL_TARGET_VOL: float = 0.16
    NEUTRAL_TARGET_VOL: float = 0.12
    BEAR_TARGET_VOL: float = 0.07
    MAX_DRAWDOWN_LIMIT: float = 0.20

    # ──────────────────────────────────────────────────────────────────────
    # ML SETTINGS
    # ──────────────────────────────────────────────────────────────────────
    PREDICTION_HORIZON: int = 21

    # FIX C: Volatility-adjusted threshold multipliers (not fixed %)
    BULL_VOL_MULTIPLIER: float = 0.5  # Bull if return > 0.5 * monthly_vol
    BEAR_VOL_MULTIPLIER: float = -0.5  # Bear if return < -0.5 * monthly_vol

    # Overfitting prevention
    MIN_SAMPLES_PER_CLASS: int = 100
    MAX_FEATURES_RATIO: float = 0.5  # Max features = 50% of sqrt(n_samples)
    CV_PURGE_DAYS: int = 21  # Gap between train/val to prevent leakage

    # ──────────────────────────────────────────────────────────────────────
    # FIX B: ROLLING OPTIMIZATION SETTINGS
    # ──────────────────────────────────────────────────────────────────────
    OPTIMIZATION_LOOKBACK_DAYS: int = 252  # 1 year rolling window
    OPTIMIZATION_REFIT_FREQUENCY: int = 21  # Refit monthly

    # ────────────────────────────────────────────────────��─────────────────
    # MONTE CARLO
    # ──────────────────────────────────────────────────────────────────────
    MC_SIMULATIONS: int = 50000
    MC_HORIZON_YEARS: int = 5

    # ──────────────────────────────────────────────────────────────────────
    # COMPUTED PROPERTIES
    # ─────────────────────────────────��────────────────────────────────────
    @property
    def TOTAL_TRANSACTION_COST(self) -> float:
        return (self.TRADING_COST_BPS + self.SLIPPAGE_BPS) / 10000

    @property
    def DAILY_MANAGEMENT_FEE(self) -> float:
        return (self.MANAGEMENT_FEE_ANNUAL_BPS / 10000) / 252

    @property
    def ALL_TRADEABLE_TICKERS(self) -> List[str]:
        return list(self.CORE_EQUITIES + self.INTL_EQUITIES +
                    self.BONDS + self.ALTERNATIVES)

    @property
    def ALL_DATA_TICKERS(self) -> List[str]:
        return self.ALL_TRADEABLE_TICKERS + [self.VOLATILITY_INDEX]


# ==============================================================================
# SECTION 2: DATA MANAGEMENT
# ==============================================================================

class DataManager:
    """Handles data acquisition and preprocessing."""

    def __init__(self, config: SystemConfig):
        self.config = config

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_market_data(tickers: List[str], start_date: str) -> pd.DataFrame:
        """Load market data with retry logic."""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                data = yf.download(
                    tickers,
                    start=start_date,
                    auto_adjust=True,
                    progress=False,
                    threads=True
                )

                if isinstance(data.columns, pd.MultiIndex):
                    prices = data['Close'].copy()
                else:
                    prices = pd.DataFrame(data['Close'])
                    prices.columns = [tickers[0]] if len(tickers) == 1 else tickers

                if prices.index.tz is not None:
                    prices.index = prices.index.tz_localize(None)

                prices = prices.ffill(limit=5).dropna()

                if len(prices) < 252:
                    raise ValueError(f"Insufficient data: {len(prices)} days")

                return prices

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise RuntimeError(f"Data load failed: {e}")

    @staticmethod
    def compute_returns(prices: pd.DataFrame, log_returns: bool = True) -> pd.DataFrame:
        """Compute returns from prices."""
        if log_returns:
            return np.log(prices / prices.shift(1)).dropna()
        else:
            return prices.pct_change().dropna()


# ==============================================================================
# SECTION 3: FEATURE ENGINEERING (FIX A - NO LEAKAGE)
# ==============================================================================

class FeatureEngine:
    """
    Feature engineering with STRICT leakage prevention.

    FIX A: The scaler is NOT fitted here. Scaling happens in the ML pipeline
           with proper train/test separation.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        # NO SCALER HERE - scaling is done in ML pipeline with proper separation

    def build_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Build features using ONLY past information at each point.
        All features are computed using .rolling() or .shift() to prevent leakage.
        """
        features = pd.DataFrame(index=prices.index)

        spy = prices['SPY']
        vix = prices['^VIX'] if '^VIX' in prices.columns else pd.Series(20, index=prices.index)
        returns = spy.pct_change()

        # ══════════════════════════════════��═══════════════════════════════
        # TREND FEATURES (all use rolling = past data only)
        # ══════════════════════════════════════════════════════════════════
        for window in [20, 50, 100, 200]:
            sma = spy.rolling(window).mean()
            features[f'dist_sma_{window}'] = (spy - sma) / sma

        features['sma_20_50_cross'] = (
                spy.rolling(20).mean() > spy.rolling(50).mean()
        ).astype(int)
        features['sma_50_200_cross'] = (
                spy.rolling(50).mean() > spy.rolling(200).mean()
        ).astype(int)

        # ══════════════════════════════════════════════════════════════════
        # MOMENTUM FEATURES
        # ══════════════════════════════════════════════════════════════════
        for period in [5, 10, 21, 63, 126, 252]:
            features[f'momentum_{period}d'] = spy.pct_change(period)

        features['momentum_accel'] = (
                features['momentum_21d'] - features['momentum_21d'].shift(21)
        )

        # ══════════════════════════════════════════════════════════════════
        # VOLATILITY FEATURES
        # ══════════════════════════════════════════════════════════════════
        for window in [10, 21, 63]:
            features[f'volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(252)

        features['vol_regime'] = features['volatility_21d'] / features['volatility_63d']

        # Expanding percentile (only uses past data)
        features['vol_percentile'] = features['volatility_21d'].rolling(252, min_periods=63).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )

        # ══════════════════════════════════════════════════════════════════
        # VIX FEATURES
        # ══════════════════════════════════════════════════════════════════
        features['vix'] = vix
        features['vix_sma_ratio'] = vix / vix.rolling(21).mean()
        features['vix_percentile'] = vix.rolling(252, min_periods=63).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )
        features['vix_term'] = vix / vix.rolling(63).mean()

        vix_upper = vix.rolling(21).mean() + 2 * vix.rolling(21).std()
        features['vix_spike'] = (vix > vix_upper).astype(int)

        # ══════════════════════════════════════════════════════════════════
        # RSI
        # ══════════════════════════════════════════════════════════════════
        for period in [7, 14, 21]:
            delta = returns
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)

            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()

            rs = avg_gain / avg_loss.replace(0, np.inf)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        features['rsi_divergence'] = features['rsi_14'] - features['rsi_14'].rolling(21).mean()

        # ══════════════════════════════════════════════════════════════════
        # MACD
        # ══════════════════════════════════════════════════════════════════
        ema_12 = spy.ewm(span=12, adjust=False).mean()
        ema_26 = spy.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()

        features['macd'] = macd / spy * 100
        features['macd_signal'] = macd_signal / spy * 100
        features['macd_histogram'] = (macd - macd_signal) / spy * 100

        # ══════════════════════════════════════════════════════════════════
        # BOLLINGER BANDS
        # ══════════════════════════════════════════════════════════════════
        bb_sma = spy.rolling(20).mean()
        bb_std = spy.rolling(20).std()
        bb_upper = bb_sma + 2 * bb_std
        bb_lower = bb_sma - 2 * bb_std

        features['bb_width'] = (bb_upper - bb_lower) / bb_sma
        features['bb_position'] = (spy - bb_lower) / (bb_upper - bb_lower)

        # ══════════════════════════════════════════════════════════════════
        # DRAWDOWN
        # ══════════════════════════════════════════════════════════════════
        rolling_max = spy.expanding().max()  # Expanding = only past data
        features['drawdown'] = (spy - rolling_max) / rolling_max
        features['drawdown_duration'] = self._compute_drawdown_duration(spy, rolling_max)

        # ══════════════════════════════════════════════════════════════════
        # CROSS-ASSET
        # ══════════════════════════════════════════════════════════════════
        if 'TLT' in prices.columns:
            features['spy_tlt_corr'] = (
                prices['SPY'].pct_change().rolling(21)
                .corr(prices['TLT'].pct_change())
            )
            ratio = prices['SPY'] / prices['TLT']
            features['spy_tlt_ratio_zscore'] = (
                    (ratio - ratio.rolling(63).mean()) / ratio.rolling(63).std()
            )

        if 'GLD' in prices.columns:
            ratio = prices['SPY'] / prices['GLD']
            features['spy_gld_ratio_zscore'] = (
                    (ratio - ratio.rolling(63).mean()) / ratio.rolling(63).std()
            )

        if 'QQQ' in prices.columns:
            ratio = prices['QQQ'] / prices['SPY']
            features['qqq_spy_momentum'] = ratio.pct_change(21)

        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)

        return features

    def _compute_drawdown_duration(self, prices: pd.Series, rolling_max: pd.Series) -> pd.Series:
        """Compute days since last all-time high."""
        in_drawdown = prices < rolling_max

        duration = pd.Series(0, index=prices.index, dtype=int)
        count = 0

        for i in range(len(prices)):
            if in_drawdown.iloc[i]:
                count += 1
            else:
                count = 0
            duration.iloc[i] = count

        return duration

    def get_ml_feature_columns(self) -> List[str]:
        """Return list of features for ML model."""
        return [
            'dist_sma_50', 'dist_sma_200', 'sma_50_200_cross',
            'momentum_21d', 'momentum_63d', 'momentum_accel',
            'volatility_21d', 'vol_regime', 'vol_percentile',
            'vix', 'vix_percentile', 'vix_term', 'vix_spike',
            'rsi_14', 'rsi_divergence', 'macd_histogram', 'bb_position',
            'drawdown',
            'spy_tlt_corr', 'spy_tlt_ratio_zscore'
        ]

    def prepare_ml_data(self, features: pd.DataFrame) -> pd.DataFrame:
        """Select ML features (no scaling here - done in ML pipeline)."""
        ml_cols = [c for c in self.get_ml_feature_columns() if c in features.columns]
        return features[ml_cols].copy()


# ==============================================================================
# SECTION 4: VOLATILITY-ADJUSTED TARGET (FIX C)
# ==============================================================================

class TargetGenerator:
    """
    FIX C: Volatility-adjusted regime classification.

    Instead of fixed thresholds (±2%), we use:
        Bull:    forward_return > BULL_MULTIPLIER * trailing_volatility
        Bear:    forward_return < BEAR_MULTIPLIER * trailing_volatility
        Neutral: in between

    This adapts to market conditions:
        - 2017 (low vol): threshold might be ±1%
        - 2020 (high vol): threshold might be ±5%
    """

    def __init__(self, config: SystemConfig):
        self.config = config

    def create_regime_target(self, spy_prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Create volatility-adjusted regime labels.

        Args:
            spy_prices: SPY price series

        Returns:
            Tuple of (target_series, threshold_series for debugging)
        """
        horizon = self.config.PREDICTION_HORIZON

        # Forward returns (what we're trying to predict)
        forward_returns = spy_prices.shift(-horizon) / spy_prices - 1

        # Trailing volatility (monthly, annualized then de-annualized to match horizon)
        # Using 21-day rolling std, scaled to horizon period
        daily_returns = spy_prices.pct_change()
        trailing_vol_daily = daily_returns.rolling(21).std()

        # Scale volatility to prediction horizon (sqrt(T) scaling)
        trailing_vol_horizon = trailing_vol_daily * np.sqrt(horizon)

        # Dynamic thresholds
        bull_threshold = self.config.BULL_VOL_MULTIPLIER * trailing_vol_horizon
        bear_threshold = self.config.BEAR_VOL_MULTIPLIER * trailing_vol_horizon

        # Classify
        target = pd.Series(index=spy_prices.index, dtype=float)

        target[forward_returns >= bull_threshold] = 2  # Bull
        target[forward_returns <= bear_threshold] = 0  # Bear
        target[(forward_returns > bear_threshold) &
               (forward_returns < bull_threshold)] = 1  # Neutral

        # Store thresholds for analysis
        thresholds = pd.DataFrame({
            'bull_threshold': bull_threshold,
            'bear_threshold': bear_threshold,
            'trailing_vol': trailing_vol_horizon,
            'forward_return': forward_returns
        })

        return target, thresholds

    def analyze_target_distribution(self, target: pd.Series,
                                    split_date: str) -> Dict[str, Any]:
        """Analyze target distribution for train/test periods."""

        train_target = target[target.index < split_date].dropna()
        test_target = target[target.index >= split_date].dropna()

        def get_dist(t):
            counts = t.value_counts().sort_index()
            return {
                'bear': counts.get(0, 0),
                'neutral': counts.get(1, 0),
                'bull': counts.get(2, 0),
                'total': len(t),
                'bear_pct': counts.get(0, 0) / len(t) if len(t) > 0 else 0,
                'neutral_pct': counts.get(1, 0) / len(t) if len(t) > 0 else 0,
                'bull_pct': counts.get(2, 0) / len(t) if len(t) > 0 else 0,
            }

        return {
            'train': get_dist(train_target),
            'test': get_dist(test_target),
            'total': get_dist(target.dropna())
        }


# ==============================================================================
# SECTION 5: ML REGIME CLASSIFIER (FIX A - PROPER SCALING)
# ==============================================================================

class PurgedTimeSeriesSplit:
    """
    Time series cross-validation with purging to prevent leakage.

    FIX A: Adds a gap (purge period) between training and validation sets
    to ensure no information leakage from the target's forward-looking nature.

    Example with purge_days=21:
        Fold 1: Train [0:1000], Gap [1000:1021], Val [1021:1200]
        Fold 2: Train [0:1200], Gap [1200:1221], Val [1221:1400]
    """

    def __init__(self, n_splits: int = 5, purge_days: int = 21):
        self.n_splits = n_splits
        self.purge_days = purge_days

    def split(self, X: np.ndarray):
        """Generate purged train/validation indices."""
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            # Training ends here
            train_end = test_size * (i + 1)

            # Validation starts after purge period
            val_start = train_end + self.purge_days
            val_end = min(val_start + test_size, n_samples)

            if val_start >= n_samples:
                continue

            train_indices = np.arange(0, train_end)
            val_indices = np.arange(val_start, val_end)

            yield train_indices, val_indices


class RegimeClassifier:
    """
    Ensemble ML classifier with STRICT leakage prevention.

    FIX A Implementation:
        1. Scaler is fitted ONLY on training data
        2. Purged cross-validation prevents target leakage
        3. Walk-forward validation for realistic performance estimates

    Overfitting Prevention:
        1. Strong regularization (high alpha, low max_depth)
        2. Early stopping
        3. Feature selection
        4. Ensemble averaging
        5. Purged CV with gap
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.model = None
        self.scaler = None  # Will be fitted on training data only
        self._is_fitted = False
        self._feature_importance = None
        self._selected_features = None
        self._training_metrics = {}

    def _build_base_models(self) -> List[Tuple[str, Any]]:
        """
        Build regularized base models to prevent overfitting.

        Overfitting Prevention:
            - Low max_depth (3-4)
            - High min_samples_leaf (30-50)
            - Strong L2 regularization
            - Early stopping
            - Subsample < 1.0
        """
        models = [
            ('gb', GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,  # Shallow trees
                learning_rate=0.05,
                subsample=0.7,  # Bagging for variance reduction
                min_samples_leaf=50,  # Prevent overfitting to noise
                min_samples_split=100,
                max_features='sqrt',  # Feature bagging
                validation_fraction=0.15,  # Early stopping
                n_iter_no_change=10,
                random_state=42
            )),

            ('rf', RandomForestClassifier(
                n_estimators=150,
                max_depth=4,  # Shallow
                min_samples_leaf=40,
                min_samples_split=80,
                max_features='sqrt',
                class_weight='balanced',
                bootstrap=True,
                oob_score=True,  # Out-of-bag for validation
                random_state=42,
                n_jobs=-1
            )),

            ('mlp', MLPClassifier(
                hidden_layer_sizes=(32, 16),  # Small network
                activation='relu',
                alpha=0.1,  # Strong L2 regularization
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=15,
                random_state=42
            ))
        ]

        # XGBoost with regularization
        if XGBOOST_AVAILABLE:
            models.append(('xgb', XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.7,
                min_child_weight=50,
                reg_alpha=0.5,  # L1 regularization
                reg_lambda=2.0,  # L2 regularization
                gamma=0.1,  # Min loss reduction for split
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss',
                verbosity=0
            )))

        # LightGBM with regularization
        if LIGHTGBM_AVAILABLE:
            models.append(('lgbm', LGBMClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.7,
                min_child_samples=50,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=42,
                verbose=-1
            )))

        return models

    def _select_features(self, X: np.ndarray, y: np.ndarray,
                         feature_names: List[str]) -> List[int]:
        """
        Feature selection to prevent overfitting.

        Uses a simple model to identify the most important features,
        then limits the feature set to prevent curse of dimensionality.
        """
        # Use gradient boosting for feature importance
        selector_model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42
        )
        selector_model.fit(X, y)

        # Get importances
        importances = selector_model.feature_importances_

        # Select top features (max 50% of sqrt(n_samples) or all if fewer)
        n_samples = len(X)
        max_features = int(self.config.MAX_FEATURES_RATIO * np.sqrt(n_samples))
        max_features = max(5, min(max_features, len(feature_names)))

        # Get indices of top features
        top_indices = np.argsort(importances)[-max_features:]

        self._selected_features = [feature_names[i] for i in top_indices]

        return sorted(top_indices)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'RegimeClassifier':
        """
        Fit the classifier with proper train/test separation.

        FIX A: Scaler is fitted ONLY on X_train.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features (for monitoring)
            y_val: Optional validation labels
        """
        # ────────────────────────────────────────────────────────────────
        # STEP 1: Handle missing values in training data
        # ────────────────────────────────────────────────────────────────
        valid_mask = X_train.notna().all(axis=1) & y_train.notna()
        X_clean = X_train[valid_mask].copy()
        y_clean = y_train[valid_mask].astype(int)

        feature_names = X_clean.columns.tolist()

        # ────────────────────────────────────────────────────────────────
        # STEP 2: FIT SCALER ON TRAINING DATA ONLY
        # ────────────────────────────────────────────────────────────────
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_clean)

        # ────────────────────────────────────────────────────────────────
        # STEP 3: Feature selection (on training data only)
        # ────────────────────────────────────────────────────────────────
        selected_indices = self._select_features(X_scaled, y_clean, feature_names)
        X_selected = X_scaled[:, selected_indices]

        print(f"Selected {len(selected_indices)} features out of {len(feature_names)}")
        print(f"Selected features: {self._selected_features}")

        # ────────────────────────────────────────────────────────────────
        # STEP 4: Build and fit ensemble
        # ───────────────────────��────────────────────────────────────────
        base_models = self._build_base_models()

        # Calibrate for better probabilities
        calibrated_models = [
            (name, CalibratedClassifierCV(model, method='isotonic', cv=3))
            for name, model in base_models
        ]

        # Meta-learner with regularization
        meta_learner = LogisticRegression(
            C=0.5,  # Regularization
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )

        # Stacking with purged CV
        self.model = StackingClassifier(
            estimators=calibrated_models,
            final_estimator=meta_learner,
            cv=PurgedTimeSeriesSplit(n_splits=5, purge_days=self.config.CV_PURGE_DAYS),
            stack_method='predict_proba',
            passthrough=False,
            n_jobs=-1
        )

        self.model.fit(X_selected, y_clean)

        # ────────────────────────────────────────────────────────────────
        # STEP 5: Compute training metrics
        # ────────────────────────────────────────────────────────────────
        y_train_pred = self.model.predict(X_selected)
        y_train_proba = self.model.predict_proba(X_selected)

        self._training_metrics['train_accuracy'] = accuracy_score(y_clean, y_train_pred)
        self._training_metrics['train_log_loss'] = log_loss(y_clean, y_train_proba)

        # ────────────────────────────────────────────────────────────────
        # STEP 6: Validation metrics (if provided)
        # ────────────────────────────────────────────────────────────────
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self._training_metrics['val_accuracy'] = val_metrics['accuracy']
            self._training_metrics['val_log_loss'] = val_metrics.get('log_loss', None)

            # Check for overfitting
            overfit_gap = (self._training_metrics['train_accuracy'] -
                           self._training_metrics['val_accuracy'])
            self._training_metrics['overfit_gap'] = overfit_gap

            if overfit_gap > 0.10:
                print(f"⚠️ WARNING: Potential overfitting detected! "
                      f"Train acc: {self._training_metrics['train_accuracy']:.3f}, "
                      f"Val acc: {self._training_metrics['val_accuracy']:.3f}, "
                      f"Gap: {overfit_gap:.3f}")

        # Store feature importance
        self._compute_feature_importance(feature_names, selected_indices)

        self._is_fitted = True
        self._feature_indices = selected_indices

        return self

    def _compute_feature_importance(self, all_features: List[str],
                                    selected_indices: List[int]):
        """Extract feature importance from base models."""
        importances = np.zeros(len(selected_indices))
        count = 0

        for name, calibrated_model in self.model.estimators_:
            # Get base model from calibrated wrapper
            if hasattr(calibrated_model, 'estimator'):
                base = calibrated_model.estimator
            else:
                base = calibrated_model

            if hasattr(base, 'feature_importances_'):
                importances += base.feature_importances_
                count += 1

        if count > 0:
            importances /= count
            selected_names = [all_features[i] for i in selected_indices]
            self._feature_importance = pd.Series(
                importances, index=selected_names
            ).sort_values(ascending=False)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict regime probabilities.

        FIX A: Uses scaler fitted on training data only.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # Handle missing values
        X_filled = X.fillna(X.mean())

        # Transform using training-fitted scaler
        X_scaled = self.scaler.transform(X_filled)

        # Select same features as training
        X_selected = X_scaled[:, self._feature_indices]

        return self.model.predict_proba(X_selected)

    def predict_regime(self, X: pd.DataFrame) -> Tuple[MarketRegime, Dict[MarketRegime, float]]:
        """Predict regime with probabilities."""
        proba = self.predict_proba(X)[0]

        prob_dict = {
            MarketRegime.BEAR: proba[0],
            MarketRegime.NEUTRAL: proba[1],
            MarketRegime.BULL: proba[2]
        }

        predicted = max(prob_dict, key=prob_dict.get)
        return predicted, prob_dict

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate model on test data."""
        valid_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_mask]
        y_clean = y[valid_mask].astype(int)

        X_filled = X_clean.fillna(X_clean.mean())
        X_scaled = self.scaler.transform(X_filled)
        X_selected = X_scaled[:, self._feature_indices]

        y_pred = self.model.predict(X_selected)
        y_proba = self.model.predict_proba(X_selected)

        metrics = {
            'accuracy': accuracy_score(y_clean, y_pred),
            'precision_macro': precision_score(y_clean, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_clean, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_clean, y_pred, average='macro', zero_division=0),
            'log_loss': log_loss(y_clean, y_proba),
            'confusion_matrix': confusion_matrix(y_clean, y_pred),
            'classification_report': classification_report(y_clean, y_pred, output_dict=True)
        }

        try:
            metrics['roc_auc_ovr'] = roc_auc_score(y_clean, y_proba, multi_class='ovr')
        except:
            metrics['roc_auc_ovr'] = None

        return metrics

    def cross_validate_purged(self, X: pd.DataFrame, y: pd.Series,
                              n_splits: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Purged time-series cross-validation.

        FIX A: Each fold fits scaler only on training portion.
        """
        valid_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_mask]
        y_clean = y[valid_mask].astype(int)

        cv = PurgedTimeSeriesSplit(n_splits=n_splits,
                                   purge_days=self.config.CV_PURGE_DAYS)

        scores = {
            'accuracy': [],
            'f1_macro': [],
            'log_loss': []
        }

        # Simple model for CV (faster)
        cv_model = GradientBoostingClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.05,
            min_samples_leaf=50,
            random_state=42
        )

        for train_idx, val_idx in cv.split(X_clean):
            X_tr = X_clean.iloc[train_idx]
            X_val = X_clean.iloc[val_idx]
            y_tr = y_clean.iloc[train_idx]
            y_val = y_clean.iloc[val_idx]

            # FIT SCALER ON TRAINING FOLD ONLY
            fold_scaler = RobustScaler()
            X_tr_scaled = fold_scaler.fit_transform(X_tr)
            X_val_scaled = fold_scaler.transform(X_val)

            cv_model.fit(X_tr_scaled, y_tr)

            y_pred = cv_model.predict(X_val_scaled)
            y_proba = cv_model.predict_proba(X_val_scaled)

            scores['accuracy'].append(accuracy_score(y_val, y_pred))
            scores['f1_macro'].append(f1_score(y_val, y_pred, average='macro', zero_division=0))
            scores['log_loss'].append(log_loss(y_val, y_proba))

        return {
            metric: {
                'mean': np.mean(vals),
                'std': np.std(vals),
                'values': vals
            }
            for metric, vals in scores.items()
        }

    def get_training_metrics(self) -> Dict:
        """Return training metrics for overfitting analysis."""
        return self._training_metrics

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Return feature importance."""
        return self._feature_importance


# ==============================================================================
# SECTION 6: ROLLING PORTFOLIO OPTIMIZER (FIX B)
# ==============================================================================

class RollingPortfolioOptimizer:
    """
    FIX B: Rolling window portfolio optimization.

    Instead of optimizing once on all historical data, this optimizer:
        1. Uses a rolling window (e.g., trailing 252 days)
        2. Re-optimizes periodically (e.g., every 21 days)
        3. Adapts to changing correlations

    This handles regime shifts like 2022 where stock-bond correlation flipped.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.regime_portfolios: Dict[MarketRegime, Dict[str, float]] = {}
        self.regime_metrics: Dict[MarketRegime, Dict[str, float]] = {}
        self._optimization_history: List[Dict] = []

    def _get_regime_constraints(self, regime: MarketRegime) -> Dict:
        """Get optimization constraints for a regime."""
        constraints = {
            MarketRegime.BULL: {
                'target_vol': self.config.BULL_TARGET_VOL,
                'equity_min': 0.55, 'equity_max': 0.85,
                'bond_min': 0.05, 'bond_max': 0.25,
                'alt_min': 0.05, 'alt_max': 0.25
            },
            MarketRegime.NEUTRAL: {
                'target_vol': self.config.NEUTRAL_TARGET_VOL,
                'equity_min': 0.35, 'equity_max': 0.60,
                'bond_min': 0.20, 'bond_max': 0.40,
                'alt_min': 0.10, 'alt_max': 0.30
            },
            MarketRegime.BEAR: {
                'target_vol': self.config.BEAR_TARGET_VOL,
                'equity_min': 0.10, 'equity_max': 0.35,
                'bond_min': 0.30, 'bond_max': 0.55,
                'alt_min': 0.20, 'alt_max': 0.40
            }
        }
        return constraints[regime]

    def optimize_for_date(self, returns: pd.DataFrame,
                          as_of_date: pd.Timestamp,
                          regime: MarketRegime) -> Dict[str, float]:
        """
        Optimize portfolio using data available as of a specific date.

        FIX B: Uses only trailing lookback window, not future data.

        Args:
            returns: Full returns DataFrame
            as_of_date: Date to optimize as of (only uses data before this)
            regime: Target regime

        Returns:
            Optimized weights
        """
        lookback = self.config.OPTIMIZATION_LOOKBACK_DAYS

        # Get returns up to as_of_date
        available_returns = returns[returns.index < as_of_date]

        # Use only trailing window
        if len(available_returns) > lookback:
            window_returns = available_returns.iloc[-lookback:]
        else:
            window_returns = available_returns

        if len(window_returns) < 63:  # Minimum 3 months
            # Not enough data, return equal weight
            tickers = returns.columns.tolist()
            return {t: 1 / len(tickers) for t in tickers}

        return self._optimize_single(window_returns, regime)

    def _optimize_single(self, returns: pd.DataFrame,
                         regime: MarketRegime) -> Dict[str, float]:
        """Run single optimization."""
        tickers = returns.columns.tolist()
        n_assets = len(tickers)

        # Statistics from window
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        # Regularize covariance matrix (Ledoit-Wolf shrinkage approximation)
        # This helps when estimation is noisy
        shrinkage = 0.1
        cov_matrix = (1 - shrinkage) * cov_matrix + shrinkage * np.diag(np.diag(cov_matrix))

        rc = self._get_regime_constraints(regime)

        # Identify asset classes
        equity_tickers = list(self.config.CORE_EQUITIES + self.config.INTL_EQUITIES)
        bond_tickers = list(self.config.BONDS)
        alt_tickers = list(self.config.ALTERNATIVES)

        equity_idx = [i for i, t in enumerate(tickers) if t in equity_tickers]
        bond_idx = [i for i, t in enumerate(tickers) if t in bond_tickers]
        alt_idx = [i for i, t in enumerate(tickers) if t in alt_tickers]

        # Objective: Maximize Sharpe
        def neg_sharpe(weights):
            port_ret = np.sum(mean_returns.values * weights)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
            if port_vol < 1e-6:
                return 1e10
            return -(port_ret - self.config.RISK_FREE_RATE) / port_vol

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
            {'type': 'ineq', 'fun': lambda x: rc['target_vol'] -
                                              np.sqrt(np.dot(x.T, np.dot(cov_matrix.values, x)))}
        ]

        if equity_idx:
            constraints.extend([
                {'type': 'ineq', 'fun': lambda x, idx=equity_idx: np.sum(x[idx]) - rc['equity_min']},
                {'type': 'ineq', 'fun': lambda x, idx=equity_idx: rc['equity_max'] - np.sum(x[idx])}
            ])

        if bond_idx:
            constraints.extend([
                {'type': 'ineq', 'fun': lambda x, idx=bond_idx: np.sum(x[idx]) - rc['bond_min']},
                {'type': 'ineq', 'fun': lambda x, idx=bond_idx: rc['bond_max'] - np.sum(x[idx])}
            ])

        if alt_idx:
            constraints.extend([
                {'type': 'ineq', 'fun': lambda x, idx=alt_idx: np.sum(x[idx]) - rc['alt_min']},
                {'type': 'ineq', 'fun': lambda x, idx=alt_idx: rc['alt_max'] - np.sum(x[idx])}
            ])

        bounds = tuple((0.02, 0.50) for _ in range(n_assets))
        init_weights = np.array([1 / n_assets] * n_assets)

        result = minimize(
            neg_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )

        weights = result.x if result.success else init_weights
        weights = weights / weights.sum()

        return dict(zip(tickers, weights))

    def optimize_all_regimes(self, returns: pd.DataFrame,
                             as_of_date: Optional[pd.Timestamp] = None) -> Dict[MarketRegime, Dict[str, float]]:
        """
        Optimize portfolios for all regimes.

        Args:
            returns: Returns DataFrame
            as_of_date: If provided, only use data before this date
        """
        if as_of_date is None:
            as_of_date = returns.index[-1] + pd.Timedelta(days=1)

        for regime in MarketRegime:
            weights = self.optimize_for_date(returns, as_of_date, regime)
            self.regime_portfolios[regime] = weights

            # Compute metrics
            lookback = self.config.OPTIMIZATION_LOOKBACK_DAYS
            window_returns = returns[returns.index < as_of_date].iloc[-lookback:]

            weight_arr = np.array([weights.get(t, 0) for t in returns.columns])
            port_return = np.sum(window_returns.mean().values * weight_arr) * 252
            port_vol = np.sqrt(np.dot(weight_arr.T,
                                      np.dot(window_returns.cov().values * 252, weight_arr)))

            self.regime_metrics[regime] = {
                'expected_return': port_return,
                'volatility': port_vol,
                'sharpe_ratio': (port_return - self.config.RISK_FREE_RATE) / port_vol if port_vol > 0 else 0
            }

        return self.regime_portfolios

    def get_blended_weights(self, regime_probs: Dict[MarketRegime, float]) -> Dict[str, float]:
        """Blend portfolios based on regime probabilities."""
        if not self.regime_portfolios:
            raise RuntimeError("Must optimize regime portfolios first")

        all_assets = set()
        for weights in self.regime_portfolios.values():
            all_assets.update(weights.keys())

        blended = {}
        for asset in all_assets:
            blended[asset] = sum(
                regime_probs.get(regime, 0) * self.regime_portfolios[regime].get(asset, 0)
                for regime in MarketRegime
            )

        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}

        return blended

    def print_regime_portfolios(self, as_of_date: Optional[str] = None):
        """Print formatted portfolio summary."""
        print("\n" + "=" * 70)
        title = f"REGIME PORTFOLIOS"
        if as_of_date:
            title += f" (as of {as_of_date})"
        print(f"{title:^70}")
        print("=" * 70)

        for regime in MarketRegime:
            if regime not in self.regime_portfolios:
                continue

            weights = self.regime_portfolios[regime]
            metrics = self.regime_metrics[regime]

            print(f"\n{'─' * 70}")
            print(f"  {regime.value.upper()} PORTFOLIO")
            print(f"  E[R]: {metrics['expected_return']:.1%} | "
                  f"Vol: {metrics['volatility']:.1%} | "
                  f"Sharpe: {metrics['sharpe_ratio']:.2f}")
            print(f"{'─' * 70}")

            for ticker, weight in sorted(weights.items(), key=lambda x: -x[1]):
                if weight > 0.001:
                    print(f"  {ticker:<10} {weight:>8.1%}")

        print("=" * 70)


# ==============================================================================
# SECTION 7: MONTE CARLO ENGINE
# ==============================================================================

class MonteCarloEngine:
    """Monte Carlo simulation for risk analysis."""

    def __init__(self, config: SystemConfig):
        self.config = config

    def simulate_portfolio(self, returns: pd.DataFrame,
                           weights: Dict[str, float],
                           n_sims: Optional[int] = None,
                           horizon_years: Optional[int] = None,
                           initial_value: float = 100000) -> Dict[str, Any]:
        """Run Monte Carlo simulation."""
        n_sims = n_sims or self.config.MC_SIMULATIONS
        horizon_years = horizon_years or self.config.MC_HORIZON_YEARS
        n_days = 252 * horizon_years

        weight_arr = np.array([weights.get(t, 0) for t in returns.columns])
        port_returns = returns.values @ weight_arr

        port_mean = np.mean(port_returns)
        port_std = np.std(port_returns)

        drift = port_mean - 0.5 * port_std ** 2

        np.random.seed(42)
        Z = np.random.standard_normal((n_days, n_sims))
        daily_returns = np.exp(drift + port_std * Z)

        price_paths = np.zeros((n_days + 1, n_sims))
        price_paths[0] = initial_value

        for t in range(1, n_days + 1):
            price_paths[t] = price_paths[t - 1] * daily_returns[t - 1]

        ending_values = price_paths[-1]
        cagrs = (ending_values / initial_value) ** (1 / horizon_years) - 1

        # Drawdowns
        max_drawdowns = []
        for i in range(n_sims):
            path = price_paths[:, i]
            rolling_max = np.maximum.accumulate(path)
            drawdown = (path - rolling_max) / rolling_max
            max_drawdowns.append(np.min(drawdown))
        max_drawdowns = np.array(max_drawdowns)

        return {
            'price_paths': price_paths,
            'ending_values': ending_values,
            'cagrs': cagrs,
            'max_drawdowns': max_drawdowns,
            'mean_ending_value': np.mean(ending_values),
            'median_ending_value': np.median(ending_values),
            'mean_cagr': np.mean(cagrs),
            'median_cagr': np.median(cagrs),
            'std_cagr': np.std(cagrs),
            'ci_95_lower': np.percentile(ending_values, 2.5),
            'ci_95_upper': np.percentile(ending_values, 97.5),
            'ci_95_lower_cagr': np.percentile(cagrs, 2.5),
            'ci_95_upper_cagr': np.percentile(cagrs, 97.5),
            'prob_loss': np.mean(ending_values < initial_value),
            'prob_double': np.mean(ending_values >= 2 * initial_value),
            'var_95': np.percentile(ending_values, 5),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': np.min(max_drawdowns),
            'annual_volatility': port_std * np.sqrt(252),
            'sharpe_ratio': (np.mean(cagrs) - self.config.RISK_FREE_RATE) / np.std(cagrs) if np.std(cagrs) > 0 else 0
        }

    def plot_simulation(self, results: Dict, title: str = "Monte Carlo",
                        n_paths: int = 200) -> Tuple[plt.Figure, plt.Figure]:
        """Visualize simulation results."""
        # Figure 1: Paths
        fig1, ax1 = plt.subplots(figsize=(12, 7))

        price_paths = results['price_paths']
        ending_values = results['ending_values']
        n_sims = price_paths.shape[1]

        cmap = plt.get_cmap('RdYlGn')
        norm = plt.Normalize(
            vmin=np.percentile(ending_values, 5),
            vmax=np.percentile(ending_values, 95)
        )

        indices = np.random.choice(n_sims, min(n_paths, n_sims), replace=False)

        for i in indices:
            ax1.plot(price_paths[:, i], color=cmap(norm(ending_values[i])),
                     alpha=0.3, linewidth=0.5)

        ax1.plot(np.mean(price_paths, axis=1), color='blue', linewidth=2.5,
                 label='Mean Path', linestyle='--')
        ax1.fill_between(
            range(len(price_paths)),
            np.percentile(price_paths, 10, axis=1),
            np.percentile(price_paths, 90, axis=1),
            alpha=0.2, color='blue', label='80% CI'
        )

        ax1.axhline(y=100000, color='black', linestyle=':', alpha=0.5)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax1, label='Ending Value ($)')

        ax1.set_title(f'{title}: {self.config.MC_HORIZON_YEARS}-Year Projection',
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('Trading Days')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Figure 2: Distribution
        fig2, ax2 = plt.subplots(figsize=(12, 7))

        cagrs = results['cagrs']
        ax2.hist(cagrs, bins=100, density=True, color='#2E86AB',
                 edgecolor='white', alpha=0.8)

        ax2.axvline(results['mean_cagr'], color='red', linewidth=2.5,
                    label=f"Mean: {results['mean_cagr']:.1%}")
        ax2.axvline(results['ci_95_lower_cagr'], color='orange', linewidth=2,
                    linestyle='--', label=f"5th %ile: {results['ci_95_lower_cagr']:.1%}")
        ax2.axvline(results['ci_95_upper_cagr'], color='green', linewidth=2,
                    linestyle='--', label=f"95")