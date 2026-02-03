"""
================================================================================
REGIME-ADAPTIVE PORTFOLIO MANAGEMENT SYSTEM v4.0 (COMPLETE)
================================================================================
FIXES IMPLEMENTED:
    ✅ A. Data Leakage Prevention - Strict train/test separation for scaling
    ✅ B. Rolling Window Optimization - Adaptive correlations
    ✅ C. Volatility-Adjusted Targets - Dynamic regime thresholds
    ✅ D. Correlation-Aware Bond Constraints - Prevents "Bond Trap"

================================================================================
"""
import numpy as np
import random
# LOCK RANDOMNESS GLOBALLY
np.random.seed(42)
random.seed(42)

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
from sklearn.linear_model import LogisticRegression
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
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
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
# SECTION 1: ENUMS AND CONFIGURATION
# ==============================================================================

class MarketRegime(Enum):
    """Market regime classifications."""
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"


class CorrelationRegime(Enum):
    """Stock-Bond correlation regimes."""
    NEGATIVE = "negative"  # Traditional: Bonds hedge stocks
    NEUTRAL = "neutral"  # No clear relationship
    POSITIVE = "positive"  # DANGER: Bonds fall with stocks


@dataclass
class SystemConfig:
    """Central configuration for the entire system."""

    # Data Settings
    TRAIN_START: str = "2006-01-01"
    TEST_START: str = "2018-01-01"

    # Asset Universe (UPDATED: Sector Rotation)
    # We provide distinct correlation profiles so the AI can actually diversify.
    CORE_EQUITIES: Tuple[str, ...] = ("SPY", "QQQ", "SMH", "XLE", "XLV")

    # Keep the rest the same...
    INTL_EQUITIES: Tuple[str, ...] = ( )
    BONDS: Tuple[str, ...] = ("TLT",)
    ALTERNATIVES: Tuple[str, ...] = ("GLD",)
    CASH_PROXY: str = "SHY"
    VOLATILITY_INDEX: str = "^VIX"

    # Transaction Costs
    TRADING_COST_BPS: float = 5.0
    SLIPPAGE_BPS: float = 2.0
    MANAGEMENT_FEE_ANNUAL_BPS: float = 10.0
    REBALANCE_THRESHOLD: float = 0.03

    # Risk Parameters (UPDATED: Higher Volatility Tolerance)
    RISK_FREE_RATE: float = 0.045
    BULL_TARGET_VOL: float = 0.20      # Was 0.16 (Match SPY Vol)
    NEUTRAL_TARGET_VOL: float = 0.15   # Was 0.12 (Allow room to run)
    BEAR_TARGET_VOL: float = 0.08      # Keep defensive
    MAX_DRAWDOWN_LIMIT: float = 0.25   # Loosen slightly

    # ML Settings
    PREDICTION_HORIZON: int = 21
    BULL_VOL_MULTIPLIER: float = 0.65
    BEAR_VOL_MULTIPLIER: float = -0.65
    MIN_SAMPLES_PER_CLASS: int = 100
    MAX_FEATURES_RATIO: float = 0.7
    CV_PURGE_DAYS: int = 21

    # Rolling Optimization
    OPTIMIZATION_LOOKBACK_DAYS: int = 252
    OPTIMIZATION_REFIT_FREQUENCY: int = 21

    # Correlation-Aware Constraints (UPDATED: Less Paranoid)
    CORRELATION_LOOKBACK_DAYS: int = 63
    CORRELATION_POSITIVE_THRESHOLD: float = 0.20
    CORRELATION_NEGATIVE_THRESHOLD: float = -0.20
    BOND_MIN_NORMAL: float = 0.30
    BOND_MIN_POSITIVE_CORR: float = 0.05
    BOND_MAX_POSITIVE_CORR: float = 0.15
    ALT_BOOST_POSITIVE_CORR: float = 0.15

    # Monte Carlo
    MC_SIMULATIONS: int = 50000
    MC_HORIZON_YEARS: int = 5

    # ──────────────────────────────────────────────────────────────────────
    # COMPUTED PROPERTIES - FIX THESE
    # ──────────────────────────────────────────────────────────────────────
    @property
    def TOTAL_TRANSACTION_COST(self) -> float:
        return (self.TRADING_COST_BPS + self.SLIPPAGE_BPS) / 10000

    @property
    def DAILY_MANAGEMENT_FEE(self) -> float:
        return (self.MANAGEMENT_FEE_ANNUAL_BPS / 10000) / 252

    @property
    def ALL_TRADEABLE_TICKERS(self) -> List[str]:
        """
        Assets that can be held in the portfolio.
        NOTE: ^VIX is NOT here - it's only for data/features!
        """
        tickers = []

        # Add each asset class
        tickers.extend(list(self.CORE_EQUITIES))
        tickers.extend(list(self.INTL_EQUITIES))
        tickers.extend(list(self.BONDS))
        tickers.extend(list(self.ALTERNATIVES))

        # Add cash proxy if not already included
        if self.CASH_PROXY and self.CASH_PROXY not in tickers:
            tickers.append(self.CASH_PROXY)

        return tickers

    @property
    def ALL_DATA_TICKERS(self) -> List[str]:
        """
        All tickers needed for data download (includes VIX for features).
        """
        tickers = self.ALL_TRADEABLE_TICKERS.copy()
        # Add VIX only for feature calculation, NOT for trading
        if self.VOLATILITY_INDEX not in tickers:
            tickers.append(self.VOLATILITY_INDEX)
        return tickers

# ==============================================================================
# SECTION 2: CORRELATION REGIME DETECTOR
# ==============================================================================

class CorrelationRegimeDetector:
    """Detects Stock-Bond correlation regime to avoid Bond Trap."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self._correlation_history: List[Dict] = []

    def compute_correlation(self, prices: pd.DataFrame,
                            as_of_date: pd.Timestamp) -> float:
        """Compute trailing stock-bond correlation."""
        lookback = self.config.CORRELATION_LOOKBACK_DAYS
        available = prices[prices.index <= as_of_date]

        if len(available) < lookback:
            return -0.30

        window = available.iloc[-lookback:]
        spy_returns = window['SPY'].pct_change().dropna()

        if 'TLT' in window.columns:
            bond_returns = window['TLT'].pct_change().dropna()
        elif 'IEF' in window.columns:
            bond_returns = window['IEF'].pct_change().dropna()
        else:
            return -0.30

        common_idx = spy_returns.index.intersection(bond_returns.index)
        if len(common_idx) < 21:
            return -0.30

        return spy_returns.loc[common_idx].corr(bond_returns.loc[common_idx])

    def detect_regime(self, prices: pd.DataFrame,
                      as_of_date: pd.Timestamp) -> Tuple[CorrelationRegime, float]:
        """Detect correlation regime."""
        correlation = self.compute_correlation(prices, as_of_date)

        # Determine regime based on thresholds
        if correlation >= self.config.CORRELATION_POSITIVE_THRESHOLD:
            regime = CorrelationRegime.POSITIVE  # BOND TRAP!
        elif correlation <= self.config.CORRELATION_NEGATIVE_THRESHOLD:
            regime = CorrelationRegime.NEGATIVE  # Safe - bonds hedge
        else:
            regime = CorrelationRegime.NEUTRAL  # Uncertain

        self._correlation_history.append({
            'date': as_of_date,
            'correlation': correlation,
            'regime': regime
        })

        return regime, correlation

    def get_correlation_timeseries(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling correlation timeseries."""
        lookback = self.config.CORRELATION_LOOKBACK_DAYS
        spy_returns = prices['SPY'].pct_change()

        if 'TLT' in prices.columns:
            bond_returns = prices['TLT'].pct_change()
        elif 'IEF' in prices.columns:
            bond_returns = prices['IEF'].pct_change()
        else:
            return pd.DataFrame()

        rolling_corr = spy_returns.rolling(lookback).corr(bond_returns)

        result = pd.DataFrame({
            'correlation': rolling_corr,
            'regime': 'neutral'
        })

        result.loc[rolling_corr >= self.config.CORRELATION_POSITIVE_THRESHOLD, 'regime'] = 'positive'
        result.loc[rolling_corr <= self.config.CORRELATION_NEGATIVE_THRESHOLD, 'regime'] = 'negative'

        return result


# ==============================================================================
# SECTION 3: DATA MANAGEMENT
# ==============================================================================

class DataManager:
    """Handles data acquisition and preprocessing."""

    def __init__(self, config: SystemConfig):
        self.config = config

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_market_data(tickers: List[str], start_date: str) -> pd.DataFrame:
        """Load market data with CSV caching to ensure stability."""
        import os

        file_path = "market_data_cache.csv"

        # 1. Try to load from local CSV first
        if os.path.exists(file_path):
            print(f"    Loading data from local cache: {file_path}")
            prices = pd.read_csv(file_path, index_col=0, parse_dates=True)
            # Ensure all requested tickers are present
            if all(t in prices.columns for t in tickers):
                return prices

        # 2. If no cache, download from Yahoo
        print("    Downloading fresh data from Yahoo Finance...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    tickers, start=start_date, auto_adjust=True,
                    progress=False, threads=True
                )

                if isinstance(data.columns, pd.MultiIndex):
                    prices = data['Close'].copy()
                else:
                    prices = pd.DataFrame(data['Close'])
                    prices.columns = [tickers[0]] if len(tickers) == 1 else tickers

                if prices.index.tz is not None:
                    prices.index = prices.index.tz_localize(None)

                # Loose forward fill to prevent dropping too much data
                prices = prices.ffill(limit=5).dropna()

                if len(prices) < 252:
                    raise ValueError(f"Insufficient data: {len(prices)} days")

                # 3. Save to cache
                prices.to_csv(file_path)
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
# SECTION 4: FEATURE ENGINEERING
# ==============================================================================

class FeatureEngine:
    """Feature engineering with leakage prevention."""

    def __init__(self, config: SystemConfig):
        self.config = config

    def build_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Build features using only past information."""
        features = pd.DataFrame(index=prices.index)

        spy = prices['SPY']
        vix = prices['^VIX'] if '^VIX' in prices.columns else pd.Series(20, index=prices.index)
        returns = spy.pct_change()

        # Trend Features
        for window in [20, 50, 100, 200]:
            sma = spy.rolling(window).mean()
            features[f'dist_sma_{window}'] = (spy - sma) / sma

        features['sma_20_50_cross'] = (spy.rolling(20).mean() > spy.rolling(50).mean()).astype(int)
        features['sma_50_200_cross'] = (spy.rolling(50).mean() > spy.rolling(200).mean()).astype(int)

        # Momentum Features
        for period in [5, 10, 21, 63, 126, 252]:
            features[f'momentum_{period}d'] = spy.pct_change(period)

        features['momentum_accel'] = features['momentum_21d'] - features['momentum_21d'].shift(21)

        # Volatility Features
        for window in [10, 21, 63]:
            features[f'volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(252)

        features['vol_regime'] = features['volatility_21d'] / features['volatility_63d']
        features['vol_percentile'] = features['volatility_21d'].rolling(252, min_periods=63).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )

        # VIX Features
        features['vix'] = vix
        features['vix_sma_ratio'] = vix / vix.rolling(21).mean()
        features['vix_percentile'] = vix.rolling(252, min_periods=63).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )
        features['vix_term'] = vix / vix.rolling(63).mean()
        vix_upper = vix.rolling(21).mean() + 2 * vix.rolling(21).std()
        features['vix_spike'] = (vix > vix_upper).astype(int)

        # RSI
        for period in [7, 14, 21]:
            delta = returns
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / avg_loss.replace(0, np.inf)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        features['rsi_divergence'] = features['rsi_14'] - features['rsi_14'].rolling(21).mean()

        # MACD
        ema_12 = spy.ewm(span=12, adjust=False).mean()
        ema_26 = spy.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        features['macd'] = macd / spy * 100
        features['macd_signal'] = macd_signal / spy * 100
        features['macd_histogram'] = (macd - macd_signal) / spy * 100

        # Bollinger Bands
        bb_sma = spy.rolling(20).mean()
        bb_std = spy.rolling(20).std()
        bb_upper = bb_sma + 2 * bb_std
        bb_lower = bb_sma - 2 * bb_std
        features['bb_width'] = (bb_upper - bb_lower) / bb_sma
        features['bb_position'] = (spy - bb_lower) / (bb_upper - bb_lower)

        # Drawdown
        rolling_max = spy.expanding().max()
        features['drawdown'] = (spy - rolling_max) / rolling_max

        # Cross-Asset Features
        if 'TLT' in prices.columns:
            features['spy_tlt_corr'] = (
                prices['SPY'].pct_change().rolling(63)
                .corr(prices['TLT'].pct_change())
            )
            ratio = prices['SPY'] / prices['TLT']
            features['spy_tlt_ratio_zscore'] = (
                    (ratio - ratio.rolling(63).mean()) / ratio.rolling(63).std()
            )
            features['tlt_momentum_21d'] = prices['TLT'].pct_change(21)
            features['tlt_momentum_63d'] = prices['TLT'].pct_change(63)

        if 'GLD' in prices.columns:
            ratio = prices['SPY'] / prices['GLD']
            features['spy_gld_ratio_zscore'] = (
                    (ratio - ratio.rolling(63).mean()) / ratio.rolling(63).std()
            )
            features['gld_momentum_21d'] = prices['GLD'].pct_change(21)

        if 'QQQ' in prices.columns:
            ratio = prices['QQQ'] / prices['SPY']
            features['qqq_spy_momentum'] = ratio.pct_change(21)

        features = features.replace([np.inf, -np.inf], np.nan)
        return features

    def get_ml_feature_columns(self) -> List[str]:
        """Return list of features for ML model - EXPANDED."""
        return [
            # Trend
            'dist_sma_20', 'dist_sma_50', 'dist_sma_100', 'dist_sma_200',
            'sma_20_50_cross', 'sma_50_200_cross',

            # Momentum (more periods)
            'momentum_5d', 'momentum_10d', 'momentum_21d', 'momentum_63d',
            'momentum_126d', 'momentum_252d', 'momentum_accel',

            # Volatility
            'volatility_10d', 'volatility_21d', 'volatility_63d',
            'vol_regime', 'vol_percentile',

            # VIX
            'vix', 'vix_sma_ratio', 'vix_percentile', 'vix_term', 'vix_spike',

            # Technical
            'rsi_7', 'rsi_14', 'rsi_21', 'rsi_divergence',
            'macd', 'macd_signal', 'macd_histogram',
            'bb_width', 'bb_position',

            # Market State
            'drawdown',

            # Cross-Asset
            'spy_tlt_corr', 'spy_tlt_ratio_zscore',
            'tlt_momentum_21d', 'tlt_momentum_63d',
            'spy_gld_ratio_zscore', 'gld_momentum_21d',
            'qqq_spy_momentum'
        ]

    def prepare_ml_data(self, features: pd.DataFrame) -> pd.DataFrame:
        """Select ML features."""
        ml_cols = [c for c in self.get_ml_feature_columns() if c in features.columns]
        return features[ml_cols].copy()


# ==============================================================================
# SECTION 5: TARGET GENERATOR
# ==============================================================================

class TargetGenerator:
    """Volatility-adjusted regime target creation."""

    def __init__(self, config: SystemConfig):
        self.config = config

    def create_regime_target(self, spy_prices: pd.Series) -> Tuple[pd.Series, pd.DataFrame]:
        """Create volatility-adjusted regime labels."""
        horizon = self.config.PREDICTION_HORIZON
        forward_returns = spy_prices.shift(-horizon) / spy_prices - 1

        daily_returns = spy_prices.pct_change()
        trailing_vol_daily = daily_returns.rolling(21).std()
        trailing_vol_horizon = trailing_vol_daily * np.sqrt(horizon)

        bull_threshold = self.config.BULL_VOL_MULTIPLIER * trailing_vol_horizon
        bear_threshold = self.config.BEAR_VOL_MULTIPLIER * trailing_vol_horizon

        target = pd.Series(index=spy_prices.index, dtype=float)
        target[forward_returns >= bull_threshold] = 2
        target[forward_returns <= bear_threshold] = 0
        target[(forward_returns > bear_threshold) & (forward_returns < bull_threshold)] = 1

        thresholds = pd.DataFrame({
            'bull_threshold': bull_threshold,
            'bear_threshold': bear_threshold,
            'trailing_vol': trailing_vol_horizon,
            'forward_return': forward_returns
        })

        return target, thresholds

# ==============================================================================
# SECTION 6: ML REGIME CLASSIFIER (FIXED - MANUAL STACKING)
# ==============================================================================

class PurgedTimeSeriesSplit:
    """
    Time series CV with purging to prevent leakage.

    The purge gap ensures that the target's forward-looking nature
    doesn't leak information from train to validation.
    """

    def __init__(self, n_splits: int = 5, purge_days: int = 21):
        self.n_splits = n_splits
        self.purge_days = purge_days

    def split(self, X):
        """Generate purged train/validation indices."""
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        test_size = n_samples // (self.n_splits + 1)

        indices = []
        for i in range(self.n_splits):
            train_end = test_size * (i + 1)
            val_start = train_end + self.purge_days
            val_end = min(val_start + test_size, n_samples)

            if val_start >= n_samples:
                continue

            train_indices = np.arange(0, train_end)
            val_indices = np.arange(val_start, val_end)

            indices.append((train_indices, val_indices))

        return indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class ManualStackingClassifier:
    """
    Manual implementation of stacking classifier that works with purged CV.

    This avoids the sklearn StackingClassifier issue with non-partition CV splits.

    Architecture:
        Level 0: Base models (GB, RF, MLP, XGB, LGBM)
        Level 1: Meta-learner (Logistic Regression) trained on base model predictions
    """

    def __init__(self, base_models: List[Tuple[str, Any]],
                 meta_learner: Any,
                 cv_splitter: PurgedTimeSeriesSplit,
                 use_proba: bool = True):
        """
        Args:
            base_models: List of (name, model) tuples
            meta_learner: Final model to combine base predictions
            cv_splitter: Cross-validation splitter
            use_proba: If True, use predict_proba for meta features
        """
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.cv_splitter = cv_splitter
        self.use_proba = use_proba

        # Fitted models
        self._fitted_base_models: List[Tuple[str, Any]] = []
        self._fitted_meta_learner = None
        self._n_classes = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ManualStackingClassifier':
        """
        Fit the stacking classifier.

        Step 1: Generate out-of-fold predictions for meta-learner training
        Step 2: Train base models on full training data
        Step 3: Train meta-learner on out-of-fold predictions
        """
        self._n_classes = len(np.unique(y))
        n_samples = len(X)

        # ════════════════════════════════════════════════════════════════
        # STEP 1: Generate out-of-fold predictions
        # ════════════════════════════════════════════════════════════════
        if self.use_proba:
            # For each base model, we need n_samples x n_classes predictions
            n_meta_features = len(self.base_models) * self._n_classes
        else:
            n_meta_features = len(self.base_models)

        meta_features = np.zeros((n_samples, n_meta_features))
        meta_mask = np.zeros(n_samples, dtype=bool)  # Track which samples have predictions

        cv_splits = self.cv_splitter.split(X)

        for train_idx, val_idx in cv_splits:
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold = y[train_idx]

            for model_idx, (name, model) in enumerate(self.base_models):
                # Clone and fit model on this fold
                model_clone = self._clone_model(model)

                try:
                    model_clone.fit(X_train_fold, y_train_fold)

                    if self.use_proba and hasattr(model_clone, 'predict_proba'):
                        preds = model_clone.predict_proba(X_val_fold)
                        start_col = model_idx * self._n_classes
                        end_col = start_col + self._n_classes
                        meta_features[val_idx, start_col:end_col] = preds
                    else:
                        preds = model_clone.predict(X_val_fold)
                        meta_features[val_idx, model_idx] = preds

                    meta_mask[val_idx] = True

                except Exception as e:
                    print(f"    Warning: {name} failed on fold: {e}")
                    continue

        # ════════════════════════════════════════════════════════════════
        # STEP 2: Train base models on FULL training data
        # ════════════════════════════════════════════════════════════════
        self._fitted_base_models = []

        for name, model in self.base_models:
            model_clone = self._clone_model(model)
            try:
                model_clone.fit(X, y)
                self._fitted_base_models.append((name, model_clone))
            except Exception as e:
                print(f"    Warning: {name} failed on full data: {e}")
                continue

        # ════════════════════════════════════════════════════════════════
        # STEP 3: Train meta-learner on out-of-fold predictions
        # ════════════════════════════════════════════════════════════════
        # Only use samples that have valid OOF predictions
        valid_idx = meta_mask
        X_meta = meta_features[valid_idx]
        y_meta = y[valid_idx]

        if len(X_meta) < 100:
            print(f"    Warning: Only {len(X_meta)} samples for meta-learner")

        self._fitted_meta_learner = self._clone_model(self.meta_learner)
        self._fitted_meta_learner.fit(X_meta, y_meta)

        return self

    def _clone_model(self, model):
        """Create a fresh copy of a model."""
        from sklearn.base import clone
        try:
            return clone(model)
        except:
            # For models that don't support sklearn clone
            import copy
            return copy.deepcopy(model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        meta_features = self._generate_meta_features(X)
        return self._fitted_meta_learner.predict(meta_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        meta_features = self._generate_meta_features(X)

        if hasattr(self._fitted_meta_learner, 'predict_proba'):
            return self._fitted_meta_learner.predict_proba(meta_features)
        else:
            # Fallback: convert predictions to one-hot
            preds = self._fitted_meta_learner.predict(meta_features)
            proba = np.zeros((len(preds), self._n_classes))
            for i, p in enumerate(preds):
                proba[i, int(p)] = 1.0
            return proba

    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-features from base model predictions."""
        if self.use_proba:
            n_meta_features = len(self._fitted_base_models) * self._n_classes
        else:
            n_meta_features = len(self._fitted_base_models)

        meta_features = np.zeros((len(X), n_meta_features))

        for model_idx, (name, model) in enumerate(self._fitted_base_models):
            try:
                if self.use_proba and hasattr(model, 'predict_proba'):
                    preds = model.predict_proba(X)
                    start_col = model_idx * self._n_classes
                    end_col = start_col + self._n_classes
                    meta_features[:, start_col:end_col] = preds
                else:
                    preds = model.predict(X)
                    meta_features[:, model_idx] = preds
            except Exception as e:
                print(f"    Warning: {name} prediction failed: {e}")
                continue

        return meta_features


class RegimeClassifier:
    """
    Ensemble ML classifier with STRICT leakage prevention.

    FIXES IMPLEMENTED:
        A. Scaler fitted ONLY on training data
        B. Purged CV prevents target leakage
        C. Manual stacking avoids sklearn partition issue

    OVERFITTING PREVENTION:
        1. Strong regularization
        2. Early stopping
        3. Feature selection
        4. Ensemble averaging
        5. Purged CV with gap
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.model: Optional[ManualStackingClassifier] = None
        self.scaler: Optional[RobustScaler] = None
        self._is_fitted = False
        self._feature_importance: Optional[pd.Series] = None
        self._selected_features: Optional[List[str]] = None
        self._feature_indices: Optional[List[int]] = None
        self._training_metrics: Dict[str, float] = {}

    def _build_base_models(self) -> List[Tuple[str, Any]]:
        """Build STRONGER base models."""
        models = [
            ('gb', GradientBoostingClassifier(
                n_estimators=200,          # More trees
                max_depth=4,               # Slightly deeper
                learning_rate=0.075,        # Slower learning
                subsample=0.8,
                min_samples_leaf=50,
                min_samples_split=70,
                max_features='sqrt',
                validation_fraction=0.15,
                n_iter_no_change=25,
                random_state=42
            )),
            ('rf', RandomForestClassifier(
                n_estimators=300,          # More trees
                max_depth=8,               # Deeper
                min_samples_leaf=25,
                min_samples_split=50,
                max_features=0.4,
                class_weight='balanced_subsample',  # Better for imbalanced
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1
            )),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),  # Bigger network
                activation='relu',
                alpha=0.01,                # Less regularization
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=42
            ))
        ]

        if XGBOOST_AVAILABLE:
            models.append(('xgb', XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=30,
                reg_alpha=0.3,
                reg_lambda=1.5,
                gamma=0.05,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss',
                verbosity=0
            )))

        if LIGHTGBM_AVAILABLE:
            models.append(('lgbm', LGBMClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=30,
                reg_alpha=0.3,
                reg_lambda=1.5,
                random_state=42,
                verbose=-1
            )))

        return models
    def _select_features(self, X: np.ndarray, y: np.ndarray,
                         feature_names: List[str]) -> List[int]:
        """Feature selection to prevent overfitting."""
        selector = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=42
        )
        selector.fit(X, y)

        importances = selector.feature_importances_
        n_samples = len(X)
        max_features = int(self.config.MAX_FEATURES_RATIO * np.sqrt(n_samples))
        max_features = max(5, min(max_features, len(feature_names)))

        top_indices = np.argsort(importances)[-max_features:]
        self._selected_features = [feature_names[i] for i in top_indices]

        # Store feature importance
        self._feature_importance = pd.Series(
            importances, index=feature_names
        ).sort_values(ascending=False)

        return sorted(top_indices)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'RegimeClassifier':
        """
        Fit the classifier with proper train/test separation.

        FIX A: Scaler is fitted ONLY on X_train.
        """
        print("    Cleaning data...")
        valid_mask = X_train.notna().all(axis=1) & y_train.notna()
        X_clean = X_train[valid_mask].copy()
        y_clean = y_train[valid_mask].astype(int)

        feature_names = X_clean.columns.tolist()

        # ════════════════════════════════════════════════════════════════
        # FIT SCALER ON TRAINING DATA ONLY
        # ═══════════���════════════════════════════════════════════════════
        print("    Fitting scaler on training data only...")
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_clean)

        # ════════════════════════════════════════════════════════════════
        # FEATURE SELECTION
        # ════════════════════════════════════════════════════════════════
        print("    Selecting features...")
        selected_indices = self._select_features(X_scaled, y_clean, feature_names)
        X_selected = X_scaled[:, selected_indices]
        self._feature_indices = selected_indices

        print(f"    Selected {len(selected_indices)} features:")
        for feat in self._selected_features[:5]:
            print(f"      - {feat}")
        if len(self._selected_features) > 5:
            print(f"      ... and {len(self._selected_features) - 5} more")

        # ════════════════════════════════════════════════════════════════
        # BUILD MANUAL STACKING CLASSIFIER
        # ════════════════════════════════════════════════════════════════
        print("    Building ensemble model...")
        base_models = self._build_base_models()

        meta_learner = LogisticRegression(
            C=0.5,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )

        cv_splitter = PurgedTimeSeriesSplit(
            n_splits=5,
            purge_days=self.config.CV_PURGE_DAYS
        )

        self.model = ManualStackingClassifier(
            base_models=base_models,
            meta_learner=meta_learner,
            cv_splitter=cv_splitter,
            use_proba=True
        )

        print("    Training ensemble (this may take a minute)...")
        self.model.fit(X_selected, y_clean.values)

        # ════════════════════════════════════════════════════════════════
        # COMPUTE TRAINING METRICS
        # ════════════════════════════════════════════════════════════════
        y_train_pred = self.model.predict(X_selected)
        y_train_proba = self.model.predict_proba(X_selected)

        self._training_metrics['train_accuracy'] = accuracy_score(y_clean, y_train_pred)
        self._training_metrics['train_log_loss'] = log_loss(y_clean, y_train_proba)

        # ════════════════════════════════════════════════════════════════
        # VALIDATION METRICS (if provided)
        # ════════════════════════════════════════════════════════════════
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self._training_metrics['val_accuracy'] = val_metrics['accuracy']
            self._training_metrics['val_log_loss'] = val_metrics.get('log_loss', None)

            overfit_gap = (self._training_metrics['train_accuracy'] -
                           self._training_metrics['val_accuracy'])
            self._training_metrics['overfit_gap'] = overfit_gap

            if overfit_gap > 0.10:
                print(f"    ⚠️ WARNING: Potential overfitting detected!")
                print(f"       Train acc: {self._training_metrics['train_accuracy']:.3f}")
                print(f"       Val acc:   {self._training_metrics['val_accuracy']:.3f}")
                print(f"       Gap:       {overfit_gap:.3f}")

        self._is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        X_filled = X.fillna(X.mean())
        X_scaled = self.scaler.transform(X_filled)
        X_selected = X_scaled[:, self._feature_indices]

        return self.model.predict_proba(X_selected)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime labels."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        X_filled = X.fillna(X.mean())
        X_scaled = self.scaler.transform(X_filled)
        X_selected = X_scaled[:, self._feature_indices]

        return self.model.predict(X_selected)

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
            'confusion_matrix': confusion_matrix(y_clean, y_pred)
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

        Each fold fits scaler only on training portion.
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

        # Simple model for CV
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
        """Return feature importance rankings."""
        return self._feature_importance

# ==============================================================================
# SECTION 7: CORRELATION-AWARE PORTFOLIO OPTIMIZER (AGGRESSIVE VERSION)
# ==============================================================================

# ==============================================================================
# SECTION 7: CORRELATION-AWARE PORTFOLIO OPTIMIZER (SMART AGGRESSION FIX)
# ==============================================================================

class CorrelationAwareOptimizer:
    """Portfolio optimizer that adapts to stock-bond correlation regime."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.correlation_detector = CorrelationRegimeDetector(config)
        self.regime_portfolios: Dict[MarketRegime, Dict[str, float]] = {}
        self.regime_metrics: Dict[MarketRegime, Dict[str, float]] = {}
        self._current_correlation_regime: CorrelationRegime = CorrelationRegime.NEGATIVE
        self._current_correlation: float = -0.30

    def _get_adaptive_constraints(self, market_regime: MarketRegime,
                                  correlation_regime: CorrelationRegime,
                                  correlation_value: float) -> Dict:
        """ULTRA-AGGRESSIVE constraints with FORCED diversification."""
        base_constraints = {
            MarketRegime.BULL: {
                'target_vol': self.config.BULL_TARGET_VOL,
                'equity_min': 0.90, 'equity_max': 1.00,
                'bond_min': 0.00, 'bond_max': 0.05,
                'alt_min': 0.00, 'alt_max': 0.05,
                'cash_min': 0.00, 'cash_max': 0.02
            },
            MarketRegime.NEUTRAL: {
                'target_vol': self.config.NEUTRAL_TARGET_VOL,
                'equity_min': 0.75, 'equity_max': 0.95,
                'bond_min': 0.00, 'bond_max': 0.10,
                'alt_min': 0.00, 'alt_max': 0.10,
                'cash_min': 0.00, 'cash_max': 0.05
            },
            MarketRegime.BEAR: {
                'target_vol': self.config.BEAR_TARGET_VOL,
                'equity_min': 0.40, 'equity_max': 0.65,
                'bond_min': 0.10, 'bond_max': 0.30,
                'alt_min': 0.10, 'alt_max': 0.25,
                'cash_min': 0.05, 'cash_max': 0.20
            }
        }

        constraints = base_constraints[market_regime].copy()

        # Bond trap: kill bonds completely
        if correlation_regime == CorrelationRegime.POSITIVE:
            constraints['bond_min'] = 0.00
            constraints['bond_max'] = 0.00

            if market_regime == MarketRegime.BEAR:
                constraints['alt_min'] = 0.15
                constraints['alt_max'] = 0.30
                constraints['cash_min'] = 0.10

        return constraints

    def optimize_for_date(self, returns: pd.DataFrame,
                          prices: pd.DataFrame,
                          as_of_date: pd.Timestamp,
                          market_regime: MarketRegime) -> Dict[str, float]:
        """Optimize portfolio for a specific date and regime."""
        tradeable = self.config.ALL_TRADEABLE_TICKERS
        returns = returns[[c for c in returns.columns if c in tradeable]]

        # Detect correlation regime
        correlation_regime, correlation_value = self.correlation_detector.detect_regime(
            prices, as_of_date
        )

        self._current_correlation_regime = correlation_regime
        self._current_correlation = correlation_value

        # Get adaptive constraints
        constraints = self._get_adaptive_constraints(
            market_regime, correlation_regime, correlation_value
        )

        # Use rolling window
        lookback = self.config.OPTIMIZATION_LOOKBACK_DAYS
        available_returns = returns[returns.index < as_of_date]
        window_returns = available_returns.iloc[-lookback:] if len(available_returns) > lookback else available_returns

        if len(window_returns) < 63:
            tickers = returns.columns.tolist()
            return {t: 1 / len(tickers) for t in tickers}

        return self._optimize_with_constraints(window_returns, constraints)

    def _optimize_with_constraints(self, returns: pd.DataFrame,
                                   constraints: Dict) -> Dict[str, float]:
        """Run optimization with FORCED DIVERSIFICATION (Max 35% per asset)."""

        tradeable = self.config.ALL_TRADEABLE_TICKERS
        available_cols = [c for c in returns.columns if c in tradeable]
        returns = returns[available_cols]
        tickers = returns.columns.tolist()
        n_assets = len(tickers)

        # Setup Data
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        shrinkage = 0.1
        cov_matrix = (1 - shrinkage) * cov_matrix + shrinkage * np.diag(np.diag(cov_matrix))

        # Objective: Maximize Risk-Adjusted Return
        def objective(weights):
            port_ret = np.sum(mean_returns.values * weights)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
            risk_aversion = 1.0
            return -(port_ret - risk_aversion * port_vol)

        # Build Indices
        idx_map = {t: i for i, t in enumerate(tickers)}

        def get_indices(ticker_list):
            return [idx_map[t] for t in ticker_list if t in idx_map]

        us_idx = get_indices(self.config.CORE_EQUITIES)
        bond_idx = get_indices(self.config.BONDS)
        alt_idx = get_indices(self.config.ALTERNATIVES)
        cash_idx = get_indices([self.config.CASH_PROXY])

        # Build Constraints
        opt_constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]

        def add_limit(indices, min_key, max_key):
            if indices:
                opt_constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=indices, mk=min_key: np.sum(x[idx]) - constraints.get(mk, 0.0)
                })
                opt_constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=indices, mk=max_key: constraints.get(mk, 1.0) - np.sum(x[idx])
                })

        add_limit(us_idx, 'equity_min', 'equity_max')
        add_limit(bond_idx, 'bond_min', 'bond_max')
        add_limit(alt_idx, 'alt_min', 'alt_max')
        add_limit(cash_idx, 'cash_min', 'cash_max')

        # 🚀 THE FIX: INDIVIDUAL ASSET LIMITS
        # Force diversification by capping every single stock at 35%
        # This forces the solver to buy at least 3 assets to get to 100% equity.
        # Exception: We let Cash/Bonds be lower if constraints demand it.

        MAX_SINGLE_ASSET_WEIGHT = 0.35

        bounds = []
        for i in range(n_assets):
            # Default bounds (0% to 35%)
            lower = 0.0
            upper = MAX_SINGLE_ASSET_WEIGHT

            # If an asset class (like Bonds) has a constraint lower than 35%,
            # we don't need to do anything (the constraint handles it).
            # But if the class allows 100% (like Equities), this 35% cap is the safety net.
            bounds.append((lower, upper))

        bounds = tuple(bounds)

        # Smart Initialization (Respecting the 35% Cap)
        init_weights = np.zeros(n_assets)
        eq_target = constraints.get('equity_min', 0.5) + 0.05
        eq_target = min(eq_target, constraints.get('equity_max', 1.0))

        if us_idx:
            # Spread equity target evenly so no single start weight exceeds 35%
            # e.g., if Target is 100% and we have 5 assets -> 20% each (Safe)
            weight_per_asset = eq_target / len(us_idx)

            # Clip just in case
            weight_per_asset = min(weight_per_asset, MAX_SINGLE_ASSET_WEIGHT - 0.01)

            for i in us_idx:
                init_weights[i] = weight_per_asset

        # Distribute remainder
        remainder = 1.0 - np.sum(init_weights)
        allowed_others = []
        if bond_idx and constraints.get('bond_max', 0) > 0.01: allowed_others.extend(bond_idx)
        if alt_idx and constraints.get('alt_max', 0) > 0.01: allowed_others.extend(alt_idx)
        if cash_idx and constraints.get('cash_max', 0) > 0.01: allowed_others.extend(cash_idx)

        # If remainder is large, dump it into the NEXT available equity assets
        # (This handles the case where simple even split didn't use up all capital)
        if remainder > 0.01:
            if allowed_others:
                for i in allowed_others: init_weights[i] += remainder / len(allowed_others)
            elif us_idx:
                # Spread small remainder back to equities
                for i in us_idx:
                    space_left = MAX_SINGLE_ASSET_WEIGHT - init_weights[i]
                    add_amount = min(space_left, remainder)
                    init_weights[i] += add_amount
                    remainder -= add_amount
                    if remainder < 0.001: break

        # Normalize
        if np.sum(init_weights) > 0:
            init_weights = init_weights / np.sum(init_weights)

        # Optimize
        result = minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=opt_constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        weights = result.x if result.success else init_weights
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()

        return dict(zip(tickers, weights))

    def optimize_all_regimes(self, returns: pd.DataFrame,
                             prices: pd.DataFrame,
                             as_of_date: Optional[pd.Timestamp] = None) -> Dict[MarketRegime, Dict[str, float]]:
        """Optimize portfolios for all market regimes."""
        if as_of_date is None:
            as_of_date = returns.index[-1] + pd.Timedelta(days=1)

        for regime in MarketRegime:
            weights = self.optimize_for_date(returns, prices, as_of_date, regime)
            self.regime_portfolios[regime] = weights

            # Simple metrics storage for debugging
            self.regime_metrics[regime] = {'status': 'optimized'}

        return self.regime_portfolios

    def get_blended_weights(self, regime_probs: Dict[MarketRegime, float]) -> Dict[str, float]:
        """
        Winner Take All Logic to prevent dilution in strong regimes.
        """
        if not self.regime_portfolios:
            raise RuntimeError("Must optimize regime portfolios first")

        winner_regime = max(regime_probs, key=regime_probs.get)
        winner_prob = regime_probs[winner_regime]

        # If confident (>60%), use the pure portfolio (e.g. 100% Equity)
        if winner_prob > 0.60:
            return self.regime_portfolios[winner_regime]

        # Otherwise blend
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

    def get_correlation_status(self) -> Dict[str, Any]:
        return {
            'regime': self._current_correlation_regime,
            'value': self._current_correlation,
            'is_bond_trap': self._current_correlation_regime == CorrelationRegime.POSITIVE
        }

# ==============================================================================
# SECTION 8: MONTE CARLO ENGINE (TREND-AWARE UPGRADE)
# ==============================================================================

class MonteCarloEngine:
    """Monte Carlo simulation with Trend-Aware Drift."""

    def __init__(self, config: SystemConfig):
        self.config = config

    def simulate_portfolio(self, returns: pd.DataFrame,
                           weights: Dict[str, float],
                           n_sims: Optional[int] = None,
                           horizon_years: Optional[int] = None,
                           initial_value: float = 100000) -> Dict[str, Any]:
        """Run Monte Carlo simulation using recent trend data."""
        n_sims = n_sims or self.config.MC_SIMULATIONS
        horizon_years = horizon_years or self.config.MC_HORIZON_YEARS
        n_days = 252 * horizon_years

        # Filter to assets in the portfolio
        active_assets = [t for t in weights.keys() if weights[t] > 0]
        if not active_assets:
            return {}  # Should not happen

        # 1. CALCULATE STATISTICS
        # Instead of using the full 20-year history (which includes 2008),
        # we use the last 252 days (1 year) to capture the CURRENT TREND.
        # This aligns the MC with the "Momentum" nature of the strategy.
            # 1. CALCULATE STATISTICS (UPDATED: 5-YEAR REALITY CHECK)
            # Use the last 5 years (1260 days) to include the 2022 Bear Market.
            # This prevents "Bull Market Delusion" while keeping trend relevance.
        lookback_days = 1260

        # Handle case where data is shorter than 5 years
        start_idx = max(0, len(returns) - lookback_days)
        recent_returns = returns[active_assets].iloc[start_idx:]

        # Calculate portfolio historical performance
        weight_arr = np.array([weights[t] for t in active_assets])
        port_returns = recent_returns.values @ weight_arr

        # Annualized stats
        daily_mean = np.mean(port_returns)
        daily_std = np.std(port_returns)

        # Drift = Average Daily Return - Variance Drag
        drift = daily_mean - (0.5 * daily_std ** 2)

        # 2. SIMULATE
        np.random.seed(42)
        # Random shocks
        Z = np.random.standard_normal((n_days, n_sims))

        # Daily returns path
        sim_returns = np.exp(drift + daily_std * Z)

        # Price paths
        price_paths = np.zeros((n_days + 1, n_sims))
        price_paths[0] = initial_value

        for t in range(1, n_days + 1):
            price_paths[t] = price_paths[t - 1] * sim_returns[t - 1]

        # 3. ANALYZE RESULTS
        ending_values = price_paths[-1]
        cagrs = (ending_values / initial_value) ** (1 / horizon_years) - 1

        # Drawdowns (Expensive calculation, optimized)
        max_drawdowns = np.zeros(n_sims)
        for i in range(n_sims):
            path = price_paths[:, i]
            peak = np.maximum.accumulate(path)
            dd = (path - peak) / peak
            max_drawdowns[i] = np.min(dd)

        return {
            'price_paths': price_paths,
            'ending_values': ending_values,
            'cagrs': cagrs,
            'max_drawdowns': max_drawdowns,
            'mean_cagr': np.mean(cagrs),
            'median_cagr': np.median(cagrs),
            'std_cagr': np.std(cagrs),
            'ci_95_lower': np.percentile(ending_values, 2.5),
            'ci_95_upper': np.percentile(ending_values, 97.5),
            'ci_95_lower_cagr': np.percentile(cagrs, 2.5),
            'ci_95_upper_cagr': np.percentile(cagrs, 97.5),
            'prob_loss': np.mean(ending_values < initial_value),
            'prob_double': np.mean(ending_values >= 2 * initial_value),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': np.min(max_drawdowns),
            'annual_volatility': daily_std * np.sqrt(252),
            'sharpe_ratio': (np.mean(cagrs) - self.config.RISK_FREE_RATE) / (daily_std * np.sqrt(252))
        }

    def plot_results(self, results: Dict, title: str = "Monte Carlo (Trend-Aware)",
                     n_paths: int = 200) -> Tuple[plt.Figure, plt.Figure]:
        # ... (Same plotting code as before, just update the title) ...
        # Copy the plotting code from the previous version or leave it as is.
        # Ideally, just paste the previous 'plot_results' method here.
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

        # Figure 2: Return Distribution
        fig2, ax2 = plt.subplots(figsize=(12, 7))

        cagrs = results['cagrs']
        ax2.hist(cagrs, bins=100, density=True, color='#2E86AB',
                 edgecolor='white', alpha=0.8)

        ax2.axvline(results['mean_cagr'], color='red', linewidth=2.5,
                    label=f"Mean: {results['mean_cagr']:.1%}")
        ax2.axvline(results['ci_95_lower_cagr'], color='orange', linewidth=2,
                    linestyle='--', label=f"5th %ile: {results['ci_95_lower_cagr']:.1%}")
        ax2.axvline(results['ci_95_upper_cagr'], color='green', linewidth=2,
                    linestyle='--', label=f"95th %ile: {results['ci_95_upper_cagr']:.1%}")
        ax2.axvline(0, color='black', linewidth=1, linestyle=':')

        ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax2.set_title('Return Distribution (CAGR)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Compound Annual Growth Rate')
        ax2.set_ylabel('Density')
        ax2.legend(loc='upper right')
        ax2.grid(axis='y', alpha=0.3)

        return fig1, fig2

# ==============================================================================
# SECTION 9: BACKTESTING ENGINE (COMPLETE)
# ==============================================================================

class BacktestEngine:
    """Comprehensive backtesting with transaction costs and correlation awareness."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.results = None

    def run_backtest(self,
                     prices: pd.DataFrame,
                     features: pd.DataFrame,
                     returns: pd.DataFrame,
                     regime_classifier: RegimeClassifier,
                     optimizer: CorrelationAwareOptimizer,
                     benchmark_ticker: str = 'SPY') -> Dict[str, Any]:
        """
        Run full backtest with correlation-aware optimization.

        This backtest:
        1. Re-optimizes portfolios periodically to capture changing correlations
        2. Uses ML to predict market regime
        3. Blends portfolios based on regime probabilities
        4. Applies realistic transaction costs
        5. Tracks correlation regime changes
        """
        feature_engine = FeatureEngine(self.config)
        ml_features = feature_engine.prepare_ml_data(features)

        # Get test period
        test_mask = ml_features.index >= self.config.TEST_START
        test_dates = ml_features[test_mask].index
        common_dates = test_dates.intersection(prices.index)

        if len(common_dates) < 2:
            raise ValueError("Insufficient test data")

        # Initialize tracking
        portfolio_values = [100000.0]
        benchmark_values = [100000.0]
        cash_values = [100000.0]

        current_weights: Dict[str, float] = {}
        weight_history = []
        regime_history = []
        regime_prob_history = []
        correlation_history = []
        correlation_regime_history = []
        trade_count = 0
        total_costs = 0.0

        # Daily rates
        daily_rf = self.config.RISK_FREE_RATE / 252
        daily_mgmt_fee = self.config.DAILY_MANAGEMENT_FEE

        # Asset returns
        asset_returns = prices.pct_change().fillna(0)

        # Refit settings
        refit_frequency = self.config.OPTIMIZATION_REFIT_FREQUENCY
        last_refit_idx = -refit_frequency  # Force refit on first day


        print(f"Period: {common_dates[0].date()} to {common_dates[-1].date()}")
        print(f"Total days: {len(common_dates)}")
        print(f"Re-optimization frequency: Every {refit_frequency} days")
        print(f"{'=' * 70}\n")

        for i, date in enumerate(common_dates[1:], 1):
            prev_date = common_dates[i - 1]

            # ═════════════════════��══════════════════════════════════════
            # STEP 1: PERIODIC RE-OPTIMIZATION
            # ═══════════════��════════════════════════════════════════════
            if i - last_refit_idx >= refit_frequency:
                if i % 100 == 0:
                    print(f"  [{i}/{len(common_dates)}] Re-optimizing at {prev_date.date()}...")

                optimizer.optimize_all_regimes(returns, prices, prev_date)
                last_refit_idx = i

            # ════════════════════════════════════════════════════════════
            # STEP 2: GET REGIME PROBABILITIES FROM ML
            # ════════════════════════════════════════════════════════════
            today_features = ml_features.loc[[prev_date]]

            if today_features.isna().all().all():
                regime_probs = {
                    MarketRegime.NEUTRAL: 1.0,
                    MarketRegime.BULL: 0.0,
                    MarketRegime.BEAR: 0.0
                }
                predicted_regime = MarketRegime.NEUTRAL
            else:
                predicted_regime, regime_probs = regime_classifier.predict_regime(today_features)

                # ════════════════════════════════════════════════════════════
                # STEP 3: THE "GOLDEN RULE" OVERRIDE (TREND VETO)
                # ════════════════════════════════════════════════════════════
                spy_price = prices.loc[prev_date, 'SPY']
                # Calculate SMA 200 efficiently
                sma_200 = prices['SPY'].loc[:prev_date].rolling(200).mean().iloc[-1]

                if not np.isnan(sma_200):
                    if spy_price > sma_200:
                        # 🚀 UPTREND DETECTED: VETO THE BEAR
                        # If the ML predicts Bear, it is WRONG. Kill the Bear signal.

                        # 1. Take all probability currently assigned to BEAR
                        bear_prob = regime_probs.get(MarketRegime.BEAR, 0.0)

                        # 2. Re-distribute it to BULL (Aggressive)
                        regime_probs[MarketRegime.BEAR] = 0.0
                        regime_probs[MarketRegime.BULL] += bear_prob

                        # 3. Optional: Boost Bull further if it was Neutral
                        # If we are in an uptrend, we want to be Long, not Neutral.
                        neutral_prob = regime_probs.get(MarketRegime.NEUTRAL, 0.0)
                        if neutral_prob > 0.5:  # If too hesitant
                            shift = neutral_prob * 0.5
                            regime_probs[MarketRegime.NEUTRAL] -= shift
                            regime_probs[MarketRegime.BULL] += shift

                    else:
                        # 📉 DOWNTREND DETECTED: ALLOW DEFENSE
                        # If price is below SMA, we respect the Bear signal or even boost it.
                        regime_probs[MarketRegime.BEAR] = min(1.0, regime_probs.get(MarketRegime.BEAR, 0.0) + 0.20)
                        regime_probs[MarketRegime.BULL] = max(0.0, regime_probs.get(MarketRegime.BULL, 0.0) - 0.20)

                    # Renormalize to ensure sum is 1.0
                    total_prob = sum(regime_probs.values())
                    if total_prob > 0:
                        regime_probs = {k: v / total_prob for k, v in regime_probs.items()}

            # ════════════════════════════════════════════════════════════
            # STEP 4: GET BLENDED TARGET WEIGHTS
            # ════════════════════════════════════════════════════════════
            target_weights = optimizer.get_blended_weights(regime_probs)

            # ════════════════════════════════════════════════════════════
            # STEP 5: APPLY REBALANCE THRESHOLD
            # ════════════════════════════════════════════════════════════
            if current_weights:
                final_weights = self._apply_rebalance_filter(current_weights, target_weights)
            else:
                final_weights = target_weights

            # ════════════════════════════════════════════════════════════
            # STEP 6: CALCULATE TRANSACTION COSTS
            # ════════════════════════════════════════════════════════���═══
            if current_weights:
                turnover = sum(
                    abs(final_weights.get(a, 0) - current_weights.get(a, 0))
                    for a in set(final_weights) | set(current_weights)
                ) / 2
            else:
                turnover = 1.0

            transaction_cost = turnover * self.config.TOTAL_TRANSACTION_COST

            if turnover > 0.01:
                trade_count += 1
                total_costs += transaction_cost * portfolio_values[-1]

            # ════════════════════════════════════════════════════════════
            # STEP 7: CALCULATE DAILY RETURNS
            # ════════════════════════════════════════════════════════════
            strategy_return = sum(
                final_weights.get(ticker, 0) * asset_returns.loc[date, ticker]
                for ticker in final_weights
                if ticker in asset_returns.columns
            )

            # Apply costs
            net_return = strategy_return - transaction_cost - daily_mgmt_fee

            # Benchmark return
            benchmark_return = asset_returns.loc[date, benchmark_ticker]

            # ════════════════════════════════════════════════════════════
            # STEP 8: UPDATE PORTFOLIO VALUES
            # ════════════════════════════════════════════════════════════
            portfolio_values.append(portfolio_values[-1] * (1 + net_return))
            benchmark_values.append(benchmark_values[-1] * (1 + benchmark_return))
            cash_values.append(cash_values[-1] * (1 + daily_rf))

            # Track history
            weight_history.append(final_weights.copy())
            regime_history.append(predicted_regime)
            regime_prob_history.append(regime_probs.copy())

            # Track correlation
            corr_status = optimizer.get_correlation_status()
            correlation_history.append(corr_status['value'])
            correlation_regime_history.append(corr_status['regime'])

            current_weights = final_weights

        # Convert to arrays
        portfolio_values = np.array(portfolio_values)
        benchmark_values = np.array(benchmark_values)
        cash_values = np.array(cash_values)

        # Calculate metrics
        metrics = self._calculate_metrics(
            portfolio_values, benchmark_values, common_dates
        )

        # Add additional stats
        metrics['trade_count'] = trade_count
        metrics['total_costs'] = total_costs
        metrics['avg_annual_trades'] = trade_count / (len(common_dates) / 252)

        # Count bond trap days
        bond_trap_days = sum(1 for r in correlation_regime_history
                             if r == CorrelationRegime.POSITIVE)
        metrics['bond_trap_days'] = bond_trap_days
        metrics['bond_trap_pct'] = bond_trap_days / len(correlation_regime_history) if correlation_regime_history else 0

        self.results = {
            'dates': common_dates,
            'portfolio_values': portfolio_values,
            'benchmark_values': benchmark_values,
            'cash_values': cash_values,
            'weight_history': weight_history,
            'regime_history': regime_history,
            'regime_prob_history': regime_prob_history,
            'correlation_history': correlation_history,
            'correlation_regime_history': correlation_regime_history,
            'metrics': metrics
        }

        return self.results

    def _apply_rebalance_filter(self, current: Dict[str, float],
                                target: Dict[str, float]) -> Dict[str, float]:
        """Only rebalance if change exceeds threshold."""
        threshold = self.config.REBALANCE_THRESHOLD
        all_assets = set(current) | set(target)
        final = {}

        for asset in all_assets:
            curr_w = current.get(asset, 0)
            tgt_w = target.get(asset, 0)

            if abs(tgt_w - curr_w) > threshold:
                final[asset] = tgt_w
            else:
                final[asset] = curr_w

        total = sum(final.values())
        if total > 0:
            final = {k: v / total for k, v in final.items()}

        return final

    def _calculate_metrics(self, portfolio: np.ndarray,
                           benchmark: np.ndarray,
                           dates: pd.DatetimeIndex) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        n_years = len(portfolio) / 252

        # Returns
        port_returns = np.diff(portfolio) / portfolio[:-1]
        bench_returns = np.diff(benchmark) / benchmark[:-1]

        total_return = (portfolio[-1] / portfolio[0]) - 1
        bench_total = (benchmark[-1] / benchmark[0]) - 1

        cagr = (1 + total_return) ** (1 / n_years) - 1
        bench_cagr = (1 + bench_total) ** (1 / n_years) - 1

        # Volatility
        volatility = np.std(port_returns) * np.sqrt(252)
        bench_vol = np.std(bench_returns) * np.sqrt(252)

        # Sharpe Ratio
        excess = port_returns - self.config.RISK_FREE_RATE / 252
        sharpe = np.mean(excess) / np.std(excess) * np.sqrt(252) if np.std(excess) > 0 else 0

        bench_excess = bench_returns - self.config.RISK_FREE_RATE / 252
        bench_sharpe = np.mean(bench_excess) / np.std(bench_excess) * np.sqrt(252) if np.std(bench_excess) > 0 else 0

        # Sortino Ratio (downside deviation)
        downside_returns = port_returns[port_returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino = (cagr - self.config.RISK_FREE_RATE) / downside_std if downside_std > 0 else 0

        # Maximum Drawdown
        rolling_max = np.maximum.accumulate(portfolio)
        drawdowns = (portfolio - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)

        bench_rolling_max = np.maximum.accumulate(benchmark)
        bench_drawdowns = (benchmark - bench_rolling_max) / bench_rolling_max
        bench_max_dd = np.min(bench_drawdowns)

        # Calmar Ratio
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win Rate
        winning_days = np.sum(port_returns > 0)
        total_days = len(port_returns)
        win_rate = winning_days / total_days if total_days > 0 else 0

        # Alpha and Beta
        if len(port_returns) == len(bench_returns) and len(bench_returns) > 0:
            covariance = np.cov(port_returns, bench_returns)
            if covariance.shape == (2, 2) and covariance[1, 1] != 0:
                beta = covariance[0, 1] / covariance[1, 1]
            else:
                beta = 1.0
            alpha = cagr - (self.config.RISK_FREE_RATE + beta * (bench_cagr - self.config.RISK_FREE_RATE))
        else:
            beta = 1.0
            alpha = cagr - bench_cagr

        # Information Ratio
        active_returns = port_returns - bench_returns
        tracking_error = np.std(active_returns) * np.sqrt(252)
        info_ratio = (cagr - bench_cagr) / tracking_error if tracking_error > 0 else 0

        # Avg Drawdown Duration
        in_drawdown = drawdowns < 0
        drawdown_periods = []
        current_period = 0
        for dd in in_drawdown:
            if dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        if current_period > 0:
            drawdown_periods.append(current_period)
        avg_dd_duration = np.mean(drawdown_periods) if drawdown_periods else 0

        return {
            # Strategy Metrics
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_drawdown,
            'calmar': calmar,
            'win_rate': win_rate,
            'alpha': alpha,
            'beta': beta,
            'info_ratio': info_ratio,
            'avg_dd_duration': avg_dd_duration,

            # Benchmark Metrics
            'benchmark_return': bench_total,
            'benchmark_cagr': bench_cagr,
            'benchmark_vol': bench_vol,
            'benchmark_sharpe': bench_sharpe,
            'benchmark_max_dd': bench_max_dd,

            # Comparison
            'excess_return': total_return - bench_total,
            'excess_cagr': cagr - bench_cagr,

            # Final Values
            'final_value': portfolio[-1],
            'benchmark_final': benchmark[-1]
        }

    def plot_results(self) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
        """Generate comprehensive backtest visualization."""
        if self.results is None:
            raise RuntimeError("Must run backtest before plotting")

        dates = self.results['dates']
        portfolio = self.results['portfolio_values']
        benchmark = self.results['benchmark_values']
        cash = self.results['cash_values']
        metrics = self.results['metrics']
        correlation_history = self.results['correlation_history']
        correlation_regime_history = self.results['correlation_regime_history']

        # ════════════════════════════════════════════════════════════════
        # FIGURE 1: PERFORMANCE CHART
        # ════════════════════════════════════════════════════════════════
        fig1, axes1 = plt.subplots(3, 1, figsize=(14, 12),
                                   gridspec_kw={'height_ratios': [3, 1, 1]})

        # Panel 1: Portfolio Value
        ax1 = axes1[0]
        ax1.plot(dates, portfolio, label=f"Strategy (CAGR: {metrics['cagr']:.1%})",
                 color='royalblue', linewidth=2)
        ax1.plot(dates, benchmark, label=f"SPY Benchmark (CAGR: {metrics['benchmark_cagr']:.1%})",
                 color='gray', linewidth=1.5, linestyle='--', alpha=0.7)
        ax1.plot(dates, cash, label='Cash (Risk-Free)',
                 color='green', linewidth=1, linestyle=':', alpha=0.5)

        ax1.set_title('Strategy Performance: Growth of $100,000', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Panel 2: Drawdown
        ax2 = axes1[1]
        rolling_max = np.maximum.accumulate(portfolio)
        drawdown = (portfolio - rolling_max) / rolling_max

        ax2.fill_between(dates, drawdown, 0, color='red', alpha=0.3)
        ax2.plot(dates, drawdown, color='red', linewidth=1)
        ax2.axhline(y=metrics['max_drawdown'], color='darkred', linestyle='--',
                    label=f"Max DD: {metrics['max_drawdown']:.1%}")
        ax2.set_ylabel('Drawdown')
        ax2.set_title('Underwater Chart', fontsize=12)
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)

        # Panel 3: Regime
        ax3 = axes1[2]
        regime_colors = []
        for r in self.results['regime_history']:
            if r == MarketRegime.BULL:
                regime_colors.append('green')
            elif r == MarketRegime.BEAR:
                regime_colors.append('red')
            else:
                regime_colors.append('gray')

        # Plot as colored bars
        regime_numeric = [1 if r == MarketRegime.BULL else (-1 if r == MarketRegime.BEAR else 0)
                          for r in self.results['regime_history']]
        ax3.bar(dates[1:], regime_numeric, color=regime_colors, width=1, alpha=0.7)
        ax3.set_ylabel('Regime')
        ax3.set_yticks([-1, 0, 1])
        ax3.set_yticklabels(['Bear', 'Neutral', 'Bull'])
        ax3.set_xlabel('Date')
        ax3.set_title('ML Regime Prediction', fontsize=12)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        # ════════════════════════════════════════════════════════════════
        # FIGURE 2: CORRELATION & BOND TRAP ANALYSIS
        # ════════════════════════════════════════════════════════════════
        fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8),
                                   gridspec_kw={'height_ratios': [2, 1]})

        # Panel 1: Correlation over time
        ax4 = axes2[0]
        corr_dates = dates[1:len(correlation_history) + 1]
        ax4.plot(corr_dates, correlation_history, color='navy', linewidth=1.5,
                 label='Stock-Bond Correlation (63-day)')

        ax4.axhline(y=self.config.CORRELATION_POSITIVE_THRESHOLD, color='red',
                    linestyle='--', label=f'Positive Threshold ({self.config.CORRELATION_POSITIVE_THRESHOLD})')
        ax4.axhline(y=self.config.CORRELATION_NEGATIVE_THRESHOLD, color='green',
                    linestyle='--', label=f'Negative Threshold ({self.config.CORRELATION_NEGATIVE_THRESHOLD})')
        ax4.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

        # Shade bond trap periods
        positive_mask = np.array(correlation_history) >= self.config.CORRELATION_POSITIVE_THRESHOLD
        ax4.fill_between(corr_dates, -1, 1, where=positive_mask,
                         color='red', alpha=0.2, label='Bond Trap Zone')

        ax4.set_ylabel('Correlation')
        ax4.set_title('Stock-Bond Correlation Regime (Bond Trap Detection)', fontsize=14, fontweight='bold')
        ax4.legend(loc='upper left')
        ax4.set_ylim(-0.8, 0.8)
        ax4.grid(True, alpha=0.3)

        # Panel 2: Correlation regime indicator
        ax5 = axes2[1]
        regime_numeric = []
        regime_colors = []
        for r in correlation_regime_history:
            if r == CorrelationRegime.POSITIVE:
                regime_numeric.append(1)
                regime_colors.append('red')
            elif r == CorrelationRegime.NEGATIVE:
                regime_numeric.append(-1)
                regime_colors.append('green')
            else:
                regime_numeric.append(0)
                regime_colors.append('gray')

        ax5.bar(corr_dates, regime_numeric, color=regime_colors, width=1, alpha=0.7)
        ax5.set_ylabel('Corr Regime')
        ax5.set_yticks([-1, 0, 1])
        ax5.set_yticklabels(['Negative\n(Safe)', 'Neutral', 'Positive\n(TRAP)'])
        ax5.set_xlabel('Date')
        ax5.set_title(f"Bond Trap Days: {metrics['bond_trap_days']} ({metrics['bond_trap_pct']:.1%} of period)",
                      fontsize=12)
        ax5.grid(True, alpha=0.3)

        plt.tight_layout()

        # ════════════════════════════════════════════════════════════════
        # FIGURE 3: METRICS SUMMARY
        # ════════════════════════════════════════════════════════════════
        fig3, ax6 = plt.subplots(figsize=(12, 8))
        ax6.axis('off')

        # Create metrics table
        metrics_data = [
            ['Metric', 'Strategy', 'Benchmark', 'Difference'],
            ['─' * 20, '─' * 15, '─' * 15, '─' * 15],
            ['Total Return', f"{metrics['total_return']:.1%}", f"{metrics['benchmark_return']:.1%}",
             f"{metrics['excess_return']:+.1%}"],
            ['CAGR', f"{metrics['cagr']:.1%}", f"{metrics['benchmark_cagr']:.1%}",
             f"{metrics['excess_cagr']:+.1%}"],
            ['Volatility', f"{metrics['volatility']:.1%}", f"{metrics['benchmark_vol']:.1%}",
             f"{metrics['volatility'] - metrics['benchmark_vol']:+.1%}"],
            ['Sharpe Ratio', f"{metrics['sharpe']:.2f}", f"{metrics['benchmark_sharpe']:.2f}",
             f"{metrics['sharpe'] - metrics['benchmark_sharpe']:+.2f}"],
            ['Sortino Ratio', f"{metrics['sortino']:.2f}", 'N/A', 'N/A'],
            ['Max Drawdown', f"{metrics['max_drawdown']:.1%}", f"{metrics['benchmark_max_dd']:.1%}",
             f"{metrics['max_drawdown'] - metrics['benchmark_max_dd']:+.1%}"],
            ['Calmar Ratio', f"{metrics['calmar']:.2f}", 'N/A', 'N/A'],
            ['Win Rate', f"{metrics['win_rate']:.1%}", 'N/A', 'N/A'],
            ['Alpha', f"{metrics['alpha']:.2%}", 'N/A', 'N/A'],
            ['Beta', f"{metrics['beta']:.2f}", '1.00', 'N/A'],
            ['Info Ratio', f"{metrics['info_ratio']:.2f}", 'N/A', 'N/A'],
            ['─' * 20, '─' * 15, '─' * 15, '─' * 15],
            ['Final Value', f"${metrics['final_value']:,.0f}", f"${metrics['benchmark_final']:,.0f}",
             f"${metrics['final_value'] - metrics['benchmark_final']:+,.0f}"],
            ['Total Trades', f"{metrics['trade_count']}", 'N/A', 'N/A'],
            ['Total Costs', f"${metrics['total_costs']:,.0f}", 'N/A', 'N/A'],
            ['Bond Trap Days', f"{metrics['bond_trap_days']}", 'N/A', f"{metrics['bond_trap_pct']:.1%}"],
        ]

        # Plot table
        table = ax6.table(
            cellText=metrics_data,
            loc='center',
            cellLoc='center',
            colWidths=[0.3, 0.2, 0.2, 0.2]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        # Style header row
        for j in range(4):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        # Color code the difference column
        for i in range(2, len(metrics_data)):
            try:
                diff_text = metrics_data[i][3]
                if diff_text.startswith('+'):
                    table[(i, 3)].set_facecolor('#C6EFCE')  # Light green
                elif diff_text.startswith('-') and diff_text != '─' * 15:
                    table[(i, 3)].set_facecolor('#FFC7CE')  # Light red
            except:
                pass

        ax6.set_title('Backtest Performance Summary', fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        return fig1, fig2, fig3

    def print_summary(self):
        """Print text summary of backtest results."""
        if self.results is None:
            raise RuntimeError("Must run backtest before printing summary")

        metrics = self.results['metrics']

        print("\n" + "=" * 70)
        print(f"{'BACKTEST RESULTS SUMMARY':^70}")
        print("=" * 70)

        print(f"\n{'RETURNS':^70}")
        print("-" * 70)
        print(f"  {'Metric':<25} {'Strategy':>15} {'Benchmark':>15} {'Diff':>12}")
        print(f"  {'-' * 25} {'-' * 15} {'-' * 15} {'-' * 12}")
        print(
            f"  {'Total Return':<25} {metrics['total_return']:>15.1%} {metrics['benchmark_return']:>15.1%} {metrics['excess_return']:>+12.1%}")
        print(
            f"  {'CAGR':<25} {metrics['cagr']:>15.1%} {metrics['benchmark_cagr']:>15.1%} {metrics['excess_cagr']:>+12.1%}")

        print(f"\n{'RISK METRICS':^70}")
        print("-" * 70)
        print(f"  {'Volatility':<25} {metrics['volatility']:>15.1%} {metrics['benchmark_vol']:>15.1%}")
        print(f"  {'Max Drawdown':<25} {metrics['max_drawdown']:>15.1%} {metrics['benchmark_max_dd']:>15.1%}")
        print(f"  {'Avg DD Duration':<25} {metrics['avg_dd_duration']:>15.0f} days")

        print(f"\n{'RISK-ADJUSTED RETURNS':^70}")
        print("-" * 70)
        print(f"  {'Sharpe Ratio':<25} {metrics['sharpe']:>15.2f} {metrics['benchmark_sharpe']:>15.2f}")
        print(f"  {'Sortino Ratio':<25} {metrics['sortino']:>15.2f}")
        print(f"  {'Calmar Ratio':<25} {metrics['calmar']:>15.2f}")
        print(f"  {'Information Ratio':<25} {metrics['info_ratio']:>15.2f}")

        print(f"\n{'PORTFOLIO ANALYTICS':^70}")
        print("-" * 70)
        print(f"  {'Alpha (Annual)':<25} {metrics['alpha']:>15.2%}")
        print(f"  {'Beta':<25} {metrics['beta']:>15.2f}")
        print(f"  {'Win Rate':<25} {metrics['win_rate']:>15.1%}")

        print(f"\n{'TRADING STATISTICS':^70}")
        print("-" * 70)
        print(f"  {'Total Trades':<25} {metrics['trade_count']:>15}")
        print(f"  {'Avg Trades/Year':<25} {metrics['avg_annual_trades']:>15.1f}")
        print(f"  {'Total Costs':<25} ${metrics['total_costs']:>14,.0f}")

        print(f"\n{'BOND TRAP ANALYSIS':^70}")
        print("-" * 70)
        print(f"  {'Bond Trap Days':<25} {metrics['bond_trap_days']:>15}")
        print(f"  {'Bond Trap %':<25} {metrics['bond_trap_pct']:>15.1%}")

        print(f"\n{'FINAL VALUES':^70}")
        print("-" * 70)
        print(f"  {'Strategy':<25} ${metrics['final_value']:>14,.0f}")
        print(f"  {'Benchmark':<25} ${metrics['benchmark_final']:>14,.0f}")
        print(f"  {'Outperformance':<25} ${metrics['final_value'] - metrics['benchmark_final']:>+14,.0f}")

        print("=" * 70)

        # Verdict
        if metrics['sharpe'] > metrics['benchmark_sharpe'] and metrics['cagr'] > metrics['benchmark_cagr']:
            print("\n✅ STRATEGY BEATS BENCHMARK on both return and risk-adjusted basis!")
        elif metrics['sharpe'] > metrics['benchmark_sharpe']:
            print("\n⚠️ Strategy has better risk-adjusted returns but lower absolute returns")
        elif metrics['cagr'] > metrics['benchmark_cagr']:
            print("\n⚠️ Strategy has higher returns but worse risk-adjusted performance")
        else:
            print("\n❌ Strategy underperforms benchmark")

        print()


# ==============================================================================
# SECTION 10: MAIN APPLICATION CLASS
# ==============================================================================

class RegimeAdaptivePortfolio:
    """
    Main application class that orchestrates all components.

    Usage:
        app = RegimeAdaptivePortfolio()
        app.initialize()
        app.train()
        app.backtest()
        app.get_current_signal()
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()

        # Components
        self.data_manager = DataManager(self.config)
        self.feature_engine = FeatureEngine(self.config)
        self.target_generator = TargetGenerator(self.config)
        self.regime_classifier = RegimeClassifier(self.config)
        self.optimizer = CorrelationAwareOptimizer(self.config)
        self.monte_carlo = MonteCarloEngine(self.config)
        self.backtester = BacktestEngine(self.config)

        # Data
        self.prices: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.target: Optional[pd.Series] = None

        # State
        self._is_initialized = False
        self._is_trained = False
        self._backtest_results: Optional[Dict] = None

    def initialize(self) -> 'RegimeAdaptivePortfolio':
        """Load data and build features."""
        print("\n" + "=" * 70)
        print(f"{'INITIALIZING REGIME-ADAPTIVE PORTFOLIO SYSTEM':^70}")
        print("=" * 70)

        # Load data
        print("\n[1/3] Loading market data...")
        self.prices = self.data_manager.load_market_data(
            self.config.ALL_DATA_TICKERS,
            self.config.TRAIN_START
        )
        print(f"      Loaded {len(self.prices)} days of data for {len(self.prices.columns)} assets")
        print(f"      Date range: {self.prices.index[0].date()} to {self.prices.index[-1].date()}")

        # Compute returns
        print("\n[2/3] Computing returns...")
        self.returns = self.data_manager.compute_returns(self.prices, log_returns=True)

        # Build features
        print("\n[3/3] Building features...")
        self.features = self.feature_engine.build_features(self.prices)
        print(f"      Created {len(self.features.columns)} features")

        # Create target
        print("\n[4/4] Creating volatility-adjusted target...")
        self.target, thresholds = self.target_generator.create_regime_target(self.prices['SPY'])

        # Analyze target distribution
        target_dist = self._analyze_target_distribution()
        print(f"      Target distribution (full period):")
        print(f"        Bear:    {target_dist['bear_pct']:.1%}")
        print(f"        Neutral: {target_dist['neutral_pct']:.1%}")
        print(f"        Bull:    {target_dist['bull_pct']:.1%}")

        self._is_initialized = True
        print("\n✅ Initialization complete!")

        return self

    def _analyze_target_distribution(self) -> Dict:
        """Analyze target variable distribution."""
        target_clean = self.target.dropna()
        counts = target_clean.value_counts().sort_index()
        total = len(target_clean)

        return {
            'bear': counts.get(0, 0),
            'neutral': counts.get(1, 0),
            'bull': counts.get(2, 0),
            'total': total,
            'bear_pct': counts.get(0, 0) / total if total > 0 else 0,
            'neutral_pct': counts.get(1, 0) / total if total > 0 else 0,
            'bull_pct': counts.get(2, 0) / total if total > 0 else 0,
        }

    def train(self) -> 'RegimeAdaptivePortfolio':
        """Train the ML model and optimize portfolios."""
        if not self._is_initialized:
            raise RuntimeError("Must call initialize() before train()")

        print("\n" + "=" * 70)
        print(f"{'TRAINING ML MODEL':^70}")
        print("=" * 70)

        # Prepare ML features
        ml_features = self.feature_engine.prepare_ml_data(self.features)

        # Split data
        train_mask = ml_features.index < self.config.TEST_START
        test_mask = ml_features.index >= self.config.TEST_START

        X_train = ml_features[train_mask]
        y_train = self.target[train_mask]
        X_test = ml_features[test_mask]
        y_test = self.target[test_mask]

        print(f"\n[1/3] Data split:")
        print(f"      Training: {len(X_train)} samples ({X_train.index[0].date()} to {X_train.index[-1].date()})")
        print(f"      Testing:  {len(X_test)} samples ({X_test.index[0].date()} to {X_test.index[-1].date()})")

        # Train model
        print("\n[2/3] Training ensemble model...")
        self.regime_classifier.fit(X_train, y_train, X_val=X_test, y_val=y_test)

        # Print training metrics
        train_metrics = self.regime_classifier.get_training_metrics()
        print(f"\n      Training Accuracy:   {train_metrics.get('train_accuracy', 0):.1%}")
        if 'val_accuracy' in train_metrics:
            print(f"      Validation Accuracy: {train_metrics['val_accuracy']:.1%}")
            print(f"      Overfit Gap:         {train_metrics.get('overfit_gap', 0):.1%}")

        # Evaluate on test set
        print("\n[3/3] Evaluating on test set...")
        test_metrics = self.regime_classifier.evaluate(X_test, y_test)
        print(f"      Test Accuracy: {test_metrics['accuracy']:.1%}")
        print(f"      Test F1 Macro: {test_metrics['f1_macro']:.3f}")

        # Print confusion matrix
        print(f"\n      Confusion Matrix:")
        cm = test_metrics['confusion_matrix']
        print(f"                    Predicted")
        print(f"                 Bear  Neut  Bull")
        print(f"      Actual Bear  {cm[0, 0]:4d}  {cm[0, 1]:4d}  {cm[0, 2]:4d}")
        print(f"            Neut  {cm[1, 0]:4d}  {cm[1, 1]:4d}  {cm[1, 2]:4d}")
        print(f"            Bull  {cm[2, 0]:4d}  {cm[2, 1]:4d}  {cm[2, 2]:4d}")

        self._is_trained = True
        print("\n✅ Training complete!")

        return self

    def backtest(self) -> Dict[str, Any]:
        """Run full backtest."""
        if not self._is_trained:
            raise RuntimeError("Must call train() before backtest()")

        print("\n" + "=" * 70)
        print(f"{'RUNNING BACKTEST':^70}")
        print("=" * 70)

        self._backtest_results = self.backtester.run_backtest(
            prices=self.prices,
            features=self.features,
            returns=self.returns,
            regime_classifier=self.regime_classifier,
            optimizer=self.optimizer,
            benchmark_ticker='SPY'
        )

        # Print summary
        self.backtester.print_summary()

        return self._backtest_results

    def get_current_signal(self) -> Dict[str, Any]:
        """Get current trading signal with Trend Veto applied."""
        if not self._is_trained:
            raise RuntimeError("Must call train() before get_current_signal()")

        # Get latest features
        ml_features = self.feature_engine.prepare_ml_data(self.features)
        latest_features = ml_features.iloc[[-1]]
        latest_date = latest_features.index[0]

        # Predict regime from ML
        regime, regime_probs = self.regime_classifier.predict_regime(latest_features)

        # ---------------------------------------------------------
        # NEW LOGIC: APPLY TREND VETO (Force Bull in Uptrend)
        # ---------------------------------------------------------
        spy_price = self.prices['SPY'].iloc[-1]
        sma_200 = self.prices['SPY'].rolling(200).mean().iloc[-1]
        above_sma = spy_price > sma_200

        if above_sma:
            # 🚀 UPTREND DETECTED: OVERRIDE DEFENSIVE SIGNALS
            bear_prob = regime_probs.get(MarketRegime.BEAR, 0.0)
            neutral_prob = regime_probs.get(MarketRegime.NEUTRAL, 0.0)

            # 1. Kill the Bear
            regime_probs[MarketRegime.BEAR] = 0.0
            regime_probs[MarketRegime.BULL] += bear_prob

            # 2. Convert Neutral to Bull (Get off the fence)
            shift = neutral_prob * 0.5
            regime_probs[MarketRegime.NEUTRAL] -= shift
            regime_probs[MarketRegime.BULL] += shift

            # 3. Recalculate dominant regime
            max_prob = max(regime_probs.values())
            for r, p in regime_probs.items():
                if p == max_prob:
                    regime = r
                    break
        else:
            # 📉 DOWNTREND: Boost Bear signal slightly for safety
            regime_probs[MarketRegime.BEAR] = min(1.0, regime_probs.get(MarketRegime.BEAR, 0.0) + 0.10)
            regime_probs[MarketRegime.BULL] = max(0.0, regime_probs.get(MarketRegime.BULL, 0.0) - 0.10)

            # Renormalize
            total = sum(regime_probs.values())
            if total > 0:
                regime_probs = {k: v / total for k, v in regime_probs.items()}
        # ---------------------------------------------------------

        # Detect correlation regime
        corr_regime, corr_value = self.optimizer.correlation_detector.detect_regime(
            self.prices, latest_date
        )

        # Force update optimizer's internal state
        self.optimizer._current_correlation_regime = corr_regime
        self.optimizer._current_correlation = corr_value

        # Optimize and get blended weights
        # Note: optimize_all_regimes uses the constraints we updated in Step 1
        self.optimizer.optimize_all_regimes(self.returns, self.prices, latest_date)
        blended_weights = self.optimizer.get_blended_weights(regime_probs)

        # Determine bond trap status
        is_bond_trap = (corr_regime == CorrelationRegime.POSITIVE)

        # Generate signal
        signal = {
            'date': latest_date.strftime('%Y-%m-%d'),
            'market_regime': regime.value,
            'regime_probabilities': {k.value: v for k, v in regime_probs.items()},
            'correlation_regime': corr_regime.value,
            'correlation_value': corr_value,
            'is_bond_trap': is_bond_trap,
            'spy_price': spy_price,
            'sma_200': sma_200,
            'above_sma': above_sma,
            'recommended_weights': blended_weights,
            'confidence': max(regime_probs.values())
        }

        return signal

    def print_current_signal(self):
        """Print formatted current signal."""
        signal = self.get_current_signal()

        print("\n" + "=" * 70)
        print(f"{'CURRENT TRADING SIGNAL':^70}")
        print(f"{'Date: ' + signal['date']:^70}")
        print("=" * 70)

        # Regime
        print(f"\n{'MARKET REGIME ANALYSIS':^70}")
        print("-" * 70)
        print(f"  Predicted Regime:    {signal['market_regime'].upper()}")
        print(f"  Confidence:          {signal['confidence']:.1%}")
        print(f"  Regime Probabilities:")
        for regime, prob in signal['regime_probabilities'].items():
            bar = '█' * int(prob * 30)
            print(f"    {regime:>8}: {prob:>6.1%} {bar}")

        # ════════════════════════════════════════════════════════════════
        # CORRELATION REGIME SECTION - REPLACE THIS PART
        # ════════════════════════════════════════════════════════════════
        print(f"\n{'CORRELATION REGIME':^70}")
        print("-" * 70)
        print(f"  Stock-Bond Correlation: {signal['correlation_value']:.2f}")
        print(f"  Correlation Regime:     {signal['correlation_regime'].upper()}")
        print(f"  Positive Threshold:     {self.config.CORRELATION_POSITIVE_THRESHOLD}")
        print(f"  Negative Threshold:     {self.config.CORRELATION_NEGATIVE_THRESHOLD}")

        # FIXED: Correct logic for bond trap warning
        if signal['is_bond_trap']:
            print(
                f"  ⚠️  BOND TRAP WARNING: Correlation ({signal['correlation_value']:.2f}) >= {self.config.CORRELATION_POSITIVE_THRESHOLD}!")
            print(f"      Bonds are moving WITH stocks - reduced bond allocation!")
        elif signal['correlation_regime'] == 'negative':
            print(f"  ✅ Bonds expected to provide hedging benefit")
        else:
            print(f"  ⚠️  Correlation is NEUTRAL - bonds may have limited hedging")

        # Trend
        print(f"\n{'TREND ANALYSIS':^70}")
        print("-" * 70)
        print(f"  SPY Price:    ${signal['spy_price']:.2f}")
        print(f"  200-Day SMA:  ${signal['sma_200']:.2f}")
        if signal['above_sma']:
            print(f"  ✅ SPY is ABOVE 200 SMA (Uptrend)")
        else:
            print(f"  ⚠️  SPY is BELOW 200 SMA (Downtrend)")

        # Recommended allocation
        print(f"\n{'RECOMMENDED ALLOCATION':^70}")
        print("-" * 70)
        print(f"  {'Asset':<10} {'Weight':>10} {'$100k':>12}")
        print(f"  {'-' * 10} {'-' * 10} {'-' * 12}")

        for ticker, weight in sorted(signal['recommended_weights'].items(), key=lambda x: -x[1]):
            if weight > 0.01:
                print(f"  {ticker:<10} {weight:>10.1%} ${100000 * weight:>11,.0f}")

        print("=" * 70)

    def run_monte_carlo(self, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation on the STRATEGY itself (Dynamic), not just assets.
        This captures the 'Alpha' of your active trading rules.
        """
        print("\n" + "=" * 70)
        print(f"{'MONTE CARLO SIMULATION (STRATEGY-AWARE)':^70}")
        print("=" * 70)

        # 1. CHECK FOR BACKTEST RESULTS
        # We need the backtest history to know how the "AI" actually behaves.
        if self._backtest_results is None:
            print("⚠️ No backtest found! Running backtest now to generate strategy history...")
            self.backtest()

        # 2. EXTRACT STRATEGY RETURNS
        # We extract the daily % change of your ACTUAL backtest equity curve.
        # This series includes all your smart decisions (going to cash, buying dips, etc.)
        portfolio_values = pd.Series(
            self._backtest_results['portfolio_values'],
            index=self._backtest_results['dates']
        )

        # Calculate daily returns of the strategy
        strategy_daily_returns = portfolio_values.pct_change().fillna(0)

        # Create a DataFrame wrapper for the engine
        # We trick the engine into thinking "My_Strategy" is a tradeable stock ticker
        strategy_df = pd.DataFrame({'My_Strategy': strategy_daily_returns})

        # 3. CONFIGURE SIMULATION
        # We tell the engine: "Simulate holding 100% of 'My_Strategy'"
        strategy_weights = {'My_Strategy': 1.0}

        # 4. RUN SIMULATION
        # Note: We pass strategy_df instead of self.returns
        # We use the LAST value of your portfolio as the starting point
        current_equity = portfolio_values.iloc[-1]

        results = self.monte_carlo.simulate_portfolio(
            strategy_df,
            strategy_weights,
            initial_value=current_equity
        )

        print(f"\n  Simulations:      {self.config.MC_SIMULATIONS:,}")
        print(f"  Horizon:          {self.config.MC_HORIZON_YEARS} years")
        print(f"  Input Source:     Dynamic Strategy History (Alpha Included)")
        print(f"\n  Expected CAGR:    {results['mean_cagr']:.1%}")
        print(f"  Volatility:       {results['annual_volatility']:.1%}")
        print(f"  Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
        print(f"\n  95% CI (Value):   ${results['ci_95_lower']:,.0f} - ${results['ci_95_upper']:,.0f}")
        print(f"  Prob of Loss:     {results['prob_loss']:.1%}")
        print(f"  Mean Max DD:      {results['mean_max_drawdown']:.1%}")

        print("=" * 70)

        return results
    def plot_backtest(self) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
        """Plot backtest results."""
        if self._backtest_results is None:
            raise RuntimeError("Must call backtest() before plot_backtest()")

        return self.backtester.plot_results()

    def plot_monte_carlo(self, results: Dict) -> Tuple[plt.Figure, plt.Figure]:
        """Plot Monte Carlo results."""
        return self.monte_carlo.plot_results(results)


# ==============================================================================
# SECTION 11: ENTRY POINT (FOR TESTING WITHOUT STREAMLIT)
# ==============================================================================

def main():
    """Main entry point for testing."""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + f"{'REGIME-ADAPTIVE PORTFOLIO SYSTEM v4.0':^68}" + "█")
    print("█" + " " * 68 + "█")
    print("█" + f"{'With Correlation-Aware Bond Trap Prevention':^68}" + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    # Verify config
    config = SystemConfig()
    print("\n" + "=" * 50)
    print("ASSET UNIVERSE VERIFICATION")
    print("=" * 50)
    print(f"Tradeable: {config.ALL_TRADEABLE_TICKERS}")
    print(f"Data:      {config.ALL_DATA_TICKERS}")
    print(f"VIX in tradeable? {config.VOLATILITY_INDEX in config.ALL_TRADEABLE_TICKERS}")
    print("=" * 50)

    # Initialize
    app = RegimeAdaptivePortfolio()

    # Run pipeline
    app.initialize()
    app.train()
    results = app.backtest()

    # Current signal
    app.print_current_signal()

    # Monte Carlo
    mc_results = app.run_monte_carlo()

    # Plot (if not in headless mode)
    try:
        fig1, fig2, fig3 = app.plot_backtest()
        plt.show()

        fig4, fig5 = app.plot_monte_carlo(mc_results)
        plt.show()
    except:
        print("\n(Plotting skipped - headless mode)")

    return app, results, mc_results

if __name__ == "__main__":
    app, results, mc_results = main()