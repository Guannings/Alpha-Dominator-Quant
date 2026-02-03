"""
================================================================================
REGIME-ADAPTIVE PORTFOLIO MANAGEMENT SYSTEM v2.0
================================================================================
Architecture: Parallel Optimization + ML Blending

Pipeline:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  PHASE 1: INITIALIZATION (Monthly)                                      │
    │  ├─► Optimize Bull Portfolio (aggressive)                               │
    │  ├─► Optimize Neutral Portfolio (balanced)                              │
    │  ├─► Optimize Bear Portfolio (defensive)                                │
    │  └─► Train ML Regime Classifier                                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  PHASE 2: DAILY SIGNAL (Real-time)                                      │
    │  ├─► Compute Features                                                   │
    │  ├─► ML Predicts Regime Probabilities                                   │
    │  ├─► Blend Portfolios by Probability                                    │
    │  └─► Apply Transaction Filters                                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  PHASE 3: RISK MANAGEMENT                                               │
    │  ├─► Monte Carlo Forward Simulation                                     │
    │  ├─► Drawdown Monitoring                                                │
    │  └─► Position Limits                                                    │
    └─────────────────────────────────────────────────────────────────────────┘

Transaction Costs Modeled:
    - Trading Commission: 5 bps per trade
    - Slippage: 2 bps per trade
    - Management Fee: 10 bps annual
    - Rebalancing Threshold: 3% (to avoid over-trading)

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
    confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import time
import warnings

warnings.filterwarnings('ignore')

# Optional advanced ML libraries
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


@dataclass
class SystemConfig:
    """
    Central configuration for the entire system.
    All parameters in one place for easy tuning.
    """

    # ──────────────────────────────────────────────────────────────────────
    # DATA SETTINGS
    # ──────────────────────────────────────────────────────────────────────
    TRAIN_START: str = "2006-01-01"  # Start of training data
    TEST_START: str = "2018-01-01"  # Start of out-of-sample test

    # ──────────────────────────────────────────────────────────────────────
    # ASSET UNIVERSE
    # ──────────────────────────────────────────────────────────────────────
    # Core assets for portfolio optimization
    CORE_EQUITIES: Tuple[str, ...] = ("SPY", "QQQ")
    INTL_EQUITIES: Tuple[str, ...] = ("VEA",)  # Developed markets
    BONDS: Tuple[str, ...] = ("TLT", "IEF")  # Long & intermediate treasury
    ALTERNATIVES: Tuple[str, ...] = ("GLD",)  # Gold

    # Supporting data (not traded directly)
    VOLATILITY_INDEX: str = "^VIX"

    # ──────────────────────────────────────────────────────────────────────
    # TRANSACTION COSTS (Realistic Institutional)
    # ──────────────────────────────────────────────────────────────────────
    TRADING_COST_BPS: float = 5.0  # 5 basis points per trade
    SLIPPAGE_BPS: float = 2.0  # 2 basis points slippage
    MANAGEMENT_FEE_ANNUAL_BPS: float = 10.0  # 10 bps annual
    REBALANCE_THRESHOLD: float = 0.03  # Only rebalance if weight changes > 3%

    # ──────────────────────────────────────────────────────────────────────
    # RISK PARAMETERS
    # ──────────────────────────────────────────────────────────────────────
    RISK_FREE_RATE: float = 0.045  # Current T-bill rate (~4.5%)

    # Regime-specific volatility targets
    BULL_TARGET_VOL: float = 0.16  # Aggressive
    NEUTRAL_TARGET_VOL: float = 0.12  # Balanced
    BEAR_TARGET_VOL: float = 0.07  # Defensive

    MAX_DRAWDOWN_LIMIT: float = 0.20  # Emergency exit trigger

    # ──────────────────────────────────────────────────────────────────────
    # ML SETTINGS
    # ──────────────────────────────────────────────────────────────────────
    PREDICTION_HORIZON: int = 21  # Days ahead to predict (1 month)

    # Regime classification thresholds (for target creation)
    BULL_THRESHOLD: float = 0.02  # > 2% forward return = bull
    BEAR_THRESHOLD: float = -0.02  # < -2% forward return = bear

    # ──────────────────────────────────────────────────────────────────────
    # MONTE CARLO SETTINGS
    # ──────────────────────────────────────────────────────────────────────
    MC_SIMULATIONS: int = 50000
    MC_HORIZON_YEARS: int = 5

    # ──────────────────────────────────────────────────────────────────────
    # COMPUTED PROPERTIES
    # ──────────────────────────────────────────────────────────────────────
    @property
    def TOTAL_TRANSACTION_COST(self) -> float:
        """Total one-way transaction cost as decimal."""
        return (self.TRADING_COST_BPS + self.SLIPPAGE_BPS) / 10000

    @property
    def DAILY_MANAGEMENT_FEE(self) -> float:
        """Daily management fee as decimal."""
        return (self.MANAGEMENT_FEE_ANNUAL_BPS / 10000) / 252

    @property
    def ALL_TRADEABLE_TICKERS(self) -> List[str]:
        """All tickers that can be held in portfolio."""
        return list(self.CORE_EQUITIES + self.INTL_EQUITIES +
                    self.BONDS + self.ALTERNATIVES)

    @property
    def ALL_DATA_TICKERS(self) -> List[str]:
        """All tickers needed for data download."""
        return self.ALL_TRADEABLE_TICKERS + [self.VOLATILITY_INDEX]


# ==============================================================================
# SECTION 2: DATA MANAGEMENT
# ==============================================================================

class DataManager:
    """
    Handles all data acquisition, validation, and preprocessing.
    Implements caching and retry logic for robustness.
    """

    def __init__(self, config: SystemConfig):
        self.config = config

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_market_data(tickers: List[str], start_date: str) -> pd.DataFrame:
        """
        Load market data with retry logic and validation.

        Args:
            tickers: List of ticker symbols
            start_date: Start date string (YYYY-MM-DD)

        Returns:
            DataFrame with adjusted close prices
        """
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

                # Handle response format
                if isinstance(data.columns, pd.MultiIndex):
                    prices = data['Close'].copy()
                else:
                    prices = pd.DataFrame(data['Close'])
                    prices.columns = [tickers[0]] if len(tickers) == 1 else tickers

                # Remove timezone
                if prices.index.tz is not None:
                    prices.index = prices.index.tz_localize(None)

                # Forward fill small gaps (weekends, holidays)
                prices = prices.ffill(limit=5)

                # Drop any remaining NaN rows
                prices = prices.dropna()

                # Validate sufficient data
                if len(prices) < 252:
                    raise ValueError(f"Insufficient data: {len(prices)} days")

                return prices

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise RuntimeError(f"Data load failed after {max_retries} attempts: {e}")

    @staticmethod
    def compute_returns(prices: pd.DataFrame, log_returns: bool = True) -> pd.DataFrame:
        """
        Compute returns from prices.

        Args:
            prices: Price DataFrame
            log_returns: If True, compute log returns (better for statistics)

        Returns:
            Returns DataFrame
        """
        if log_returns:
            return np.log(prices / prices.shift(1)).dropna()
        else:
            return prices.pct_change().dropna()


# ==============================================================================
# SECTION 3: FEATURE ENGINEERING
# ==============================================================================

class FeatureEngine:
    """
    Comprehensive feature engineering for ML regime detection.
    Creates technical indicators, volatility measures, and cross-asset signals.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.scaler = RobustScaler()
        self._is_fitted = False

    def build_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Build complete feature set from price data.

        Args:
            prices: DataFrame with asset prices (must include SPY and ^VIX)

        Returns:
            DataFrame with all engineered features
        """
        features = pd.DataFrame(index=prices.index)

        spy = prices['SPY']
        vix = prices['^VIX'] if '^VIX' in prices.columns else pd.Series(20, index=prices.index)
        returns = spy.pct_change()

        # ══════════════════════════════════════════════════════════════════
        # TREND FEATURES
        # ══════════════════════════════════════════════════════════════════
        for window in [20, 50, 100, 200]:
            sma = spy.rolling(window).mean()
            features[f'dist_sma_{window}'] = (spy - sma) / sma

        # Trend strength (ADX proxy using price momentum)
        features['trend_strength'] = abs(features['dist_sma_20']).rolling(10).mean()

        # SMA crossover signals
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

        # Momentum acceleration
        features['momentum_accel'] = (
                features['momentum_21d'] - features['momentum_21d'].shift(21)
        )

        # ══════════════════════════════════════════════════════════════════
        # VOLATILITY FEATURES
        # ══════════════════════════════════════════════════════════════════
        for window in [10, 21, 63]:
            features[f'volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(252)

        # Volatility regime (short vs long term)
        features['vol_regime'] = features['volatility_21d'] / features['volatility_63d']

        # Volatility percentile (historical ranking)
        features['vol_percentile'] = features['volatility_21d'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )

        # ══════════════════════════════════════════════════════════════════
        # VIX FEATURES (Fear Gauge)
        # ══════════════════════════════════════════════════════════════════
        features['vix'] = vix
        features['vix_sma_ratio'] = vix / vix.rolling(21).mean()
        features['vix_percentile'] = vix.rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )

        # VIX term structure proxy (current vs moving average)
        features['vix_term'] = vix / vix.rolling(63).mean()

        # VIX spike detection
        vix_upper = vix.rolling(21).mean() + 2 * vix.rolling(21).std()
        features['vix_spike'] = (vix > vix_upper).astype(int)

        # ══════════════════════════════════════════════════════════════════
        # RSI (Relative Strength Index)
        # ══════════════════════════════════════════════════════════════════
        for period in [7, 14, 21]:
            delta = returns
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)

            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()

            rs = avg_gain / avg_loss.replace(0, np.inf)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # RSI divergence
        features['rsi_divergence'] = features['rsi_14'] - features['rsi_14'].rolling(21).mean()

        # ══════════════════════════════════════════════════════════════════
        # MACD
        # ════════════════════════════════════���═════════════════════════════
        ema_12 = spy.ewm(span=12, adjust=False).mean()
        ema_26 = spy.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()

        features['macd'] = macd / spy * 100  # Normalize
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
        # DRAWDOWN FEATURES
        # ══════════════════════════════════════════════════════════════════
        rolling_max = spy.rolling(252, min_periods=1).max()
        features['drawdown'] = (spy - rolling_max) / rolling_max
        features['drawdown_duration'] = self._compute_drawdown_duration(spy, rolling_max)

        # ══════════════════════════════════════════════════════════════════
        # CROSS-ASSET FEATURES
        # ══════════════════════════════════════════════════════════════════
        if 'TLT' in prices.columns:
            # Stock-bond correlation (risk-on/risk-off indicator)
            features['spy_tlt_corr'] = (
                prices['SPY'].pct_change().rolling(21)
                .corr(prices['TLT'].pct_change())
            )
            features['spy_tlt_ratio'] = prices['SPY'] / prices['TLT']
            features['spy_tlt_ratio_zscore'] = (
                    (features['spy_tlt_ratio'] - features['spy_tlt_ratio'].rolling(63).mean()) /
                    features['spy_tlt_ratio'].rolling(63).std()
            )

        if 'GLD' in prices.columns:
            features['spy_gld_ratio'] = prices['SPY'] / prices['GLD']

        if 'QQQ' in prices.columns:
            # Growth vs Value proxy
            features['qqq_spy_ratio'] = prices['QQQ'] / prices['SPY']
            features['qqq_spy_momentum'] = features['qqq_spy_ratio'].pct_change(21)

        # ══════════════════════════════════════════════════════════════════
        # CLEAN UP
        # ══════════════════════════════════════════════════════════════════
        # Drop intermediate columns, keep only final features
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
        """Return list of features to use for ML model."""
        return [
            # Trend
            'dist_sma_50', 'dist_sma_200', 'sma_50_200_cross',
            # Momentum
            'momentum_21d', 'momentum_63d', 'momentum_accel',
            # Volatility
            'volatility_21d', 'vol_regime', 'vol_percentile',
            # VIX
            'vix', 'vix_percentile', 'vix_term', 'vix_spike',
            # Oscillators
            'rsi_14', 'rsi_divergence', 'macd_histogram', 'bb_position',
            # Drawdown
            'drawdown',
            # Cross-asset
            'spy_tlt_corr', 'spy_tlt_ratio_zscore'
        ]

    def prepare_ml_data(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model (select and handle missing).

        Args:
            features: Full feature DataFrame

        Returns:
            Cleaned feature DataFrame for ML
        """
        ml_cols = [c for c in self.get_ml_feature_columns() if c in features.columns]
        ml_features = features[ml_cols].copy()

        return ml_features

    def fit_scaler(self, features: pd.DataFrame) -> 'FeatureEngine':
        """Fit the scaler on training data."""
        clean_features = features.dropna()
        self.scaler.fit(clean_features)
        self._is_fitted = True
        return self

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scaler."""
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted before transform")
        return self.scaler.transform(features)


# ==============================================================================
# SECTION 4: ML REGIME CLASSIFIER
# ==============================================================================

class RegimeClassifier:
    """
    Ensemble ML model for market regime classification.
    Outputs probabilities for Bull/Neutral/Bear regimes.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.model = None
        self.scaler = RobustScaler()
        self._is_fitted = False
        self._feature_importance = None
        self._class_labels = [MarketRegime.BEAR, MarketRegime.NEUTRAL, MarketRegime.BULL]

    def _build_ensemble(self) -> StackingClassifier:
        """Build stacking ensemble classifier."""

        # Base models with different strengths
        base_models = [
            ('gb', GradientBoostingClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=30,
                random_state=42
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=20,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                alpha=0.01,
                learning_rate='adaptive',
                max_iter=500,
                early_stopping=True,
                random_state=42
            ))
        ]

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            base_models.append(('xgb', XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss',
                verbosity=0
            )))

        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            base_models.append(('lgbm', LGBMClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )))

        # Calibrate base models for better probability estimates
        calibrated_models = [
            (name, CalibratedClassifierCV(model, method='isotonic', cv=3))
            for name, model in base_models
        ]

        # Meta-learner
        meta_learner = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )

        # Stacking ensemble
        ensemble = StackingClassifier(
            estimators=calibrated_models,
            final_estimator=meta_learner,
            cv=TimeSeriesSplit(n_splits=5),
            stack_method='predict_proba',
            passthrough=False,
            n_jobs=-1
        )

        return ensemble

    def create_regime_target(self, spy_prices: pd.Series) -> pd.Series:
        """
        Create regime labels based on forward returns.

        Args:
            spy_prices: SPY price series

        Returns:
            Series with regime labels (0=Bear, 1=Neutral, 2=Bull)
        """
        horizon = self.config.PREDICTION_HORIZON
        forward_returns = spy_prices.shift(-horizon) / spy_prices - 1

        target = pd.Series(index=spy_prices.index, dtype=float)

        # Classify based on forward return thresholds
        target[forward_returns <= self.config.BEAR_THRESHOLD] = 0  # Bear
        target[forward_returns >= self.config.BULL_THRESHOLD] = 2  # Bull
        target[(forward_returns > self.config.BEAR_THRESHOLD) &
               (forward_returns < self.config.BULL_THRESHOLD)] = 1  # Neutral

        return target

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RegimeClassifier':
        """
        Fit the ensemble model.

        Args:
            X: Feature DataFrame
            y: Target Series (0/1/2 for Bear/Neutral/Bull)

        Returns:
            self
        """
        # Handle missing values
        valid_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)

        # Build and fit ensemble
        self.model = self._build_ensemble()
        self.model.fit(X_scaled, y_clean.astype(int))

        # Compute feature importance (from gradient boosting component)
        self._compute_feature_importance(X.columns)

        self._is_fitted = True
        return self

    def _compute_feature_importance(self, feature_names: pd.Index):
        """Extract feature importance from base models."""
        importances = []

        for name, calibrated_model in self.model.estimators_:
            base_model = calibrated_model.estimator
            if hasattr(base_model, 'feature_importances_'):
                importances.append(base_model.feature_importances_)

        if importances:
            avg_importance = np.mean(importances, axis=0)
            self._feature_importance = pd.Series(
                avg_importance,
                index=feature_names
            ).sort_values(ascending=False)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict regime probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            Array of shape (n_samples, 3) with [P(Bear), P(Neutral), P(Bull)]
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # Handle missing values
        X_filled = X.fillna(X.mean())
        X_scaled = self.scaler.transform(X_filled)

        return self.model.predict_proba(X_scaled)

    def predict_regime(self, X: pd.DataFrame) -> Tuple[MarketRegime, Dict[str, float]]:
        """
        Predict current regime with probabilities.

        Args:
            X: Feature DataFrame (single row)

        Returns:
            Tuple of (predicted_regime, probability_dict)
        """
        proba = self.predict_proba(X)[0]

        prob_dict = {
            MarketRegime.BEAR: proba[0],
            MarketRegime.NEUTRAL: proba[1],
            MarketRegime.BULL: proba[2]
        }

        # Get regime with highest probability
        predicted_regime = max(prob_dict, key=prob_dict.get)

        return predicted_regime, prob_dict

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model performance.

        Args:
            X: Feature DataFrame
            y: True labels

        Returns:
            Dictionary of metrics
        """
        valid_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_mask]
        y_clean = y[valid_mask].astype(int)

        X_scaled = self.scaler.transform(X_clean)

        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)

        metrics = {
            'accuracy': accuracy_score(y_clean, y_pred),
            'precision_macro': precision_score(y_clean, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_clean, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_clean, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y_clean, y_pred),
            'classification_report': classification_report(y_clean, y_pred, output_dict=True)
        }

        # ROC AUC for multi-class
        try:
            metrics['roc_auc_ovr'] = roc_auc_score(y_clean, y_proba, multi_class='ovr')
        except:
            metrics['roc_auc_ovr'] = None

        return metrics

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict:
        """
        Time-series cross-validation.

        Args:
            X: Feature DataFrame
            y: Target Series
            n_splits: Number of CV folds

        Returns:
            Dictionary of CV scores
        """
        valid_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_mask]
        y_clean = y[valid_mask].astype(int)

        X_scaled = self.scaler.fit_transform(X_clean)

        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Use simpler model for CV (faster)
        cv_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )

        scores = {'accuracy': [], 'f1_macro': []}

        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]

            cv_model.fit(X_train, y_train)
            y_pred = cv_model.predict(X_val)

            scores['accuracy'].append(accuracy_score(y_val, y_pred))
            scores['f1_macro'].append(f1_score(y_val, y_pred, average='macro', zero_division=0))

        return {
            metric: {'mean': np.mean(vals), 'std': np.std(vals)}
            for metric, vals in scores.items()
        }

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Return feature importance rankings."""
        return self._feature_importance


# ==============================================================================
# SECTION 5: PORTFOLIO OPTIMIZER
# ==============================================================================

class PortfolioOptimizer:
    """
    Mean-variance portfolio optimization with regime-specific constraints.
    Creates separate optimal portfolios for each market regime.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.regime_portfolios: Dict[MarketRegime, Dict[str, float]] = {}
        self.regime_metrics: Dict[MarketRegime, Dict[str, float]] = {}

    def _get_regime_constraints(self, regime: MarketRegime) -> Dict:
        """Get optimization constraints for a specific regime."""

        constraints = {
            MarketRegime.BULL: {
                'target_vol': self.config.BULL_TARGET_VOL,
                'equity_min': 0.60,
                'equity_max': 0.90,
                'bond_min': 0.05,
                'bond_max': 0.25,
                'alt_min': 0.05,
                'alt_max': 0.20
            },
            MarketRegime.NEUTRAL: {
                'target_vol': self.config.NEUTRAL_TARGET_VOL,
                'equity_min': 0.40,
                'equity_max': 0.65,
                'bond_min': 0.20,
                'bond_max': 0.40,
                'alt_min': 0.10,
                'alt_max': 0.25
            },
            MarketRegime.BEAR: {
                'target_vol': self.config.BEAR_TARGET_VOL,
                'equity_min': 0.15,
                'equity_max': 0.35,
                'bond_min': 0.35,
                'bond_max': 0.55,
                'alt_min': 0.15,
                'alt_max': 0.35
            }
        }

        return constraints[regime]

    def optimize_for_regime(self, returns: pd.DataFrame,
                            regime: MarketRegime) -> Dict[str, float]:
        """
        Optimize portfolio for a specific regime.

        Args:
            returns: Asset returns DataFrame
            regime: Target regime

        Returns:
            Dictionary of {ticker: weight}
        """
        tickers = returns.columns.tolist()
        n_assets = len(tickers)

        # Statistics
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        # Get regime constraints
        rc = self._get_regime_constraints(regime)

        # Identify asset classes
        equity_tickers = list(self.config.CORE_EQUITIES + self.config.INTL_EQUITIES)
        bond_tickers = list(self.config.BONDS)
        alt_tickers = list(self.config.ALTERNATIVES)

        equity_idx = [i for i, t in enumerate(tickers) if t in equity_tickers]
        bond_idx = [i for i, t in enumerate(tickers) if t in bond_tickers]
        alt_idx = [i for i, t in enumerate(tickers) if t in alt_tickers]

        # Objective: Maximize Sharpe Ratio
        def neg_sharpe(weights):
            port_return = np.sum(mean_returns.values * weights)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
            if port_vol == 0:
                return 1e10
            sharpe = (port_return - self.config.RISK_FREE_RATE) / port_vol
            return -sharpe

        # Constraints
        constraints = [
            # Weights sum to 1
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},

            # Volatility constraint
            {'type': 'ineq', 'fun': lambda x: rc['target_vol'] -
                                              np.sqrt(np.dot(x.T, np.dot(cov_matrix.values, x)))},
        ]

        # Asset class constraints
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

        # Bounds: min 2% per asset for diversification
        bounds = tuple((0.02, 0.50) for _ in range(n_assets))

        # Initial guess
        init_weights = np.array([1 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            neg_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )

        if result.success:
            weights = result.x
        else:
            # Fallback to equal weight within constraints
            weights = init_weights

        # Normalize
        weights = weights / weights.sum()

        # Store results
        weight_dict = dict(zip(tickers, weights))
        self.regime_portfolios[regime] = weight_dict

        # Compute metrics
        port_return = np.sum(mean_returns.values * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
        sharpe = (port_return - self.config.RISK_FREE_RATE) / port_vol

        self.regime_metrics[regime] = {
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe
        }

        return weight_dict

    def optimize_all_regimes(self, returns: pd.DataFrame) -> Dict[MarketRegime, Dict[str, float]]:
        """
        Optimize portfolios for all regimes.

        Args:
            returns: Asset returns DataFrame

        Returns:
            Dictionary of {regime: {ticker: weight}}
        """
        for regime in MarketRegime:
            self.optimize_for_regime(returns, regime)

        return self.regime_portfolios

    def get_blended_weights(self, regime_probs: Dict[MarketRegime, float]) -> Dict[str, float]:
        """
        Blend regime portfolios based on probabilities.

        Args:
            regime_probs: Dictionary of {regime: probability}

        Returns:
            Blended portfolio weights
        """
        if not self.regime_portfolios:
            raise RuntimeError("Must optimize regime portfolios first")

        # Get all assets
        all_assets = set()
        for weights in self.regime_portfolios.values():
            all_assets.update(weights.keys())

        # Blend
        blended = {}
        for asset in all_assets:
            blended[asset] = sum(
                regime_probs.get(regime, 0) * self.regime_portfolios[regime].get(asset, 0)
                for regime in MarketRegime
            )

        # Normalize (should already sum to 1, but just in case)
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}

        return blended

    def print_regime_portfolios(self):
        """Print formatted regime portfolio summary."""
        print("\n" + "=" * 70)
        print(f"{'REGIME PORTFOLIO ALLOCATIONS':^70}")
        print("=" * 70)

        for regime in MarketRegime:
            if regime not in self.regime_portfolios:
                continue

            weights = self.regime_portfolios[regime]
            metrics = self.regime_metrics[regime]

            print(f"\n{'─' * 70}")
            print(f"  {regime.value.upper()} PORTFOLIO")
            print(f"  Expected Return: {metrics['expected_return']:.1%} | "
                  f"Volatility: {metrics['volatility']:.1%} | "
                  f"Sharpe: {metrics['sharpe_ratio']:.2f}")
            print(f"{'─' * 70}")
            print(f"  {'Ticker':<10} {'Weight':>10} {'$100k':>12}")
            print(f"  {'-' * 10} {'-' * 10} {'-' * 12}")

            for ticker, weight in sorted(weights.items(), key=lambda x: -x[1]):
                if weight > 0.001:
                    print(f"  {ticker:<10} {weight:>10.1%} {100000 * weight:>12,.0f}")

        print("=" * 70)


# ==============================================================================
# SECTION 6: MONTE CARLO SIMULATION
# ==============================================================================

class MonteCarloEngine:
    """
    Monte Carlo simulation for forward-looking risk analysis.
    Simulates both static portfolios and the dynamic blended strategy.
    """

    def __init__(self, config: SystemConfig):
        self.config = config

    def simulate_portfolio(self, returns: pd.DataFrame,
                           weights: Dict[str, float],
                           n_sims: Optional[int] = None,
                           horizon_years: Optional[int] = None,
                           initial_value: float = 100000) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for a portfolio.

        Args:
            returns: Asset returns DataFrame
            weights: Portfolio weights
            n_sims: Number of simulations
            horizon_years: Simulation horizon
            initial_value: Starting portfolio value

        Returns:
            Dictionary of simulation results
        """
        n_sims = n_sims or self.config.MC_SIMULATIONS
        horizon_years = horizon_years or self.config.MC_HORIZON_YEARS
        n_days = 252 * horizon_years

        # Portfolio statistics
        weight_arr = np.array([weights.get(t, 0) for t in returns.columns])
        port_returns = returns.values @ weight_arr

        port_mean = np.mean(port_returns)
        port_std = np.std(port_returns)

        # GBM simulation
        drift = port_mean - 0.5 * port_std ** 2

        np.random.seed(42)
        Z = np.random.standard_normal((n_days, n_sims))

        daily_returns = np.exp(drift + port_std * Z)

        # Price paths
        price_paths = np.zeros((n_days + 1, n_sims))
        price_paths[0] = initial_value

        for t in range(1, n_days + 1):
            price_paths[t] = price_paths[t - 1] * daily_returns[t - 1]

        # Metrics
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

        results = {
            'price_paths': price_paths,
            'ending_values': ending_values,
            'cagrs': cagrs,
            'max_drawdowns': max_drawdowns,

            # Summary statistics
            'mean_ending_value': np.mean(ending_values),
            'median_ending_value': np.median(ending_values),
            'mean_cagr': np.mean(cagrs),
            'median_cagr': np.median(cagrs),
            'std_cagr': np.std(cagrs),

            # Confidence intervals
            'ci_95_lower': np.percentile(ending_values, 2.5),
            'ci_95_upper': np.percentile(ending_values, 97.5),
            'ci_95_lower_cagr': np.percentile(cagrs, 2.5),
            'ci_95_upper_cagr': np.percentile(cagrs, 97.5),

            # Risk metrics
            'prob_loss': np.mean(ending_values < initial_value),
            'prob_double': np.mean(ending_values >= 2 * initial_value),
            'var_95': np.percentile(ending_values, 5),
            'cvar_95': np.mean(ending_values[ending_values <= np.percentile(ending_values, 5)]),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': np.min(max_drawdowns),

            # Performance
            'annual_volatility': port_std * np.sqrt(252),
            'sharpe_ratio': (np.mean(cagrs) - self.config.RISK_FREE_RATE) / np.std(cagrs) if np.std(cagrs) > 0 else 0
        }

        return results

    def simulate_all_regimes(self, returns: pd.DataFrame,
                             regime_portfolios: Dict[MarketRegime, Dict[str, float]]) -> Dict[MarketRegime, Dict]:
        """
        Run Monte Carlo for all regime portfolios.

        Args:
            returns: Asset returns
            regime_portfolios: Portfolio weights for each regime

        Returns:
            Dictionary of {regime: simulation_results}
        """
        results = {}

        for regime, weights in regime_portfolios.items():
            results[regime] = self.simulate_portfolio(returns, weights)

        return results

    def plot_simulation(self, results: Dict, title: str = "Monte Carlo Simulation",
                        n_paths: int = 200) -> Tuple[plt.Figure, plt.Figure]:
        """
        Create visualization of simulation results.

        Returns:
            Tuple of (paths_figure, distribution_figure)
        """
        # Figure 1: Price Paths
        fig1, ax1 = plt.subplots(figsize=(12, 7))

        price_paths = results['price_paths']
        ending_values = results['ending_values']
        n_sims = price_paths.shape[1]

        # Color by ending value
        cmap = plt.get_cmap('RdYlGn')
        norm = plt.Normalize(vmin=np.percentile(ending_values, 5),
                             vmax=np.percentile(ending_values, 95))

        # Sample paths to plot
        indices = np.random.choice(n_sims, min(n_paths, n_sims), replace=False)

        for i in indices:
            ax1.plot(price_paths[:, i], color=cmap(norm(ending_values[i])),
                     alpha=0.3, linewidth=0.5)

        # Mean and percentile bands
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
        sm.set_array([])
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

        # Reference lines
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
# SECTION 7: BACKTESTING ENGINE
# ==============================================================================

class BacktestEngine:
    """
    Comprehensive backtesting with realistic transaction costs.
    Tests the full blended strategy against benchmarks.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.results = None

    def run_backtest(self,
                     prices: pd.DataFrame,
                     features: pd.DataFrame,
                     regime_classifier: RegimeClassifier,
                     portfolio_optimizer: PortfolioOptimizer,
                     benchmark_ticker: str = 'SPY') -> Dict[str, Any]:
        """
        Run full backtest of the blended strategy.

        Args:
            prices: Asset prices
            features: Feature DataFrame
            regime_classifier: Fitted regime classifier
            portfolio_optimizer: Optimizer with regime portfolios
            benchmark_ticker: Benchmark for comparison

        Returns:
            Backtest results dictionary
        """
        # Prepare data
        feature_engine = FeatureEngine(self.config)
        ml_features = feature_engine.prepare_ml_data(features)

        # Get test period
        test_mask = ml_features.index >= self.config.TEST_START
        test_dates = ml_features[test_mask].index

        # Align prices
        common_dates = test_dates.intersection(prices.index)

        # Initialize tracking
        portfolio_values = [100000.0]
        benchmark_values = [100000.0]
        cash_values = [100000.0]

        current_weights: Dict[str, float] = {}
        weight_history = []
        regime_history = []
        prob_history = []

        # Daily rates
        daily_rf = self.config.RISK_FREE_RATE / 252
        daily_mgmt_fee = self.config.DAILY_MANAGEMENT_FEE

        # Asset returns
        asset_returns = prices.pct_change().fillna(0)

        print(f"\nRunning backtest from {common_dates[0].date()} to {common_dates[-1].date()}...")
        print(f"Total days: {len(common_dates)}")

        for i, date in enumerate(common_dates[1:], 1):
            prev_date = common_dates[i - 1]

            # ────────────────────────────────────────────────────────────
            # STEP 1: Get regime probabilities from ML
            # ────────────────────────────────────────────────────────────
            today_features = ml_features.loc[[prev_date]]  # Use previous day's features

            if today_features.isna().all().all():
                # Skip if no valid features
                regime_probs = {MarketRegime.NEUTRAL: 1.0, MarketRegime.BULL: 0.0, MarketRegime.BEAR: 0.0}
            else:
                _, regime_probs = regime_classifier.predict_regime(today_features)

            # ────────────────────────────────────────────────────────────
            # STEP 2: SMA Safety Override
            # ────────────────────────────────────────────────────────────
            spy_price = prices.loc[prev_date, 'SPY']
            sma_200 = prices['SPY'].loc[:prev_date].rolling(200).mean().iloc[-1]

            if spy_price < sma_200:
                # Force more bearish allocation when below 200 SMA
                regime_probs[MarketRegime.BEAR] = min(1.0, regime_probs[MarketRegime.BEAR] + 0.3)
                regime_probs[MarketRegime.BULL] = max(0.0, regime_probs[MarketRegime.BULL] - 0.2)
                regime_probs[MarketRegime.NEUTRAL] = max(0.0, regime_probs[MarketRegime.NEUTRAL] - 0.1)

                # Renormalize
                total_prob = sum(regime_probs.values())
                regime_probs = {k: v / total_prob for k, v in regime_probs.items()}

            # ────────────────────────────────────────────────────────────
            # STEP 3: Get blended target weights
            # ────────────────────────────────────────────────────────────
            target_weights = portfolio_optimizer.get_blended_weights(regime_probs)

            # ────────────────────────────────────────────────────────────
            # STEP 4: Apply transaction filter (avoid over-trading)
            # ────────────────────────────────────────────────────────────
            if current_weights:
                final_weights = self._apply_rebalance_filter(current_weights, target_weights)
            else:
                final_weights = target_weights

            # ───────────────────────────────────���────────────────────────
            # STEP 5: Calculate transaction costs
            # ────────────────────────────────────────────────────────────
            if current_weights:
                turnover = sum(
                    abs(final_weights.get(a, 0) - current_weights.get(a, 0))
                    for a in set(final_weights) | set(current_weights)
                ) / 2  # One-way turnover
            else:
                turnover = 1.0  # Initial investment

            transaction_cost = turnover * self.config.TOTAL_TRANSACTION_COST

            # ────────────────────────────────────────────────────────────
            # STEP 6: Calculate daily returns
            # ────────────────────────────────────────────────────────────
            # Strategy return
            strategy_return = sum(
                final_weights.get(ticker, 0) * asset_returns.loc[date, ticker]
                for ticker in final_weights
                if ticker in asset_returns.columns
            )

            # Apply costs
            net_return = strategy_return - transaction_cost - daily_mgmt_fee

            # Benchmark return
            benchmark_return = asset_returns.loc[date, benchmark_ticker]

            # ────────────────────────────────────────────────────────────
            # STEP 7: Update portfolio values
            # ────────────────────────────────────────────────────────────
            portfolio_values.append(portfolio_values[-1] * (1 + net_return))
            benchmark_values.append(benchmark_values[-1] * (1 + benchmark_return))
            cash_values.append(cash_values[-1] * (1 + daily_rf))

            # Track history
            weight_history.append(final_weights.copy())
            regime_history.append(max(regime_probs, key=regime_probs.get))
            prob_history.append(regime_probs.copy())

            current_weights = final_weights

        # Convert to arrays
        portfolio_values = np.array(portfolio_values)
        benchmark_values = np.array(benchmark_values)
        cash_values = np.array(cash_values)

        # Calculate metrics
        metrics = self._calculate_metrics(
            portfolio_values, benchmark_values, common_dates
        )

        self.results = {
            'dates': common_dates,
            'portfolio_values': portfolio_values,
            'benchmark_values': benchmark_values,
            'cash_values': cash_values,
            'weight_history': weight_history,
            'regime_history': regime_history,
            'prob_history': prob_history,
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

        # Normalize
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

        # Sharpe
        excess = port_returns - self.config.RISK_FREE_RATE / 252
        sharpe = np.mean(excess) / np.std(excess) * np.sqrt(252) if np.std(excess) > 0 else 0

        bench_excess = bench_returns - self.config.RISK_FREE_RATE / 252
        bench_sharpe = np