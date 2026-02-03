# ##Strategy: The Alpha Dominator (v10.0)##
##Overview##
The Alpha Dominator is a regime-adaptive, multi-asset quantitative strategy designed to structurally outperform the S&P 500 (SPY) in both CAGR and Sharpe Ratio. It achieves this by aggressively focusing capital on high-conviction structural growth leaders during bull markets while utilizing regularized machine learning to detect regime shifts.

⚖️ The Quantitative Constitution (Mandatory Logic)
To prevent "Gilded Cowardice" (hiding in safe-havens during growth periods) and "Robotic Blocks" (50/50 splits), the following rules are non-negotiable:

Kill the Sharpe Trap: In RISK_ON (Bull) regimes, never penalize assets like QQQ or XLK for high volatility. High-growth assets are inherently volatile; maximizing Sharpe in a bull market is a failure state.

Kill the Volatility Trap: Low-volatility assets (e.g., SHY, Cash) are forbidden from receiving high allocations in bull markets just because their risk-adjusted scores appear smooth.

The Velvet Rope (IR Filter): Asset eligibility in RISK_ON requires a 6-month Information Ratio (IR) > 0.5 against SPY.

Formula: (Asset_Return - SPY_Return) / Tracking_Error.

The Growth Anchor: Combined weight of QQQ + XLK must be Minimum 50% during RISK_ON.

The Gold Cap: Gold (GLD) is a hedge, not a driver. Total weight is capped at 5.0% in RISK_ON.

Shannon Entropy Diversification: Use Shannon Entropy to ensure a professional "waterfall" weight distribution rather than robotic equal-weighted blocks.

🤖 Machine Learning Framework
The strategy utilizes a Regularized Random Forest to predict the probability of a positive forward 21-day return.

Intrinsic Regularization: To prevent overfitting, the model is hard-coded with max_depth=4, min_samples_leaf=100, and ccp_alpha=0.01.

Macro Features: VIX is dropped as a lagging indicator. The model uses the Yield Spread Proxy (3m momentum of TLT/SHY) and Equity Risk Premium Proxy (Earnings Yield - Treasury Yield).

Signal Processing: Apply a 3-day EMA to ml_probs to eliminate high-turnover regime flickering.

⚙️ Operational Protocols
Adaptive Rebalancing: The system automatically selects the optimal rebalance period (21, 42, or 63 days) based on the training window's highest Information Ratio.

Transaction Modeling: Costs are based on Turnover (sum(abs(new_weights - old_weights))).

Monte Carlo Standard: Use Daily Risk-Free Rate conversions for all future projections.

📊 Health Dashboard Specifications
The plot_validation_curves method must serve as a professional health monitor:

Rolling Accuracy: Display a 252-day rolling test accuracy line.

Stability Band: Shade the area between Train and Test accuracy.

Overfitting Warning: If the accuracy gap exceeds 12% or test accuracy falls below 51%, the background must turn Light Red.

🛠 Instructions for Copilot Agent
When working on alpha_dominator_v10.py, follow these implementation tasks:

Integrate IR Logic: Ensure DataManager correctly calculates Tracking Error and Information Ratio for the "Velvet Rope" filter.

Complete Boilerplate: Finish the MonteCarloSimulator.plot_distribution and main() methods.

Enforce Optimization Constraints: In AlphaDominatorOptimizer, implement the 50% Growth Anchor floor and 5% Gold cap as high-priority soft penalties in the objective function.

De-indent main(): Ensure the main() execution block is standalone at the bottom of the file.

Terminal Output: Ensure the Final Allocation Receipt prints Asset, Weight, IR_Score, Trend Status, and Risk Contribution in a clean table format.
