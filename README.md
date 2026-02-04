# ⚠️ HARDWARE WARNING: HIGH COMPUTATIONAL LOAD
**PLEASE READ BEFORE RUNNING:** 

This script is currently **configured to run 1,000,000 (1 Million) Monte Carlo simulations.** This is an extreme stress test intended for high-performance workstations.

**1. System Requirements:**

Minimum 32GB RAM and a multi-core processor (e.g., Ryzen 7 / Core i7 or better).

**2. Memory Intensity:**

Each simulation path generates and stores thousands of data points for the 5-year projection period, leading to a massive memory footprint.

**3. Risk:** 

Running this on a standard office laptop or non-gaming PC (8GB/16GB RAM) will likely cause a Memory Overflow (OOM), resulting in a system freeze or crash.

**4. Execution Time:** 

At 1,000,000 iterations, the calculation of the probability of loss, confidence intervals, and ending value distributions will take several minutes even on high-end hardware.

## Recommendation for Standard Users: ##

If you were to run the script after carefully reading and agreeing with the "⚠️ Disclaimer and Terms of Use" below, **before running,** open alpha_dominator_v10.py and **find the configuration line: n_simulations = 1000000** (inside the main function or StrategyConfig) and **change it to 10,000** or a smaller number of your choice.

====================================================================================
# **⚠️ Disclaimer and Terms of Use**
**1. Educational Purpose Only**

This software is for educational and research purposes only and was built as a personal project by a student at National Chengchi University (NCCU). It is not intended to be a source of financial advice, and the authors are not registered financial advisors. The algorithms, simulations, and optimization techniques implemented herein—including Consensus Machine Learning, Shannon Entropy, and Geometric Brownian Motion—are demonstrations of theoretical concepts and should not be construed as a recommendation to buy, sell, or hold any specific security or asset class.

**2. No Financial Advice**

Nothing in this repository constitutes professional financial, legal, or tax advice. Investment decisions should be made based on your own research and consultation with a qualified financial professional. The strategies modeled in this software—specifically the 60% Growth Anchor and IR Filter—may not be suitable for your specific financial situation, risk tolerance, or investment goals.

**3. Risk of Loss**

All investments involve risk, including the possible loss of principal.

Past Performance: Historical returns (such as the 19.5% CAGR) and volatility data used in these simulations are not indicative of future results.

Simulation Limitations: Monte Carlo simulations are probabilistic models based on assumptions (such as constant drift and volatility) that may not reflect real-world market conditions, black swan events, or liquidity crises.

Model Vetoes: While the Rate Shock Guard and Anxiety Veto are designed to mitigate losses, they are based on historical thresholds that may fail in unprecedented macro-economic environments.

Market Data: Data fetched from third-party APIs (e.g., Yahoo Finance) may be delayed, inaccurate, or incomplete.

**4. Hardware and Computation Liability**

The author assumes no responsibility for hardware failure, system instability, or data loss resulting from the execution of the 1,000,000 Monte Carlo simulations. Execution of this code at the configured scale is a high-stress computational event that should only be performed on hardware meeting the minimum specified 32GB RAM requirements.

**5. "AS-IS" SOFTWARE WARRANTY**

**THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.**

**BY USING THIS SOFTWARE, YOU AGREE TO ASSUME ALL RISKS ASSOCIATED WITH YOUR INVESTMENT DECISIONS AND HARDWARE USAGE, RELEASING THE AUTHOR (KUANMIN KUO) FROM ANY LIABILITY REGARDING YOUR FINANCIAL OUTCOMES OR SYSTEM INTEGRITY.**
