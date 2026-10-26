# Market Risk: VaR, Expected Shortfall and Out-of-Sample Backtesting

A single, self-contained Jupyter notebook that builds, forecasts and **validates** nine one-day Value-at-Risk / Expected Shortfall models on daily equity-index returns, using the statistical toolkit of a market-risk backtesting function.

**Research question.** How do historical, parametric, Monte Carlo and conditional-volatility VaR models compare in out-of-sample coverage, violation clustering and tail-risk forecasting, particularly during market stress?

## Models

| Family | Models | Window |
|---|---|---|
| Historical simulation | HS (with 250/500/1,000-day sensitivity) | 500 d |
| Unconditional parametric | Gaussian, variance-standardised Student-t | 500 d |
| Monte Carlo | Student-t, 10,000 draws/day (seeded) | 500 d |
| Conditional volatility | GARCH(1,1) and EGARCH(1,1), Gaussian and Student-t innovations | 1,000 d, refit every 21 d |
| **Extension** | **Adaptive Hybrid**: GARCH-QMLE volatility × empirical tail of re-standardised residuals (filtered historical simulation) | 1,000 d |

## Validation toolkit

* **VaR coverage:** Kupiec POF, Christoffersen independence and conditional coverage, Engle–Manganelli Dynamic Quantile test with **exact Monte Carlo p-values**.
* **Expected Shortfall:** McNeil–Frey exceedance-residual test (bootstrap) and the strictly consistent **FZ0 joint VaR–ES score**, with Diebold–Mariano (Newey–West) tests.
* **Regulatory and stress views:** Basel traffic-light zones; violation rates by an **ex-ante** volatility regime (measurable at *t−1*, so a correct model must hit the nominal rate in both regimes); named crisis episodes.
* **Engineering checks:** a look-ahead test (future returns overwritten with noise; past forecasts must be bit-identical), closed-form ES vs numerical integration, unit tests for test-statistic boundary cases, an admissibility screen for "converged" but degenerate GARCH estimates, and own volatility recursions verified against `arch`.

## Principal findings (reference run)

> The notebook downloads **SPY** via `yfinance` and caches it to `data/`. The saved outputs were produced offline, so the notebook used its documented fallback: the daily **S&P 500 index bundled with `arch`** (1999–2018; 4,030 out-of-sample days, Dec 2002 – Dec 2018). Re-running online regenerates every table and chart for SPY.

| Model (99% VaR) | Violations (exp. 40.3) | Kupiec p | Christoffersen ind. p | DQ p (MC) | Mean FZ0 ↓ |
|---|---:|---:|---:|---:|---:|
| HS | 56 | 0.019 | 0.001 | <0.001 | −3.119 |
| Gaussian | 105 | <0.001 | <0.001 | <0.001 | −2.664 |
| Student-t | 83 | <0.001 | <0.001 | <0.001 | −2.995 |
| GARCH-N | 90 | <0.001 | 0.505 | <0.001 | −3.323 |
| GARCH-t | 63 | 0.001 | 0.095 | <0.001 | −3.447 |
| EGARCH-t | 66 | <0.001 | 0.420 | 0.002 | −3.373 |
| **Hybrid** | **50** | **0.139** | **0.157** | 0.001 | **−3.468** |

1. **Volatility conditioning dominates.** Unconditional models fail the independence, conditional-coverage and DQ tests at 95% and 99%. They breach the 99% VaR on 3–7% of ex-ante stressed days and record 15–21 violations in the 2008–09 crisis (1.5 expected).
2. **Innovation shape matters next.** Gaussian GARCH under-covers at 99% (2.2%). A symmetric Student-t fixes most of that but over-violates at 95% (6.1%).
3. **EGARCH's in-sample advantage did not carry out of sample,** and its estimation needed admissibility checks: `arch`'s internal variance bounds can mask a non-invertible EGARCH whose unclipped forecast explodes.
4. **The Adaptive Hybrid ranks first on the FZ0 score at both levels** and is the only model that passes Kupiec, independence and conditional coverage at both 95% and 99%. Its edge comes from the negative skew of standardised residuals. It is significantly better than GARCH-N at 99%, but not significantly better than GARCH-t.
5. **No model passes everything.** DQ rejects all models at 99% (violations cluster at lags 2–4, invisible to first-order tests), and the asymptotic χ² critical value (12.6) understates the simulated one (14–16). Monte Carlo simulation reproduces the analytical Student-t model plus noise.

## Repository

```
var_backtesting.ipynb   # the complete study (executed, with outputs)
requirements.txt
data/SPY_adj_close.csv  # created on first online run (frozen dataset for reproducibility)
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute var_backtesting.ipynb --inplace   # ~2-3 minutes
```

Every parameter (ticker, dates, windows, refit frequency, seeds, simulation counts) is in the configuration cell. The run is deterministic: all simulations use fixed seeds, and the downloaded data are frozen in `data/` after the first run. Commit that CSV to make published results reproducible.

## References

* Acerbi, C. & Tasche, D. (2002). On the coherence of Expected Shortfall. *Journal of Banking & Finance*, 26(7).
* Barone-Adesi, G., Giannopoulos, K. & Vosper, L. (1999). VaR without correlations for portfolios of derivative securities. *Journal of Futures Markets*, 19(5).
* Basel Committee on Banking Supervision (1996; 2019). Backtesting framework; *Minimum capital requirements for market risk* (FRTB).
* Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3).
* Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4).
* Engle, R. F. & Manganelli, S. (2004). CAViaR. *Journal of Business & Economic Statistics*, 22(4).
* Fissler, T. & Ziegel, J. F. (2016). Higher order elicitability and Osband's principle. *Annals of Statistics*, 44(4).
* Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2).
* McNeil, A. J. & Frey, R. (2000). Estimation of tail-related risk measures for heteroscedastic financial time series. *Journal of Empirical Finance*, 7.
* McNeil, A. J., Frey, R. & Embrechts, P. (2015). *Quantitative Risk Management* (revised ed.). Princeton University Press.
* Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. *Econometrica*, 59(2).
* Patton, A. J., Ziegel, J. F. & Chen, R. (2019). Dynamic semiparametric models for expected shortfall (and value-at-risk). *Journal of Econometrics*, 211(2).

The notebook's Section 12 has the full list, including the Diebold–Mariano, Dufour, Newey–West, Straumann–Mikosch and Wintenberger references.
