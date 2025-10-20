# Code Summary — SMA-200 Regime Analysis

This script performs a 200-day moving average (SMA-200) regime analysis for TSLA daily data.  
It loads the technical feature dataset, computes forward returns across multiple horizons,  
and labels each day as “above” or “below” the SMA-200 line. The script aggregates regime  
probabilities, as well as average and median forward returns, and derives lift metrics  
to quantify performance differences between regimes. It also generates static plots and  
optional HTML summaries using Great Tables when available.

### CSV Outputs
- **tsla_sma200_regime_summary.csv** — Per-horizon sample counts, rise/drop probabilities, and return statistics split by regime.  
- **tsla_sma200_regime_lift.csv** — Above-minus-below SMA-200 differences for all key probability and return metrics.

