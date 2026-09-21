"""
run_evaluation.py — Portfolio construction & evaluation from pre-computed predictions
====================================================================================
Input: test_data.parquet with actual returns + 6 models' predictions
Output: portfolio returns, metrics, comparison table, verdict

    python run_evaluation.py  # uses test_data.parquet from current dir
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def build_portfolios_from_predictions(df, cost_bps=50.0):
    """
    Build portfolios from pre-computed predictions.
    
    df: DataFrame with columns [eom, gvkey, ret_exc_true, ret_exc_pred_<model>...]
    cost_bps: transaction cost in basis points
    
    Returns: dict of {model: {strategy: Series of monthly returns}}
    """
    model_cols = [c for c in df.columns if c.startswith("ret_exc_pred_")]
    models = [c.replace("ret_exc_pred_", "") for c in model_cols]
    
    ports = {}
    for model, pred_col in zip(models, model_cols):
        ports[model] = {}
        
        # ── Long/Short Decile (L/S 10% EW) ───────────────────────────────────
        rets = []
        for date, grp in df.groupby("eom"):
            n = len(grp)
            top_n = max(1, int(0.10 * n))
            
            # Long: top 10%
            top_idx = grp[pred_col].nlargest(top_n).index
            top_ret = grp.loc[top_idx, "ret_exc_true"].mean()
            
            # Short: bottom 10%
            bot_idx = grp[pred_col].nsmallest(top_n).index
            bot_ret = grp.loc[bot_idx, "ret_exc_true"].mean()
            
            # L/S return: long top, short bottom, equal weight
            ls_ret = (top_ret - bot_ret) / 2.0
            
            # Subtract transaction cost (rough: turnover ~50% per month for L/S)
            turnover = 0.70  # assumption
            cost = turnover * (cost_bps / 10000)
            net_ret = ls_ret - cost
            
            rets.append(net_ret)
        
        ports[model]["LS_decile_EW"] = pd.Series(rets, 
            index=sorted(df["eom"].unique()))
        
        # ── Long-Only Decile (Long 10% EW) ───────────────────────────────────
        rets = []
        for date, grp in df.groupby("eom"):
            n = len(grp)
            top_n = max(1, int(0.10 * n))
            top_idx = grp[pred_col].nlargest(top_n).index
            top_ret = grp.loc[top_idx, "ret_exc_true"].mean()
            
            # Subtract cost (turnover ~50% for long-only decile rebalance)
            turnover = 0.50
            cost = turnover * (cost_bps / 10000)
            net_ret = top_ret - cost
            
            rets.append(net_ret)
        
        ports[model]["LongOnly_EW"] = pd.Series(rets,
            index=sorted(df["eom"].unique()))
    
    return ports


def compute_metrics(ret_series, freq=12):
    """Compute backtest metrics from monthly return series."""
    r = ret_series.dropna()
    if len(r) < 2:
        return {}
    
    cum = (1 + r).cumprod()
    dd = (1 - cum / cum.cummax())
    
    ann_ret = r.mean() * freq
    ann_vol = r.std(ddof=1) * np.sqrt(freq) if r.std(ddof=1) > 0 else 0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    max_dd = dd.max()
    calmar = ann_ret / max_dd if max_dd > 0 else 0
    
    # Sortino: downside deviation only
    ds = r[r < 0].std(ddof=1) * np.sqrt(freq) if len(r[r < 0]) > 0 else 0
    sortino = ann_ret / ds if ds > 0 else np.inf
    
    hit_rate = (r > 0).mean()
    
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "hit_rate": hit_rate,
        "n_months": len(r),
    }


def evaluate_signal_quality(df):
    """Compute signal quality metrics (RIC, R²) for each model."""
    model_cols = [c for c in df.columns if c.startswith("ret_exc_pred_")]
    models = [c.replace("ret_exc_pred_", "") for c in model_cols]
    
 
    rows = []
    for model, pred_col in zip(models, model_cols):
        # Rank IC: monthly Spearman correlation
        ric_list = []
        for date, grp in df.groupby("eom"):
            if len(grp) < 3:
                continue
            from scipy.stats import spearmanr
            ric, _ = spearmanr(grp[pred_col], grp["ret_exc_true"])
            ric_list.append(ric)
        
        ric_mean = np.nanmean(ric_list)
        ric_std = np.nanstd(ric_list)
        ric_t = ric_mean / (ric_std / np.sqrt(len(ric_list))) if ric_std > 0 else 0
        hit_rate = (np.array(ric_list) > 0).mean()
        
        # OOS R²: cross-sectional R² per month
        ss_res = ((df['ret_exc_true'] - df[pred_col]) ** 2).sum()
        ss_tot = (df['ret_exc_true'] ** 2).sum()
        r2_oos = 1 - ss_res / ss_tot
        
        rows.append({
            "model": model,
            "ric_mean": ric_mean,
            "ric_std": ric_std,
            "ic_t_stat": ric_t,
            "hit_rate": hit_rate,
            "oos_r2": r2_oos,
        })
    
    return pd.DataFrame(rows).set_index("model")


def main():
    print("Loading test_data.parquet...")
    df = pd.read_parquet("data/processed/test_data.parquet")
    print(f"  {len(df)} rows, {df['eom'].nunique()} dates, "
          f"{df['gvkey'].nunique()} companies")
    
    # ── Signal Quality ───────────────────────────────────────────────────────
    print("\n=== SECTION 1: Signal Quality ===")
    signals = evaluate_signal_quality(df)
    print(signals.round(4))
    
    # ── Build Portfolios ─────────────────────────────────────────────────────
    print("\n=== SECTION 2: Building Portfolios ===")
    ports = build_portfolios_from_predictions(df, cost_bps=10.0)
    print(f"  {len(ports)} models × {len(next(iter(ports.values())))} strategies "
          f"= {len(ports) * len(next(iter(ports.values())))} total")
    
    # ── Metrics ──────────────────────────────────────────────────────────────
    print("\n=== SECTION 3: Grand Comparison ===")
    rows = []
    for model in sorted(ports.keys()):
        for strat in sorted(ports[model].keys()):
            ret = ports[model][strat]
            metrics = compute_metrics(ret)
            metrics["model"] = model
            metrics["strategy"] = strat
            rows.append(metrics)
    
    comp = pd.DataFrame(rows)
    cols_order = ["model", "strategy", "ann_return", "ann_vol", "sharpe",
                  "sortino", "max_drawdown", "calmar", "hit_rate", "n_months"]
    comp = comp[[c for c in cols_order if c in comp.columns]]
    print(comp.round(3).to_string(index=False))
    
    # ── Final Verdict ────────────────────────────────────────────────────────
    print("\n=== SECTION 4: Final Verdict ===")
    best_idx = comp["sharpe"].idxmax()
    best = comp.loc[best_idx]
    print(f"WINNER: {best['model'].upper()} + {best['strategy']}")
    print(f"  Sharpe: {best['sharpe']:.2f}")
    print(f"  Return: {best['ann_return']:.2%}  |  Vol: {best['ann_vol']:.2%}")
    print(f"  Max DD: {best['max_drawdown']:.2%}")
    
    # ── Save ─────────────────────────────────────────────────────────────────
    import os
    out_dir = "." if os.access(".", os.W_OK) else "/tmp"
    csv_path = os.path.join(out_dir, "data/processed/grand_comparison.csv")
    comp.to_csv(csv_path, index=False)
    print(f"\n✓ Saved {csv_path}")
    print("  Open dashboard.ipynb in VS Code and click Run All to see all charts.")
    
    return comp, signals, ports


if __name__ == "__main__":
    main()
