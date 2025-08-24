#!/usr/bin/env python3
"""
R:R Verification Harness
Tests synthetic scenarios and live trade data to verify entry/stop/target calculations.
"""

import argparse
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

def synthetic_rr_test(s: float, t: float, R: float, entry: float, direction: str, 
                     offsets: List[float], tol_rr: float, tol_risk: float) -> Dict:
    """
    Run synthetic R:R invariance test.
    
    Args:
        s: Stop distance in R units
        t: Target distance in R units  
        R: ATR value
        entry: Base entry price
        direction: 'long' or 'short'
        offsets: List of entry offsets in R units
        tol_rr: R:R tolerance
        tol_risk: Risk tolerance
    
    Returns:
        Dict with test results
    """
    print(f"\n=== SYNTHETIC R:R INVARIANCE TEST ===")
    print(f"Parameters: s={s}, t={t}, R={R}, entry={entry}, direction={direction}")
    print(f"Offsets: {offsets}")
    print(f"Tolerances: R:R={tol_rr:.1%}, Risk={tol_risk:.1%}")
    print("-" * 80)
    
    results = []
    risks = []
    
    for offset in offsets:
        # Calculate actual fill price
        E_f = entry + offset * R
        
        # Calculate stop and target based on direction
        if direction == "long":
            stop = E_f - s * R
            target = E_f + t * R
        else:  # short
            stop = E_f + s * R
            target = E_f - t * R
        
        # Calculate realized R:R
        rr_realized = abs(target - E_f) / abs(E_f - stop)
        
        # Calculate position size for fixed $100 risk
        risk_usd = 100.0
        position_size = risk_usd / abs(E_f - stop)
        actual_risk = position_size * abs(E_f - stop)
        
        results.append({
            'offset': offset,
            'E_f': E_f,
            'stop': stop,
            'target': target,
            'rr_realized': rr_realized,
            'rr_expected': t / s,
            'rr_error': abs(rr_realized - t/s),
            'position_size': position_size,
            'actual_risk': actual_risk
        })
        risks.append(actual_risk)
    
    # Check R:R invariance
    rr_expected = t / s
    rr_errors = [r['rr_error'] for r in results]
    max_rr_error = max(rr_errors)
    rr_pass = max_rr_error <= tol_rr
    
    # Check risk invariance
    risk_mean = np.mean(risks)
    risk_errors = [abs(r - risk_mean) for r in risks]
    max_risk_error = max(risk_errors)
    risk_pass = max_risk_error <= tol_risk * risk_mean
    
    # Print results table
    print(f"{'Offset':>8} {'E_f':>10} {'Stop':>10} {'Target':>10} {'RR_real':>8} {'RR_exp':>8} {'RR_err':>8} {'Risk':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['offset']:>8.1f} {r['E_f']:>10.2f} {r['stop']:>10.2f} {r['target']:>10.2f} "
              f"{r['rr_realized']:>8.3f} {r['rr_expected']:>8.3f} {r['rr_error']:>8.3f} {r['actual_risk']:>8.2f}")
    
    print("-" * 80)
    print(f"R:R Test: {'PASS' if rr_pass else 'FAIL'} (max error: {max_rr_error:.3f}, tol: {tol_rr:.3f})")
    print(f"Risk Test: {'PASS' if risk_pass else 'FAIL'} (max error: ${max_risk_error:.2f}, tol: ${tol_risk * risk_mean:.2f})")
    print(f"Overall: {'PASS' if (rr_pass and risk_pass) else 'FAIL'}")
    
    return {
        'pass': rr_pass and risk_pass,
        'rr_pass': rr_pass,
        'risk_pass': risk_pass,
        'max_rr_error': max_rr_error,
        'max_risk_error': max_risk_error,
        'results': results
    }

def load_invariants(path: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load R:R invariants from sidecar CSV file.
    
    Args:
        path: Path to invariants CSV file
    
    Returns:
        Tuple of (DataFrame, status) where status is "ok", "missing", or "empty"
    """
    invariants_path = Path(path)
    
    if not invariants_path.exists():
        return None, "missing"
    
    try:
        # Read CSV and normalize column names to lowercase
        df = pd.read_csv(invariants_path)
        df.columns = df.columns.str.lower()
        
        # Filter for fills only (warn_missing_fill == 0)
        if 'warn_missing_fill' in df.columns:
            df = df[df['warn_missing_fill'] == 0].copy()
        
        if df.empty:
            return None, "empty"
        
        # Coerce numeric fields with error handling
        numeric_fields = ['rr_planned', 'rr_realized_from_prices', 'rr_realized_from_fill', 'rr_distortion']
        warning_count = 0
        
        for field in numeric_fields:
            if field in df.columns:
                try:
                    df[field] = pd.to_numeric(df[field], errors='coerce')
                except Exception:
                    warning_count += 1
                    print(f"Warning: Could not parse {field} column")
        
        # For rr_real, prefer rr_realized_from_prices if finite else rr_realized_from_fill
        df['rr_real'] = df['rr_realized_from_prices'].fillna(df['rr_realized_from_fill'])
        
        # Build result DataFrame with required columns
        result_df = df[['setup_id', 'tf', 'direction', 'rr_planned', 'rr_real', 'rr_distortion']].copy()
        
        # Drop rows with NaN values in required fields
        result_df = result_df.dropna(subset=['rr_planned', 'rr_real', 'rr_distortion'])
        
        if result_df.empty:
            return None, "empty"
        
        print(f"Loaded {len(result_df)} fills from {invariants_path}")
        if warning_count > 0:
            print(f"Warning: {warning_count} numeric parsing issues encountered")
        
        return result_df, "ok"
        
    except Exception as e:
        print(f"Error loading invariants from {invariants_path}: {e}")
        return None, "missing"


def load_trade_data(runs_dir: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load trade data from CSV files.
    
    Args:
        runs_dir: Directory containing CSV files
    
    Returns:
        Tuple of (setups_df, trades_df)
    """
    setups_path = Path(runs_dir) / "setups.csv"
    trades_path = Path(runs_dir) / "trade_history.csv"
    
    setups_df = None
    trades_df = None
    
    if setups_path.exists():
        try:
            setups_df = pd.read_csv(setups_path)
            print(f"Loaded {len(setups_df)} setups from {setups_path}")
        except Exception as e:
            print(f"Error loading setups: {e}")
    
    if trades_path.exists():
        try:
            trades_df = pd.read_csv(trades_path)
            print(f"Loaded {len(trades_df)} trades from {trades_path}")
        except Exception as e:
            print(f"Error loading trades: {e}")
    
    return setups_df, trades_df

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to handle common aliases.
    
    Args:
        df: DataFrame to normalize
    
    Returns:
        DataFrame with normalized column names
    """
    column_mapping = {
        'trigger_price': 'price_at_trigger',
        'exit_ts': 'trigger_ts',
        'exit_price': 'trigger_price',
        'setup_id': 'id'
    }
    
    df = df.copy()
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]
    
    return df

def live_audit_invariants(invariants_df: pd.DataFrame, tol_rr: float) -> Dict:
    """
    Run live audit using R:R invariants data.
    
    Args:
        invariants_df: Invariants DataFrame
        tol_rr: R:R tolerance
    
    Returns:
        Dict with audit results
    """
    print(f"\n=== LIVE TRADE AUDIT (invariants) ===")
    print(f"R:R tolerance: {tol_rr:.1%}")
    print("-" * 80)
    
    if invariants_df is None or invariants_df.empty:
        print("No invariants data available for audit")
        return {'trades_analyzed': 0, 'within_tolerance': 0, 'flagged_trades': []}
    
    # Calculate metrics
    total = len(invariants_df)
    ok = sum(invariants_df['rr_distortion'] <= tol_rr)
    bad = total - ok
    
    # Calculate medians
    median_rr_planned = np.median(invariants_df['rr_planned'])
    median_rr_realized = np.median(invariants_df['rr_real'])
    median_distortion = np.median(invariants_df['rr_distortion'])
    
    # Print summary
    print(f"Fills analyzed: {total}")
    print(f"Within tolerance: {ok} ({ok/total:.1%})")
    print(f"Median R:R planned: {median_rr_planned:.3f}")
    print(f"Median R:R realized: {median_rr_realized:.3f}")
    print(f"Median distortion: {median_distortion:.3f}")
    print(f"Flagged: {bad}")
    
    # Get flagged trades
    flagged_df = invariants_df[invariants_df['rr_distortion'] > tol_rr].copy()
    
    # Print flagged trades table (first 10)
    if not flagged_df.empty:
        print(f"\nFlagged trades (first {min(10, len(flagged_df))}):")
        print(f"{'Setup ID':<20} {'TF':<4} {'Dir':<4} {'RR_plan':<7} {'RR_real':<7} {'Distortion':<10}")
        print("-" * 60)
        for _, row in flagged_df.head(10).iterrows():
            print(f"{str(row['setup_id'])[:19]:<20} {str(row['tf'])[:3]:<4} {str(row['direction'])[:3]:<4} "
                  f"{row['rr_planned']:<7.3f} {row['rr_real']:<7.3f} {row['rr_distortion']:<10.3f}")
    
    return {
        'trades_analyzed': total,
        'within_tolerance': ok,
        'within_tolerance_pct': (ok / total) * 100 if total > 0 else 0,
        'median_rr_planned': median_rr_planned,
        'median_rr_realized': median_rr_realized,
        'median_distortion': median_distortion,
        'flagged_trades': flagged_df.to_dict('records') if not flagged_df.empty else [],
        'analyzed_trades': invariants_df.to_dict('records'),
        'source': 'invariants'
    }


def live_audit(setups_df: pd.DataFrame, trades_df: pd.DataFrame, tol_rr: float) -> Dict:
    """
    Run live audit on trade data.
    
    Args:
        setups_df: Setups DataFrame
        trades_df: Trades DataFrame
        tol_rr: R:R tolerance
    
    Returns:
        Dict with audit results
    """
    print(f"\n=== LIVE TRADE AUDIT ===")
    print(f"R:R tolerance: {tol_rr:.1%}")
    print("-" * 80)
    
    # Normalize column names
    setups_df = normalize_column_names(setups_df)
    trades_df = normalize_column_names(trades_df)
    
    # Use trades_df directly if available, otherwise setups_df
    if trades_df is not None and not trades_df.empty:
        merged_df = trades_df
    elif setups_df is not None and not setups_df.empty:
        merged_df = setups_df
    else:
        merged_df = None
    
    if merged_df is None or merged_df.empty:
        print("No trade data available for audit")
        return {'trades_analyzed': 0, 'within_tolerance': 0, 'flagged_trades': []}
    
    # Analyze each trade
    analyzed_trades = []
    flagged_trades = []
    
    for idx, row in merged_df.iterrows():
        try:
            # Extract key fields with fallbacks
            entry = None
            stop = None
            target = None
            trigger_price = None
            rr_planned = None
            direction = None
            setup_id = None
            interval = None
            
            # Try different column names
            for col in ['entry', 'entry_price']:
                if col in row and pd.notna(row[col]) and str(row[col]).strip() != '':
                    try:
                        entry = float(row[col])
                        break
                    except (ValueError, TypeError):
                        continue
            
            for col in ['stop', 'stop_price']:
                if col in row and pd.notna(row[col]) and str(row[col]).strip() != '':
                    try:
                        stop = float(row[col])
                        break
                    except (ValueError, TypeError):
                        continue
            
            for col in ['target', 'target_price']:
                if col in row and pd.notna(row[col]) and str(row[col]).strip() != '':
                    try:
                        target = float(row[col])
                        break
                    except (ValueError, TypeError):
                        continue
            
            for col in ['trigger_price', 'price_at_trigger', 'exit_price']:
                if col in row and pd.notna(row[col]) and str(row[col]).strip() != '':
                    try:
                        trigger_price = float(row[col])
                        break
                    except (ValueError, TypeError):
                        continue
            
            for col in ['rr_planned', 'rr']:
                if col in row and pd.notna(row[col]) and str(row[col]).strip() != '':
                    try:
                        rr_planned = float(row[col])
                        break
                    except (ValueError, TypeError):
                        continue
            
            for col in ['direction', 'side']:
                if col in row and pd.notna(row[col]):
                    direction = str(row[col]).lower()
                    break
            
            for col in ['setup_id', 'id']:
                if col in row and pd.notna(row[col]):
                    setup_id = str(row[col])
                    break
            
            if 'interval' in row and pd.notna(row['interval']):
                interval = str(row['interval'])
            
            # Skip if missing essential data
            if entry is None or stop is None or target is None:
                continue
            
            # Calculate realized R:R
            rr_realized = abs(target - entry) / abs(entry - stop)
            
            # Calculate entry shift if trigger price available
            entry_shift = None
            if trigger_price is not None:
                entry_shift = trigger_price - entry
            
            # Calculate distortion if planned R:R available
            distortion = None
            if rr_planned is not None:
                distortion = abs(rr_realized - rr_planned) / max(rr_planned, 1e-9)
            
            trade_result = {
                'setup_id': setup_id,
                'interval': interval,
                'direction': direction,
                'entry': entry,
                'stop': stop,
                'target': target,
                'trigger_price': trigger_price,
                'entry_shift': entry_shift,
                'rr_planned': rr_planned,
                'rr_realized': rr_realized,
                'distortion': distortion
            }
            
            analyzed_trades.append(trade_result)
            
            # Flag if distortion exceeds tolerance
            if distortion is not None and distortion > tol_rr:
                flagged_trades.append(trade_result)
                
        except Exception as e:
            print(f"Error analyzing trade {idx}: {e}")
            continue
    
    # Calculate summary statistics
    if analyzed_trades:
        distortions = [t['distortion'] for t in analyzed_trades if t['distortion'] is not None]
        rr_realized_values = [t['rr_realized'] for t in analyzed_trades]
        
        median_distortion = np.median(distortions) if distortions else 0
        median_rr_realized = np.median(rr_realized_values) if rr_realized_values else 0
        within_tolerance = len([t for t in analyzed_trades if t['distortion'] is None or t['distortion'] <= tol_rr])
        within_tolerance_pct = (within_tolerance / len(analyzed_trades)) * 100 if analyzed_trades else 0
    else:
        median_distortion = 0
        median_rr_realized = 0
        within_tolerance = 0
        within_tolerance_pct = 0
    
    # Print summary
    print(f"Trades analyzed: {len(analyzed_trades)}")
    print(f"Within tolerance: {within_tolerance} ({within_tolerance_pct:.1f}%)")
    print(f"Median R:R realized: {median_rr_realized:.3f}")
    print(f"Median distortion: {median_distortion:.3f}")
    print(f"Flagged trades: {len(flagged_trades)}")
    
    # Print flagged trades table (first 10)
    if flagged_trades:
        print(f"\nFlagged trades (first {min(10, len(flagged_trades))}):")
        print(f"{'Setup ID':<20} {'TF':<4} {'Dir':<4} {'Entry':<8} {'Stop':<8} {'Target':<8} {'RR_plan':<7} {'RR_real':<7} {'Dist%':<6}")
        print("-" * 80)
        for trade in flagged_trades[:10]:
            print(f"{str(trade['setup_id'])[:19]:<20} {str(trade['interval'])[:3]:<4} {str(trade['direction'])[:3]:<4} "
                  f"{trade['entry']:<8.2f} {trade['stop']:<8.2f} {trade['target']:<8.2f} "
                  f"{trade['rr_planned']:<7.3f} {trade['rr_realized']:<7.3f} {trade['distortion']*100:<6.1f}")
    
    return {
        'trades_analyzed': len(analyzed_trades),
        'within_tolerance': within_tolerance,
        'within_tolerance_pct': within_tolerance_pct,
        'median_rr_realized': median_rr_realized,
        'median_distortion': median_distortion,
        'flagged_trades': flagged_trades,
        'analyzed_trades': analyzed_trades
    }

def save_verification_report(results: Dict, filename: str = "verification_report.csv"):
    """
    Save verification results to CSV.
    
    Args:
        results: Results from live audit
        filename: Output filename
    """
    if 'analyzed_trades' in results and results['analyzed_trades']:
        df = pd.DataFrame(results['analyzed_trades'])
        
        # Add source metadata if using invariants
        if results.get('source') == 'invariants':
            # Add source column to the DataFrame
            df['source'] = 'invariants'
        
        df.to_csv(filename, index=False)
        print(f"\nVerification report saved to {filename}")
        
        # Print source info
        if results.get('source') == 'invariants':
            print("Source: R:R invariants sidecar")
        else:
            print("Source: Legacy setups/trade_history CSVs")

def main():
    parser = argparse.ArgumentParser(description="R:R Verification Harness")
    parser.add_argument("--runs-dir", type=str, help="Directory containing CSV files")
    parser.add_argument("--synthetic", nargs=5, metavar=('s', 't', 'R', 'entry', 'direction'),
                       help="Synthetic test parameters: s t R entry direction")
    parser.add_argument("--offsets", type=str, default="-0.6,-0.3,0,0.3,0.6",
                       help="Entry offsets in R units (comma-separated)")
    parser.add_argument("--tol-rr", type=float, default=0.02,
                       help="R:R tolerance (default: 0.02)")
    parser.add_argument("--tol-risk", type=float, default=0.02,
                       help="Risk tolerance (default: 0.02)")
    parser.add_argument("--invariants-csv", default=os.getenv("RR_INVARIANT_SIDECAR_PATH", "runs/rr_invariants.csv"),
                       help="Path to R:R invariants CSV file")
    parser.add_argument("--prefer-invariants", type=int, default=1,
                       help="Prefer invariants over legacy CSVs (1=true, 0=false)")
    
    args = parser.parse_args()
    
    # Parse synthetic parameters (use defaults if not provided)
    if args.synthetic:
        s = float(args.synthetic[0])
        t = float(args.synthetic[1])
        R = float(args.synthetic[2])
        entry = float(args.synthetic[3])
        direction = args.synthetic[4].lower()
    else:
        s = 1.0
        t = 1.8
        R = 250
        entry = 118000
        direction = 'long'
    
    # Parse offsets
    offsets = [float(x.strip()) for x in args.offsets.split(',')]
    
    # Run synthetic test
    synthetic_results = synthetic_rr_test(s, t, R, entry, direction, offsets, args.tol_rr, args.tol_risk)
    
    # Run live audit if runs directory provided
    live_results = None
    if args.runs_dir:
        # Try invariants path first if preferred
        if args.prefer_invariants:
            invariants_df, status = load_invariants(args.invariants_csv)
            if status == "ok":
                live_results = live_audit_invariants(invariants_df, args.tol_rr)
                save_verification_report(live_results)
            else:
                print(f"Invariants sidecar {status} — falling back to legacy CSVs")
                setups_df, trades_df = load_trade_data(args.runs_dir)
                if setups_df is not None or trades_df is not None:
                    live_results = live_audit(setups_df, trades_df, args.tol_rr)
                    save_verification_report(live_results)
        else:
            # Use legacy path
            setups_df, trades_df = load_trade_data(args.runs_dir)
            if setups_df is not None or trades_df is not None:
                live_results = live_audit(setups_df, trades_df, args.tol_rr)
                save_verification_report(live_results)
    
    # Print final summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Synthetic test: {'PASS' if synthetic_results['pass'] else 'FAIL'}")
    if live_results:
        print(f"Live audit: {live_results['trades_analyzed']} trades analyzed, "
              f"{live_results['within_tolerance_pct']:.1f}% within tolerance")
    
    return 0 if synthetic_results['pass'] else 1

if __name__ == "__main__":
    sys.exit(main())
