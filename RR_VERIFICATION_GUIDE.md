# R:R Verification Harness

## Overview

The R:R Verification Harness is a comprehensive testing tool that validates the consistency and accuracy of entry/stop/target calculations across different entry offsets. It performs both synthetic mathematical tests and live trade data audits.

## Features

### 1. Synthetic R:R Invariance Test
- **Purpose**: Verifies that R:R ratios remain constant regardless of entry price offsets
- **Method**: Tests multiple entry offsets and validates R:R invariance
- **Risk Validation**: Ensures dollar risk remains constant across offsets

### 2. Live Trade Audit
- **Purpose**: Analyzes actual trade data for R:R consistency
- **Data Sources**: `runs/setups.csv` and `runs/trade_history.csv`
- **Detection**: Identifies trades with R:R distortions

## Usage

### Basic Usage
```bash
# Run with default parameters (synthetic only)
python3 verify_rr.py

# Run with live audit
python3 verify_rr.py --runs-dir runs

# Run with custom parameters
python3 verify_rr.py --synthetic 1.0 1.8 250 118000 long --runs-dir runs
```

### Command Line Options

#### Synthetic Test Parameters
```bash
--synthetic s t R entry direction
```
- `s`: Stop distance in R units (default: 1.0)
- `t`: Target distance in R units (default: 1.8)
- `R`: ATR value (default: 250)
- `entry`: Base entry price (default: 118000)
- `direction`: 'long' or 'short' (default: 'long')

#### Offsets
```bash
--offsets "offset1,offset2,offset3"
```
- Comma-separated list of entry offsets in R units
- Default: "-0.6,-0.3,0,0.3,0.6"

#### Tolerances
```bash
--tol-rr 0.02      # R:R tolerance (default: 2%)
--tol-risk 0.02    # Risk tolerance (default: 2%)
```

#### Data Directory
```bash
--runs-dir runs    # Directory containing CSV files
```

## Examples

### Example 1: Default Synthetic Test
```bash
python3 verify_rr.py
```
**Output**:
```
=== SYNTHETIC R:R INVARIANCE TEST ===
Parameters: s=1.0, t=1.8, R=250, entry=118000, direction=long
Offsets: [-0.6, -0.3, 0.0, 0.3, 0.6]
Tolerances: R:R=2.0%, Risk=2.0%
--------------------------------------------------------------------------------
  Offset        E_f       Stop     Target  RR_real   RR_exp   RR_err     Risk
--------------------------------------------------------------------------------
    -0.6  117850.00  117600.00  118300.00    1.800    1.800    0.000   100.00
    -0.3  117925.00  117675.00  118375.00    1.800    1.800    0.000   100.00
     0.0  118000.00  117750.00  118450.00    1.800    1.800    0.000   100.00
     0.3  118075.00  117825.00  118525.00    1.800    1.800    0.000   100.00
     0.6  118150.00  117900.00  118600.00    1.800    1.800    0.000   100.00
--------------------------------------------------------------------------------
R:R Test: PASS (max error: 0.000, tol: 0.020)
Risk Test: PASS (max error: $0.00, tol: $2.00)
Overall: PASS
```

### Example 2: Short Direction Test
```bash
python3 verify_rr.py --synthetic 1.5 2.5 300 50000 short
```
**Output**:
```
=== SYNTHETIC R:R INVARIANCE TEST ===
Parameters: s=1.5, t=2.5, R=300.0, entry=50000.0, direction=short
Offsets: [-0.6, -0.3, 0.0, 0.3, 0.6]
Tolerances: R:R=2.0%, Risk=2.0%
--------------------------------------------------------------------------------
  Offset        E_f       Stop     Target  RR_real   RR_exp   RR_err     Risk
--------------------------------------------------------------------------------
    -0.6   49820.00   50270.00   49070.00    1.667    1.667    0.000   100.00
    -0.3   49910.00   50360.00   49160.00    1.667    1.667    0.000   100.00
     0.0   50000.00   50450.00   49250.00    1.667    1.667    0.000   100.00
     0.3   50090.00   50540.00   49340.00    1.667    1.667    0.000   100.00
     0.6   50180.00   50630.00   49430.00    1.667    1.667    0.000   100.00
--------------------------------------------------------------------------------
R:R Test: PASS (max error: 0.000, tol: 0.020)
Risk Test: PASS (max error: $0.00, tol: $2.00)
Overall: PASS
```

### Example 3: Live Trade Audit
```bash
python3 verify_rr.py --runs-dir runs
```
**Output**:
```
=== LIVE TRADE AUDIT ===
R:R tolerance: 2.0%
--------------------------------------------------------------------------------
Trades analyzed: 71
Within tolerance: 45 (63.4%)
Median R:R realized: 1.800
Median distortion: 0.000
Flagged trades: 26

Flagged trades (first 10):
Setup ID             TF   Dir  Entry    Stop     Target   RR_plan RR_real Dist% 
--------------------------------------------------------------------------------
setup_002            5m   sho  45200.00 45400.00 44800.00 1.800   2.000   11.1  
setup_003            5m   lon  45100.00 44900.00 45400.00 1.800   1.500   16.7  
setup_004            15m  lon  2800.00  2780.00  2828.00  1.800   1.400   22.2  
setup_005            15m  sho  2810.00  2830.00  2770.00  1.800   2.000   11.1  
setup_006            5m   lon  45300.00 45100.00 45600.00 1.800   1.500   16.7  
setup_007            5m   sho  45500.00 45700.00 45100.00 1.800   2.000   11.1  
setup_008            15m  lon  2820.00  2800.00  2844.00  1.800   1.200   33.3  
setup_009            5m   lon  45400.00 45200.00 45720.00 1.800   1.600   11.1  
setup_010            15m  sho  2830.00  2850.00  2790.00  1.800   2.000   11.1  
BTCUSDT-1h-20250820  1h   sho  113497.68 114143.41 112335.38 0.696   1.800   158.5 
```

## Mathematical Foundation

### Synthetic Test Logic

#### Long Direction
- **Entry**: `E_f = E_0 + ΔE * R`
- **Stop**: `stop = E_f - s * R`
- **Target**: `target = E_f + t * R`
- **R:R**: `RR_realized = |target - E_f| / |E_f - stop| = t / s`

#### Short Direction
- **Entry**: `E_f = E_0 + ΔE * R`
- **Stop**: `stop = E_f + s * R`
- **Target**: `target = E_f - t * R`
- **R:R**: `RR_realized = |target - E_f| / |E_f - stop| = t / s`

### Risk Invariance
- **Position Size**: `qty = risk_usd / |E_f - stop|`
- **Actual Risk**: `actual_risk = qty * |E_f - stop| = risk_usd`
- **Verification**: `actual_risk` should be constant across all offsets

## Live Audit Analysis

### Data Processing
1. **Load Data**: Reads `setups.csv` and `trade_history.csv`
2. **Normalize Columns**: Handles common column aliases
3. **Extract Fields**: Parses entry, stop, target, trigger_price, rr_planned
4. **Calculate Metrics**: Computes realized R:R and distortions

### Distortion Detection
- **R:R Realized**: `|target - entry| / |entry - stop|`
- **Distortion**: `|RR_realized - rr_planned| / rr_planned`
- **Flagging**: Trades with distortion > tolerance are flagged

### Output Files
- **Console**: Summary statistics and flagged trades table
- **CSV Report**: `verification_report.csv` with detailed per-trade analysis

## Interpretation

### Synthetic Test Results
- **PASS**: R:R and risk remain invariant across entry offsets
- **FAIL**: Mathematical inconsistency detected

### Live Audit Results
- **High % within tolerance**: Good R:R consistency
- **Low % within tolerance**: Potential issues with setup calculations
- **Flagged trades**: Individual trades requiring investigation

### Common Issues Detected
1. **R:R Distortion**: Planned vs realized R:R mismatch
2. **Entry Shift**: Trigger price differs from planned entry
3. **Stop/Target Movement**: Levels not properly adjusted for actual fill

## Use Cases

### 1. System Validation
```bash
# Validate setup calculation logic
python3 verify_rr.py --runs-dir runs
```

### 2. Parameter Testing
```bash
# Test different R:R configurations
python3 verify_rr.py --synthetic 1.0 2.0 200 100000 long
python3 verify_rr.py --synthetic 1.5 3.0 300 50000 short
```

### 3. Quality Assurance
```bash
# Strict tolerance for production validation
python3 verify_rr.py --tol-rr 0.01 --tol-risk 0.01 --runs-dir runs
```

### 4. Research & Development
```bash
# Test edge cases and extreme offsets
python3 verify_rr.py --synthetic 1.0 1.8 250 118000 long --offsets "-1.0,-0.5,0,0.5,1.0"
```

## Troubleshooting

### Common Issues

#### "No trades analyzed"
- Check if CSV files exist in specified directory
- Verify CSV format and column names
- Check for data type issues in numeric fields

#### "High distortion rates"
- Review setup calculation logic
- Check for entry price adjustments
- Verify stop/target calculation consistency

#### "Argument parsing errors"
- Use proper quoting for comma-separated values
- Ensure all required parameters are provided
- Check argument order and format

### Data Requirements

#### Required CSV Columns
- **setups.csv**: `id`, `entry`, `stop`, `target`, `rr`, `direction`, `interval`
- **trade_history.csv**: `setup_id`, `entry`, `stop`, `target`, `rr_planned`, `direction`, `interval`

#### Optional Columns
- `trigger_price` or `price_at_trigger`: For entry shift analysis
- `exit_price`: For actual fill price comparison

## Benefits

1. **Mathematical Validation**: Ensures R:R invariance across entry offsets
2. **Risk Consistency**: Verifies dollar risk remains constant
3. **Live Monitoring**: Detects real-world R:R distortions
4. **Quality Assurance**: Provides quantitative metrics for system reliability
5. **Debugging Support**: Identifies specific trades with issues
6. **Documentation**: Generates detailed reports for analysis

## Future Enhancements

1. **Advanced Metrics**: Include Sharpe ratio, win rate analysis
2. **Visualization**: Generate charts and graphs
3. **Real-time Monitoring**: Continuous validation during trading
4. **Alert System**: Notifications for distortion thresholds
5. **Database Integration**: Support for database storage systems
