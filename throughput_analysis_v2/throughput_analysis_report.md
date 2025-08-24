# UTC Throughput Analysis Report (V2)

*Generated on: 2025-08-24 15:21:15*

## Analysis Overview

This analysis compares emoji proposal processing efficiency between:
- **Pre-2017 Era:** Before process improvements
- **Post-2017 Era:** After process improvements

**Data Sources (V2):**
- Accepted proposals: `single_concept_accepted_proposals_v2.xlsx`
- Uses `acceptance_date_v2` and `processing_time_v2` columns
- Filtered for `nature_v2 == 'normal'` proposals only

**Sample Size:**
- Pre-2017: 119 proposals
- Post-2017: 233 proposals
- Total: 352 proposals

## Proposal Volume Context

Understanding proposal volumes is crucial for interpreting efficiency metrics, as changes in processing efficiency might be influenced by changes in proposal volume or composition over time.

### Monthly Proposal Rates

| Era | Total Proposals/Month | Accepted/Month | Rejected/Month | Active Months |
|-----|---------------------|----------------|----------------|---------------|
| Pre-2017 | 2.90 | 1.05 | 1.85 | 41 |
| Post-2017 | 6.13 | 4.61 | 1.53 | 38 |

### Proposal Outcome Composition

| Era | Total Proposals | Accepted | Rejected | Acceptance Rate | Rejection Rate |
|-----|----------------|----------|----------|-----------------|----------------|
| Pre-2017 | 119 | 43 | 76 | 36.1% | 63.9% |
| Post-2017 | 233 | 175 | 58 | 75.1% | 24.9% |

### Annual Proposal Volumes (Top 10 Years)

| Year | Total | Accepted | Rejected |
|------|-------|----------|----------|
| 2017 | 72 | 51 | 21 |
| 2018 | 70 | 55 | 15 |
| 2019 | 64 | 47 | 17 |
| 2016 | 55 | 28 | 27 |
| 2015 | 31 | 6 | 25 |
| 2020 | 27 | 22 | 5 |
| 2014 | 16 | 1 | 15 |
| 2011 | 11 | 4 | 7 |
| 2012 | 3 | 2 | 1 |
| 2013 | 3 | 2 | 1 |

**Key Volume Insights:**
- Monthly proposal rate increased by 111.3% post-2017
- Acceptance rate increased by 39.0 percentage points post-2017
- This context is important when interpreting processing efficiency metrics below

## Processing Efficiency Metrics

| Metric | Pre-2017 Mean ± SD | Post-2017 Mean ± SD | Improved | P-Value | Significant |
|--------|-------------------|---------------------|----------|---------|-------------|
| Processing Days | 101.12 ± 122.95 | 99.15 ± 148.14 | ✓ Yes | 0.0020 | ✓ Yes |
| Reference Count | 4.25 ± 2.18 | 3.10 ± 1.42 | ✗ No | 0.0000 | ✓ Yes |
| Max Dormancy Days | 68.73 ± 86.29 | 68.67 ± 94.90 | ✓ Yes | 0.0176 | ✓ Yes |
| Unique People | 49.39 ± 25.00 | 28.15 ± 25.92 | ✗ No | 0.0000 | ✓ Yes |
| Unique Entities | 16.34 ± 8.20 | 9.02 ± 5.19 | ✗ No | 0.0000 | ✓ Yes |
| Ref Distribution Mean Position | 0.45 ± 0.11 | 0.47 ± 0.09 | ✓ Yes | 0.0058 | ✓ Yes |
| Ref Distribution Early Pct | 51.76 ± 15.95 | 49.87 ± 13.54 | ✗ No | 0.1815 | ✗ No |
| Ref Distribution Late Pct | 38.77 ± 17.55 | 43.69 ± 14.50 | ✗ No | 0.0015 | ✓ Yes |

## Reference Distribution Patterns

Analysis of temporal engagement patterns shows when during a proposal's lifecycle references occur:

### Distribution Pattern Counts by Era

| Era | Head Heavy | Tail Heavy | Balanced | Single Day | Insufficient Data |
|-----|------------|------------|----------|------------|------------------|
| Pre-2017 | 45 | 16 | 58 | 0 | 0 |
| Post-2017 | 49 | 23 | 161 | 0 | 0 |

### Key Reference Distribution Insights

- **Mean reference position (0=early, 1=late):** Pre-2017: 0.452, Post-2017: 0.474
- **Early references (first third):** Pre-2017: 51.8%, Post-2017: 49.9%
- **Late references (last third):** Pre-2017: 38.8%, Post-2017: 43.7%

**Interpretation:**
- Both eras show head-heavy reference patterns (more activity early in proposal lifecycle)
- See reference distribution visualizations for detailed patterns

## Methodology

- **Statistical tests:** Mann-Whitney U (non-parametric)
- **Significance threshold:** p < 0.05
- **Improvement definition:** Lower processing time and dormancy = better; Higher engagement metrics = better
- **Reference distribution:** Timeline divided into thirds (early: 0-33%, middle: 33-67%, late: 67-100%)
- **Era classification:** Based on proposal submission date vs 2017-01-01
- **Volume normalization:** Raw metrics shown; volume context provided for interpretation

## Interpretation Notes

- **Volume effects:** Changes in efficiency metrics should be interpreted alongside volume changes
- **Composition effects:** Changes in acceptance rates may influence perceived processing efficiency
- **Temporal effects:** Longer time series in pre-2017 era may affect metric distributions
- **Process maturity:** Post-2017 improvements may reflect both procedural changes and institutional learning

