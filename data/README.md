# Data

The analysis uses CAS Schedule P data (publicly available). Download the two CSV files and place them in this directory:

## Commercial Auto

- **File**: `comauto_pos_98-07.csv`
- **URL**: https://www.casact.org/sites/default/files/2026-03/comauto_pos_98-07.csv

## Other Liability (Occurrence)

- **File**: `othliab_pos_98-07.csv`
- **URL**: https://www.casact.org/sites/default/files/2026-03/othliab_pos_98-07.csv

Alternatively, the analysis scripts will automatically download the files if they are not present in this directory.

## Format

Both files are CAS Schedule P annual statement data. Relevant columns:

| Column | Description |
|--------|-------------|
| `GRCODE` | Company group identifier |
| `GRNAME` | Company group name |
| `AccidentYear` | Accident year |
| `DevelopmentLag` | Development lag (we use lag = 10, i.e., 10-year ultimate) |
| `IncurredLosses` | Incurred losses at the selected development lag |
| `EarnedPremNet` | Net earned premium ($000s) |

After filtering (`DevelopmentLag == 10`, `EarnedPremNet >= 100`, complete 10-year history, positive loss ratios):

- Commercial Auto: 96 qualifying company groups
- Other Liability: ~45 qualifying company groups (one extreme outlier removed)

## Why data is not committed

The CAS data files are large (~1MB each) and publicly available via the links above. Committing them would bloat the repository unnecessarily.
