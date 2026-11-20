# SoccerNet GSR 2025: swapped outfield team labels

`audit_team_labels.py` reads the official `train.zip`, `valid.zip` and `test.zip` and, for every clip, compares the jersey numbers of the outfield players labelled `left` and `right` with the other clips of the same match half. Five clips have their outfield `team` labels swapped; their goalkeepers are labelled consistently.

```bash
python dataset-audit/audit_team_labels.py --archives path/to/SoccerNet/SN-GSR-2025 --out audit
```

- [`report.json`](report.json): jersey agreement with every other clip of the half, goalkeeper pitch positions and annotation counts.
- [`corrections.json`](corrections.json): the proposed fix.
- `SNGS-*.jpg`: outfield players and goalkeeper under each label, for the flagged clip and a clip of the same half.
