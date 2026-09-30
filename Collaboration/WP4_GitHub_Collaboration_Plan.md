# WP4 GitHub Collaboration Plan

Repository: `Ashish16811/PRT661-Energy-Demand-Forecasting`

This plan describes the **next** genuine collaboration cycle. No branch, issue number, PR number or commit is claimed to exist yet; GitHub assigns numbers when `GitHub_Issues/create_wp4_github_issues.sh` is run. Do not back-date branches or commits.

## Members, issues and branches

| Member | Issue(s) | Branch | Code file | Reviewer | Expected PR | Merge dependency |
|---|---|---|---|---|---|---|
| Ashish Shrestha | 01 (lead 09, 11) | `wp4/ashish-mlp-integration` | `WP4_Ashish_MLP_Model_Integration.py` | Bishal Dahal | "WP4: MLP module extracted from pipeline Section Q" | `wp4_contracts.py` |
| Bishal Dahal | 02, 07, 10 | `wp4/bishal-mlp-validation` | `WP4_Bishal_MLP_Validation_Evidence.py` | Ashish Shrestha | "WP4: independent MLP validation and evidence" | Issue 01 interfaces |
| Suraj Raut | 04 (co-owner 12) | `wp4/suraj-gru-sequence` | `WP4_Suraj_GRU_Sequence_Engine.py` | Sudip Lamichhane | "WP4: GRU sequence and recursive engine" | `wp4_contracts.py` |
| Sudip Lamichhane | 05, 08 (co-owner 12) | `wp4/sudip-gru-training` | `WP4_Sudip_GRU_Training_Provenance.py` | Suraj Raut | "WP4: GRU training controls and provenance" | Issue 04 (`SequenceBatch`) |
| All four | 03, 06, 09 | `wp4/neural-model-integration` | pipeline + all modules | all four | "WP4: integrate neural modules (equivalence-gated)" | Issues 01-08 |

## Review flow

```
Ashish PR  --review-->  Bishal        Suraj PR  --review-->  Sudip
Bishal PR  --review-->  Ashish        Sudip PR  --review-->  Suraj
                 \                        /
          wp4/neural-model-integration  (all four review; Ashish integrates)
                           |
               Issue 10 regression gate (Bishal)
                           |
                         main
```

## Working rules

1. One issue per PR; reference it in the PR description (`Refs #<number GitHub assigned>`).
2. Each PR attaches the console output of its module self-test and of `run_wp4_equivalence_tests.py`.
3. A reviewer runs the tests locally before approving; approvals are not given on reading alone.
4. Merge with "squash and merge" into the integration branch; only the integration branch merges into `main`.
5. A PR that changes model behaviour gets the `rerun-required` label and cannot be described as producing the WP4 results until the full academic run is repeated.
6. Issues stay open until the Definition of Done in the issue file is met.

## Suggested command sequence (per member)

```bash
git checkout main && git pull
git checkout -b wp4/<member-branch>
# copy the module + wp4_contracts.py into the repo folder agreed in Issue 09
python run_wp4_equivalence_tests.py
git add <files> && git commit -m "WP4: <short description> (refs #<issue>)"
git push -u origin wp4/<member-branch>
gh pr create --base wp4/neural-model-integration --title "<PR title>" --body-file <notes>.md
```
