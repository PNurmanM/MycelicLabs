# Mycelic vNext — adversarial review

_Written to falsify the claims in `NEXT_RESEARCH_REPORT.md`. Each attack is followed by what the artifacts say and what was done about it. Where an attack lands, it says so._

## Attack 1 — the gains are tuned on the evaluation seeds

The calibration seeds are 500–502; the evaluation seeds are 0–4 (10k) and 0–2 (50k). Every knob in the frozen configuration carries the tag of the paired file that confirmed it (`frozen_config_table` in the report). Two candidates that won on the calibration seeds lost on the evaluation seeds and were dropped, which is the protocol working, not a coincidence. **Where it lands**: the evaluation seeds were read more than once during this work (each accepted change was read on them). That is sequential testing on a fixed panel; the report's numbers are therefore optimistic by an amount five paired seeds cannot bound. Mitigation offered: the first item under *What would change my mind* is a fresh seed panel.

## Attack 2 — the ranker is a simulator artefact

Its features are exact quantities the simulator hands the kernel (witness signatures, lag, lineage). A real kernel would estimate them from claims with noise. The live-model harness in the main report measures the discrimination a frontier model achieves from raw notes; it does not yet measure discrimination from kernel-side claims. **Lands partly**: the *size* of the ranker gain is a simulator number; the *direction* (ordering was the binding stage) rests on the funnel, which counts gold patterns, not on the ranker.

## Attack 3 — the baselines were handicapped

The centralised controls adopt the ranker only where their own calibration says it is not worse; A2 adopted it and went from 0.685 to 0.740 at 10k. A_flat_rag kept the hand score by the same rule. The old A2 rows are in the OLD column of the same tables. **Does not land** on the comparison; it does mean the "gap to A2" moved less than the hierarchy's own gain.

## Attack 4 — metric gaming

Register cap: unchanged (`min(6000, max(600, n_entities))`); the unbounded register appears only in rows labelled diagnostic. Gold, worlds, seeds, metrics: unchanged (the funnel code was verified by three independent refuters before the work started, and the corrections it needed are in the research log). Question budget: a fraction of the triage queue, not of the number of gold patterns. **Does not land.**

## Attack 5 — hidden centralisation

Nothing new moves up: raw text leaving a node is 0.0 and the claim exposure fraction is identical before and after (the paired tables show `claims out` as *same*). The ranker's anchor-context features are counts over candidates the kernel already holds. Re-extraction, the one change that touched raw records, stayed local and was rejected anyway. **Does not land.**

## Attack 6 — the decoy regression is being minimised

It is not: decoy acceptance at 10k went from 0.180 to 0.400 and the one fix tried (decoy-weighted selection) was rejected because it cost found and rare recall. FDR fell slightly. **Lands**: a register with more true patterns and more decoys is a better register only if the reader's cost of a decoy is below the value of a pattern; the report does not claim otherwise.

## Attack 7 — the hierarchy still loses to centralised discovery

Yes: n/a vs 0.790 at 50k, at nan× of A2's compute. The brief's targets (0.70 / 0.60) were not met. The report says which stage holds the rest (ordering at 10k, coverage at 50k) and what would test it. **Lands.**

## Attack 8 — single-seed screens are being cited

Two negative results (weak sketch bits, descent-evidence merge) rest on one calibration seed. They are cited as screens that stopped further spending, not as findings, and they are not in the ledger table. **Lands on wording, addressed.**

## Attack 9 — the hybrid DP could leak scrambled-time decoys

Letting a link float across its clusters relaxes the order test for single-witness links. The paired runs show decoy acceptance +0.03 for the hybrid arm with the refitted ranker (inside noise) and −0.01 with the previous ranker; the D2 (temporal scramble) family is in `decoy_D2_temporal_scramble` in every row for anyone who wants to check the family separately. **Not resolved with five seeds**; watch it on the fresh panel.

## Revision after the review

- Kept: the three accepted changes and the lean arm as a labelled Pareto point.
- Reworded: the single-seed screens; the decoy regression is stated in the first paragraph of the report.
- Added to the queue: a fresh evaluation panel (seeds 5–9) before any of the reported deltas is quoted outside this repository.

## Independent review

_Three refuters were run as separate agents with read-only access to the repository, each told to falsify one group of claims (protocol and leakage; the hybrid DP and decoys; metrics, cap and funnel integrity). Their reports follow verbatim, each with the author's response._

### Refuter 1 — protocol and leakage (verbatim)

**Claim 1 — the stored ranker was fitted only on seeds 500–502.** Two of the three source dumps contain evaluation-seed rows (`hyp_features_v3C.jsonl`: 5,400 cal + 9,000 eval; `hyp_features_J.jsonl`: 8,810 cal + 15,335 eval; `hyp_features_hybH.jsonl`: 9,404 rows, all cal). `fit_and_store` keeps only `r["seed"] in CAL_SEEDS`; `_loso_found` trains and tests inside the calibration seeds. 9,404 + 5,400 + 8,810 = 23,614 = the stored `n_train`. The stored weights equal `calibration.hyb.json`'s. The fit (10:55) preceded the first evaluation read of hybrid (10:57). Caveats, not leakage: `hyp_features_J.jsonl` is the same candidate set as `hyp_features_v3H.jsonl` (identical features; only `conf` differs on 881 of 24,145 rows), so 8,810 of the training rows are min-timing hierarchy candidates relabelled "J", and "refitted under the pipeline it will run in" holds for the 9,404 hybH rows (40%). **Verdict: SURVIVES.** Severity low. Fix: write the dump's `cfg_over`/`link_time` into every row and into the ranker provenance; drop or relabel the J duplicate.

**Claim 2 — every frozen knob was chosen on calibration seeds and only confirmed on evaluation seeds.** `question_frac = 0.65` was chosen on the evaluation seeds, twice: the only calibration-seed budget sweep is the v1 grid over (0.25, 0.55, 1.0) with the hand ranker, which chose 0.25; under a learned ranker the budget was swept only on seeds 0–4 (`quick_rk2_qf*.jsonl`: 0.510 / 0.520 / 0.510 / 0.515; `quick_qf_v3.jsonl`: 0.555 / 0.585 / 0.565 / 0.565), and the 50k sweep is on evaluation seeds 0–2. `local_reextract = False` was decided on the evaluation seeds against the single calibration-seed screen (+0.125). `link_time = hybrid` is a calibration-consistent choice with an evaluation-seed accept gate (modal rejected there, hybrid kept; hybrid's evaluation delta `[0, +0.125, 0, +0.075, 0]`). `batched_descent` is pure metering (identical per-seed found). The ranker design was iterated v1 → v2 → v3 with an evaluation-seed A/B after each; only its hyperparameters were leave-one-seed-out on calibration seeds. The provenance column in `vnext_data.py` emits a literal "calibration seeds 500–502, confirmed on held-out 0–4" for every knob, and `report_text.py` says every knob is fitted on calibration seeds — false for `question_frac` and `local_reextract`. **Verdict: REFUTED** (for those two knobs; partially for `link_time`). Severity high for the written claim, medium for the numbers (the 0.65 margin is 0.02–0.03 found, about one winner's curse). Fix: derive the "chosen on" column from the seeds in the cited evidence file; relabel or re-run the budget sweep on seeds 500–502 under the frozen ranker.

**Claim 3 — sequential testing on a fixed panel.** 86 arm-reads on the evaluation seeds across 33 `quick_*.jsonl` files: about 40 configurations at 10k on seeds 0–4 and 17 arms at 50k on seeds 0–2; about 11 accept/reject decisions at 10k and 6 at 50k. Report §6's "read once on seeds 0–4" is not what the files show. Median sd of a paired found delta over 5 seeds = 0.045, so the standard error of a paired mean is 0.020; max-of-k under a null: k=4 → +0.021, k=8 → +0.029, k=12 → +0.033, k=20 → +0.038 found. The small accepted deltas (budget +0.030, hybrid +0.040, lean +0.025) are 1–2 standard errors and at or below the winner's curse; the large one (+0.210, per seed `[0.175, 0.200, 0.225, 0.175, 0.275]`, se 0.019) is more than 10 standard errors and survives any correction. Expect 0.625 to read about 0.58–0.60 on a fresh panel and 0.587 about 0.55. No quick file, dump or funnel row ever touched seeds 5–9 at 10k, and `final_rerun.sh` runs `e1c` (seeds 5–9) — but it appends into `e1_baselines.jsonl`, so the report would pool 0–9 into one mean. **Verdict: REFUTED** as a "read once" claim. Severity medium. Fix: report seeds 5–9 at 10k as a separate confirmation panel and label 0–4 as the development panel.

**Claim 4 — per-architecture adoption is decided on calibration seeds.** `select_archs` defaults to `CAL_SEEDS`; the adoption log's values are multiples of 1/120 (3 seeds × 40 gold). No code path passes evaluation seeds; `H_mycelic_prev` is excluded; B2 adopts on an exact tie; A_flat_rag flipped to adopt under the refitted ranker. **Verdict: SURVIVES.** Severity low. Fix: write the seeds and per-seed found into `ranker_arch_table`.

**Claim 5 — the A/B base rows had the ranker OFF.** `quick_v3_H.jsonl` base rows equal the archived v1 `e1_baselines.jsonl` rows for H at 10k seeds 0–4 on all 12 metrics compared (found 0.300/0.300/0.425/0.475/0.375, compute 1042841.82, calls 86616, 220 questions); the v1 rows predate the ranker, so the base is provably hand-ranked and the vNext code changes did not alter v1 behaviour. The variant rows match the `qf0.65` arm that forces the ranker on. `quick_paired` rows carry no `ranker` key, so ON/OFF was inferred, not logged. **Verdict: SURVIVES.** Severity low. Fix: record the ranker state in every quick row.

**What a careful reader should believe (refuter's summary).** The ranker fit and the adoption are clean. The frozen-config "chosen on calibration seeds" column was hardcoded text; `question_frac` and `local_reextract` were decided on the evaluation seeds, and the hybrid/modal gate was evaluation-seed too. Seeds 0–4 were read about 86 times over about 40 configurations with about 11 decisions; the winner's curse is about +0.02–0.04 found, so the +0.03 (budget) and +0.04 (hybrid) gains are not established. The +0.21 at 10k and +0.16 at 50k survive any selection correction on these worlds, in this simulator. Treat 0.625 / 0.587 as development-panel numbers; seeds 5–9 at 10k should be reported separately as the confirmation.

**Author's response.** Accepted in full. Changes made: (1) the provenance column is now derived from the seeds in each cited evidence file, so a knob whose only evidence is on the evaluation panel says so; (2) the research report reports seeds 5–9 at 10k as a separate confirmation panel (OLD rows for them exist in the archive; no development decision read them) and labels 0–4 as the development panel; (3) §6 of the report is rewritten around the refuter's counts and the winner's-curse estimate; (4) a budget sweep on the calibration seeds under the frozen ranker and timing was run (`quick_qf_cal.jsonl`) and cited next to the evaluation-seed sweeps; (5) `quick_paired` rows now record the ranker state; (6) the `report_text.py` sentence is corrected. The J-dump duplication and the missing per-row `cfg_over` in dumps are noted as fixes for the next refit, not applied to the frozen ranker (which would change the numbers under review).


## What would change my mind

- If `H_mycelic_prev` inside the new suite does not reproduce the archived v1 rows to the third decimal, the old-vs-new comparison is contaminated by a code change and every Δ in section 1 is suspect.
- If the learned ranker's out-of-sample AUC in section 7 is not above the hand-set score's, the found gain is a gate artefact and should not survive a different register cap.
- If a fresh set of evaluation seeds (5–9) gives hybrid timing a negative paired delta, it joins modal timing in the rejected column.
- If A2 with the ranker beats the hierarchy at 50k at equal compute (it does not today: 10.2e6 vs 3.9e6 units), the compute argument for the hierarchy is gone and only the privacy argument remains.
- If queue item 1 shows 50k coverage is depth-limited, the lean arm's mechanism is the wrong direction and breadth spending should be reverted.
- If the live discrimination harness shows a frontier model extracting a signal from raw notes that no kernel-side feature carries, the ranker's ceiling is a property of the simulator, not of the design.
