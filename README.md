# Chinese Customer-Service Question Similarity with Cosine Similarity

This project studies a narrow mathematical question: can character n-gram vectors and cosine similarity distinguish semantically similar Chinese customer-service questions? The experiment compares a character 2-gram CountVectorizer baseline with a character 2-gram TF-IDF model. It is an educational benchmark, not a deployed support system.

## Data source and use limits

The experiments use the AFQMC split distributed by the CLUE benchmark. The source archive is available from the [AFQMC download](https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip); the benchmark documentation is in the [CLUE repository](https://github.com/CLUEbenchmark/CLUE). The checked source pages did not specify redistribution permission for the data. Obtain the archive yourself, review the applicable terms, and do not commit the archive, extracted JSON, row-level questions, split manifest, or predictions containing examples.

The expected archive SHA-256 and member fingerprints are recorded in `stage03b_official_validation/pre_dev_freeze_manifest.json`. SHA-256 values identify the inputs; they do not grant redistribution rights. The repository intentionally has no software license file. Public visibility does not select a code license or change the dataset's terms.

## Methods

Both methods use the same vocabulary of character 2-grams and the same paired cosine calculation. The formal primary method is TF-IDF; Count is the fixed baseline. TF-IDF document frequency is fitted from the 54,330 unique, normalized question strings from internal fit only. The text normalizer applies Unicode NFC, trims surrounding whitespace, and collapses whitespace runs to one ASCII space. It preserves punctuation, digits, English letters, case, and negation.

Each method uses its threshold selected on the internal tune split by positive-class F1, with the frozen rule `score >= threshold`. These thresholds are not probabilities. Do not retune them on official Dev.

## Environment and installation

The recorded environment is Python 3.12.10, NumPy 2.5.3, SciPy 1.17.0, scikit-learn 1.9.1, and Pillow 12.3.0. Install the pinned dependencies with:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r stage03a_internal_experiment\requirements.txt
```

The Stage 2 synthetic math checks also use the numerical dependencies in `stage02_math_protocol/requirements.txt`.

## Preparing data locally

1. Download the archive from the source above into `stage01_data_audit/raw/afqmc_public.zip`.
2. Compare its SHA-256 with the frozen manifest before extracting or using it.
3. Keep all raw data and row-level audit outputs local. The historical Stage 3A runner expects the Stage 1 split manifest at `stage01_data_audit/split_manifest.csv`; that file is intentionally not included here because it contains row-level identifiers and assignments.
4. To regenerate the Stage 1 audit and local manifest, run `python stage01_data_audit/audit_afqmc.py --archive stage01_data_audit/raw/afqmc_public.zip`. This audit reads the three source splits for data-quality and overlap checks, but it does not fit text features, score pairs, select thresholds, or compute model metrics. Its row-level output files remain local and are excluded by `.gitignore`.
5. For the current frozen experiment, compare the locally generated input fingerprints with `stage03b_official_validation/pre_dev_freeze_manifest.json` before running Stage 3A. Do not treat Stage 1 auditing as permission to fit or evaluate a model on Dev or Test.

The SHA-256 fingerprint is a version check only. It is not a substitute for obtaining the data lawfully.

## Reproduction commands

Run Stage 2's hand-built vector and tiny synthetic-corpus tests:

```powershell
python stage02_math_protocol\run_math_tests.py
```

Run the Stage 3A internal fit/tune experiment only after the local Stage 1 prerequisites are present:

```powershell
python stage03a_internal_experiment\run_experiment.py --root .
```

The Stage 3A implementation tests use synthetic fixtures and also verify that the loader opens only `train.json`:

```powershell
python stage03a_internal_experiment\run_tests.py
```

Official Dev evaluation is a separate, single authorized step. First review `stage03b_official_validation/01_Dev前配置冻结审计.md` and ensure the exact pre-Dev commit is published. The runner defaults to a fit-space preflight that does not open Dev. Its explicit evaluation mode requires the local-only publication gate written after the public repository and commit have been verified. Never alter the frozen input hashes, vocabulary, IDF source, normalizer, thresholds, or evaluation code after seeing Dev results.

## Current internal results and limitations

All values below are from internal tune, where the thresholds were selected. They are not independent validation results.

| Method | F1 | ROC-AUC | Average Precision | False-positive rate |
|---|---:|---:|---:|---:|
| Count character 2-gram + cosine | 0.472888 | 0.552451 | 0.349964 | 0.968434 |
| TF-IDF character 2-gram + cosine | 0.473543 | 0.572263 | 0.360341 | 0.814604 |
| Always predict positive | 0.470942 | n/a | n/a | 1.000000 |

The TF-IDF F1 is only about 0.00260 above the always-positive baseline. Its selected threshold predicts positive for about 83.41% of tune examples. The score distributions overlap substantially and the ranking signal is weak. These results do not show strong classification performance and must not be described as a successful real customer-service system.

Reviewed, data-free aggregate evidence is collected in [`public_evidence/`](public_evidence/): dataset fingerprints, internal-tune metrics, a QA summary, five reviewed QA notes, and the score-distribution figure. The full error-case note with actual question pairs is deliberately kept local.

## Project status

Stage 0 passed; Stage 1 is conditional pass; Stage 2 passed; Stage 3A and its QA review are conditional pass. The formal primary method and Count baseline are frozen before official Dev. Character 2–3-gram TF-IDF was a preregistered internal exploratory sensitivity run and is excluded from the formal Dev comparison. Stage 4 report production and Stage 5 packaging require separate review.

## Review and academic integrity

The repository contains code, aggregate results, and data-free audit evidence. Keep the source archive, row-level questions, predictions, and error examples local. Check all calculations and cited sources, document any AI assistance required by course policy, and make the final academic decisions and report your own. No result here establishes production readiness.
