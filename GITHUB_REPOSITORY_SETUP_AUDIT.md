# Public Repository Setup and Pre-Dev Freeze Audit

## Destination and license

- Intended repository: `https://github.com/thktyv8gjx-pixel/afqmc-cosine-similarity-lab`
- Intended visibility: Public
- Software license: deliberately not selected; no `LICENSE` file is included.
- AFQMC redistribution: the checked CLUE source pages did not specify redistribution permission. The archive and extracted data are not included.

## Public content boundary

The public allowlist consists of reviewed source code, synthetic math tests, pinned configuration, aggregate internal-tune metrics, data-free QA reports, the pre-Dev freeze record, README, and issue forms. It excludes raw archives and raw split JSON, the split manifest and overlap rows, fit-corpus text, feature strings, per-example predictions, error examples, student reports, the instructor template, runtime logs and unsanitized local logs (the committed test log is redacted), caches, secrets, and machine-specific absolute paths.

The `.gitignore` is a secondary safeguard, not the allowlist. The repository is initialized without importing any pre-existing project history. Files are staged by explicit path and the staged tree is reviewed before publication.

## Findings before first publication

- The workspace did not have an existing `.git` directory, branch, remote, or commit history.
- Git is installed. GitHub CLI and Git Credential Manager were not available in the local command environment.
- The authenticated GitHub connector identified `thktyv8gjx-pixel`; the requested repository name was not found.
- The browser was not signed in to GitHub. No password, passkey, OTP, or access token is stored in this repository.
- No license is inferred from the Public visibility setting.
- A secret-pattern scan of 163 project files found no credential-pattern hits. The AFQMC ZIP was excluded from content scanning and was not extracted for this audit.

## Pre-Dev release gate

The frozen configuration and evaluator source hashes are stored in `stage03b_official_validation/pre_dev_freeze_manifest.json`. The local-only publication record must be created only after the repository is verified Public and the exact `main` commit is verified remotely. The evaluator's default preflight fits the internal-fit feature space and does not open official Dev. The Dev execution command remains gated on that local record.

The only formal methods are character 2-gram TF-IDF plus cosine (primary) and character 2-gram Count plus cosine (baseline). The exact thresholds are frozen from internal tune. The preregistered character 2–3-gram sensitivity experiment is not part of official Dev.

## Publication verification record

This file records the setup procedure, not a claim that a remote repository has already been created or pushed. Add the verified repository URL, visibility, initial commit SHA, and issue links only after the corresponding GitHub state is confirmed.
