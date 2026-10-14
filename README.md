# Retinal Frailty Progression

Training code for retinal model-derived risk scores for two baseline-defined transitions in UK Biobank:

| Training task | Argument | Baseline Fried score | Target |
| --- | --- | --- | --- |
| Robustness to prefrailty | `prefrailty` | 0 | First observed score 1–2 |
| Prefrailty to frailty | `frailty` | 1–2 | First observed score ≥3 |

The tasks are fitted independently using the same training implementation. In the prefrailty task, directly observed frailty is a competing event and is censored at detection in the Cox loss. In the frailty task, recovery to robustness does not end observation.

## Scope

This repository contains model definitions, losses, fold-local covariate preprocessing, five-fold training, frozen RETFound image feature extraction, and validation used for early stopping/checkpoint selection. Feature extraction and image transforms are retained because they are required to train from prepared images. It does not contain raw UK Biobank data preparation, foundation-model pre-training, separate downstream evaluation, retinal association analyses, or genetic analyses. Participant data, feature caches, fold assignments, and trained/pretrained weights are not distributed.

## Installation

Use a Python 3.10+ environment. Install a matching PyTorch/torchvision build for your hardware, then install the remaining dependencies:

```bash
python -m venv .venv
# Activate the environment using your operating system's standard command.
python -m pip install -r requirements.txt
```

`requirements.txt` lists direct dependencies; it is not an exact lock of the original training environment. CUDA is recommended for feature extraction and training. CPU execution is selected automatically when CUDA is unavailable. Exact numerical reproduction can depend on library versions and hardware.

Obtain compatible RETFound CFP weights separately through the original provider and comply with their terms. The loader expects a compatible ViT checkpoint (the study uses 1024-dimensional features). Weights are loaded locally; this repository does not download them.

## Prepared input files

Provide an already prepared cohort directory, for example:

```text
data/
  demographicdata.csv
  baseline_image_manifest.csv
  images/
    participant_left.jpg
pretrained/
  RETFound_mae_natureCFP.pth
```

Images must correspond to the selected clinical baseline. Each participant must occur once in the cohort table and once in the image manifest. Both tasks may be present in one cohort table; `--pattern` selects the task. Separate directories may also be used.

### demographicdata.csv

Required endpoint and audit columns:

| Column | Meaning / accepted values |
| --- | --- |
| `eid` (or `Eid`) | Unique integer participant identifier |
| `pattern` | `prefrailty` or `frailty` |
| `schema_version` | `first_observed_transition_v1` |
| `score_min_components` | 3 |
| `baseline_score` | 0 for prefrailty; 1 or 2 for frailty |
| `baseline_date` | Selected baseline date in ISO format |
| `baseline_visit_instance` | Integer baseline visit instance |
| `Frailty_Score` | Score at the previously determined endpoint/last assessment, integer 0–5 |
| `months_between` | Positive follow-up in months, exactly `(endpoint_date - baseline_date).days / 30.44` |
| `event_type` | 0: censored; 1: target event; 2: directly observed frailty competing event (prefrailty task only) |
| `event` | 1 if `event_type == 1`, otherwise 0 |
| `endpoint_date` | Endpoint or censoring date in ISO format |
| `endpoint_visit` | Endpoint/censoring visit identifier |
| `last_negative_date` | Audit field for the last negative assessment |
| `last_assessment_date` | Audit field for the last evaluable assessment |

Supply all six covariates used by the multimodal model:

`Gender`, `Ethnicity`, `Age_Raw_At_Image`, `BMI_At_Image`, `Smoking_At_Image`, `Alcohol_At_Image`.

Age and BMI are numerical; gender, ethnicity, smoking and alcohol are categorical. Missing values are imputed using training-fold medians/modes. Scaling and categorical encodings are fitted on each training fold. Preserve the study's original category definitions when reproducing its models. The loader validates prepared endpoints; it does not reconstruct longitudinal events from raw visits. In the prefrailty task, endpoint scores 1–2 imply event type 1 and scores ≥3 imply event type 2. In the frailty task, endpoint scores ≥3 imply event type 1 and lower scores imply event type 0.

### baseline_image_manifest.csv

Required columns: `Eid`, `baseline_visit_instance`, `fundus_left`, `fundus_right`.

`fundus_left` contains the selected baseline left-eye image path relative to the cohort directory. `fundus_right` may be blank; the distributed training entry point uses the left eye. The manifest and cohort must contain the same participant IDs and baseline visit instances. Relative image paths make the data portable across machines.

## Train the two models

Run these commands from the repository root. Replace the example paths with your prepared input directory and locally obtained weight file. Paths containing spaces should be quoted.

### Robustness to prefrailty

```bash
python src/train.py --pattern prefrailty --modality demo_fundus --data_dir /path/to/prepared_data --cfp_weights /path/to/RETFound_mae_natureCFP.pth --output_dir outputs/robust_to_prefrailty
```

### Prefrailty to frailty

```bash
python src/train.py --pattern frailty --modality demo_fundus --data_dir /path/to/prepared_data --cfp_weights /path/to/RETFound_mae_natureCFP.pth --output_dir outputs/prefrail_to_frailty
```

These commands also work with quoted Windows paths. Without path overrides, data and weights are resolved relative to the repository, independently of the current working directory. `--output_dir` is required and is interpreted relative to the current working directory if a relative path is supplied. Use a new output directory for a new run.

The first run extracts frozen-backbone image features; later runs reuse the verified cache. By default the cache is stored inside `--data_dir`; override `--cache_path` to place it in another writable directory. Keep its `.transition.json` sidecar with the cache. The cache verification binds it to image content; use a new cache path if changing the pretrained checkpoint or feature-extraction settings.

Defaults: token cross-attention architecture; hidden dimension 128; 8 attention heads; 1 cross-attention layer; dropout 0.3; 5 folds; seed 42; up to 60 epochs; batch size 32; AdamW learning rate 2e-5; weight decay 0.05; gradient clipping 1.0; 4 warm-up epochs; patience 20; Cox weight 0.10; ranking weight 0.70; focal gamma 1.5. RETFound is frozen in this training pipeline, rather than fine-tuned end to end.

Optional single-modality training uses `--modality fundus_only` or `--modality demo_only`. These variants still use the same image-eligible cohort and verified feature cache. To reuse a task's folds across modalities, supply `--fold_assignment_csv` together with its automatically generated `.transition.json` sidecar. Do not reuse folds across different tasks/cohorts. See all options with:

```bash
python src/train.py --help
```

## Outputs

Each fold saves `best_model.pth`, `model_config.json`, training/validation predictions, and training information. The run directory also saves fold assignments with a cohort signature, `cv_summary.csv`, `cv_final_summary.txt`, `oof_predictions.csv`, and a training log. These are local training outputs and may contain participant identifiers; `.gitignore` excludes the default data, weight and output directories from version control. Training-validation C-index is used for early stopping and checkpoint selection; these validation folds are not an independent external test set.

## Source layout

- `src/train.py`: task selection and five-fold orchestration.
- `src/cross_validation.py`: fold-local preprocessing and training loop.
- `src/model.py`: model architectures.
- `src/losses.py`: survival/ranking losses.
- `src/dataset.py`: backbone loader, image transforms, feature extraction and datasets.
- `src/transition_data.py`: prepared-input validation and cache/fold signatures.
- `src/folds.py`: endpoint-class and score-balanced fold assignment.
- `src/training_validation.py`: validation required for checkpoint selection.
- `src/utils.py`: random seeds and logging.

`SOURCE_PROVENANCE.json` records source-file hashes for this export. Packaging changes make paths configurable, extract the shared fold builder, rename the training validation module, and remove unused raw-data preparation functions. Original working files are unchanged.

## Data and code availability

UK Biobank participant data require approved access and are not included. This repository provides the two prediction tasks' model training implementation; it does not provide all analyses in the manuscript. Cite the associated manuscript when bibliographic details become available, and cite RETFound and other methods as appropriate. No software license has been selected in this export; the authors should add an appropriate license before distribution.

## Verification of this export

The export was checked for Python syntax, local-module dependency completeness, removal of original machine-specific paths, prepared-cohort schema handling, and fold assignment on synthetic records. Full model training and numerical equivalence were not tested locally because PyTorch and several training dependencies are unavailable in the packaging environment. Dependency versions should be recorded from the actual training environment for a reproducible release.
