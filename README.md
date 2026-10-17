# BLIP Image Captioning: Fine-tuning, Evaluation & Error Analysis

An experimental study of fine-tuning **BLIP** for image captioning on **MS COCO**, including hyperparameter comparisons, official captioning metrics, paired prediction analysis, and human-reviewed qualitative examples.

> **Main finding:** Fine-tuning changed some captions and occasionally improved their specificity, but did **not** outperform the original pretrained BLIP checkpoint on the primary test-set CIDEr metric. This repository documents both the gains and the limitations.

## Project overview

This project investigates how additional supervised fine-tuning affects an already pretrained vision-language captioning model. The workflow covers:

- Fine-tuning BLIP on subsets of COCO caption annotations.
- Comparing learning rates and training durations while keeping the evaluation protocol consistent.
- Evaluating generated captions against the **COCO Karpathy test split (5,000 images)**.
- Comparing baseline and fine-tuned predictions image by image.
- Inspecting 20 selected images to distinguish genuine visual changes from lexical differences.

The fine-tuning and evaluation experiments were run on a university SLURM cluster using **one NVIDIA A40 GPU**. CPU-only scripts were used for post-hoc caption analysis and qualitative sample selection.

## Experimental setup

| Setting | Value |
|---|---|
| Model | BLIP image captioning, base configuration |
| Starting weights | Pretrained BLIP captioning checkpoint (`model_base_caption_capfilt_large.pth`) |
| Dataset | MS COCO, Karpathy splits |
| Main training subset | 100,000 caption annotations (19,987 unique images) |
| Test set | 5,000 images |
| Batch size | 32 |
| Optimizer | AdamW |
| Caption decoding | Beam search, beam size 3; minimum length 5; maximum length 20 |
| Random seed | 42 |
| Hardware | 1 × NVIDIA A40 |
| Scheduler | SLURM |

**Sampling detail:** The 100K subset was taken from the first 100,000 entries of the available training annotation list, rather than from a uniformly random sample of images. Because multiple captions may refer to the same image, 100K annotations corresponded to 19,987 distinct training images.

## Quantitative results

All scores below were computed on the **same 5,000-image COCO Karpathy test split**. Higher is better.

| Model / experiment | BLEU-4 | CIDEr | SPICE |
|---|---:|---:|---:|
| Original pretrained BLIP | **0.3968** | **1.3330** | 0.2378 |
| Fine-tuned, 10K annotations, 1 epoch, LR 1e-5 | 0.3935 | 1.3130 | 0.2360 |
| Fine-tuned, 100K annotations, 1 epoch, LR 5e-6 | 0.3914 | 1.3210 | 0.2363 |
| Fine-tuned, 100K annotations, 1 epoch, LR 1e-6 | 0.3946 | 1.3279 | **0.2380** |

The original pretrained model retained the highest test CIDEr and BLEU-4 scores. The best fine-tuned checkpoint by validation CIDEr used **LR = 1e-6** and was selected after **epoch 1**.

### Training duration and learning rate

For the 100K-annotation experiments, learning rates of **1e-6, 2e-6, 5e-6, and 1e-5** were explored. Three-epoch runs were also performed at 1e-6 and 5e-6. Training loss continued to decrease, but the highest validation CIDEr occurred after epoch 1 in both three-epoch runs. This suggests that additional optimization did not translate into improved validation caption quality under the tested setup; it does not, by itself, prove overfitting.

The original checkpoint may already have substantial exposure to COCO caption data, which limits what can be inferred from additional fine-tuning on this dataset.

## Paired caption analysis

We compared the original pretrained BLIP predictions with the best selected fine-tuned model (**100K annotations, LR 1e-6, best checkpoint from a three-epoch run**) on all 5,000 test images.

| Measure | Result |
|---|---:|
| Paired test images | 5,000 |
| Changed captions | 1,505 (30.1%) |
| Unchanged captions | 3,495 (69.9%) |
| Improved best-reference unigram F1 | 679 (13.58%) |
| Regressed best-reference unigram F1 | 691 (13.82%) |
| Unchanged best-reference unigram F1 | 3,630 (72.60%) |
| Mean change in best-reference unigram F1 | -0.00096 |
| Paired bootstrap 95% confidence interval | [-0.00321, 0.00121] |

**Important:** Best-reference unigram F1 is a lightweight *lexical-overlap diagnostic*, not an official COCO metric. The bootstrap interval includes zero, so this diagnostic does not provide clear evidence of an average improvement or regression. It is **not** a confidence interval for CIDEr.

## Qualitative analysis

We selected **10 higher-F1 and 10 lower-F1 cases** for manual visual review. Each group was sampled with seed 42 from the 100 largest corresponding F1 changes. This is an **illustrative, score-conditioned sample**, not a representative random sample of the entire test set.

Human inspection found examples of:

- **Better specificity:** more precise object identity, object counts, spatial relationships, or actions.
- **Information omission:** dropped objects, actions, or background context.
- **Counting and category mistakes:** descriptions that became less faithful to the image.
- **Lexical metric limitations:** captions that remained visually plausible despite a lower word-overlap score.

The 20 initial human judgments were: 6 improved, 4 regressed, 1 slightly regressed, 7 similar, and 2 ambiguous. These are exploratory labels, not a blinded multi-annotator evaluation.

If included in this repository, the comparison sheets can be viewed here:

- [Higher-F1 caption examples](figures/improvements_contact_sheet.jpg)
- [Lower-F1 caption examples](figures/regressions_contact_sheet.jpg)
- [20 annotated cases](results/selected_20_with_human_judgment.csv)

## Repository structure

```text
.
├── README.md
├── scripts/                  # Fine-tuning and caption analysis scripts
├── slurm/                    # SLURM batch job configurations
├── results/                  # Summary metrics and qualitative annotations
├── figures/                  # Selected visual comparisons
├── configs/                  # Optional sanitized experiment configurations
└── docs/                     # Optional experiment notes
```

This repository contains **project-specific experiment scripts**, not a replacement for the upstream BLIP implementation.

## Reproducing the experiments

### 1. Obtain the upstream BLIP implementation

Clone the original [Salesforce BLIP repository](https://github.com/salesforce/BLIP) and follow its installation and COCO data-preparation instructions. Obtain the original BLIP captioning checkpoint separately.

The training scripts in `scripts/` import modules such as `models.blip`, `data`, and `train_caption` from the **upstream BLIP repository**. They should be executed from that repository's root, or adapted to an equivalent Python import path.

### 2. Prepare the dataset and configuration

Configure the COCO image directory, Karpathy train/validation/test annotation files, and COCO ground-truth evaluation files as expected by the upstream BLIP dataset loader.

Some experimental scripts were originally run with cluster-specific default paths. **Replace these defaults with paths in your own environment**, including `--config`, `--pretrained`, and `--output_dir`. The example below assumes the scripts have been copied or linked into the upstream BLIP working tree.

### 3. Fine-tune

From the upstream BLIP root, an example single-GPU command is:

```bash
python train_blip_100k_3ep_lr1e6.py \
  --config /path/to/caption_config.yaml \
  --pretrained /path/to/model_base_caption_capfilt_large.pth \
  --output_dir /path/to/experiment_output \
  --train_samples 100000 \
  --lr 1e-6 \
  --epochs 3 \
  --batch_size 32 \
  --seed 42
```

On SLURM, adapt the scripts in `slurm/` to your cluster's partition, account, environment, and filesystem paths before submitting with `sbatch`.

### 4. Analyze paired predictions

After producing two JSON prediction files with records of the form `{"image_id": 123, "caption": "..."}`, run the caption comparison script:

```bash
python scripts/analyze_blip_captions.py \
  --baseline /path/to/baseline_test.json \
  --finetuned /path/to/finetuned_test.json \
  --references /path/to/coco_karpathy_test_gt.json \
  --output_dir /path/to/caption_analysis
```

Use the **matching COCO Karpathy test ground-truth file**. A successful run should match references for all 5,000 images; a zero-match result indicates an annotation-format or image-ID problem.

### 5. Select qualitative examples

```bash
python scripts/sample_blip_qualitative.py \
  --analysis_dir /path/to/caption_analysis \
  --annotation /path/to/coco_karpathy_test.json \
  --image_root /path/to/coco/images \
  --output_dir /path/to/qualitative_output \
  --n 10 --pool 100 --seed 42
```

This produces caption comparison contact sheets and a CSV of selected examples. The human-judgment columns in `results/selected_20_with_human_judgment.csv` were added separately after visual review.

## Limitations

1. The original pretrained checkpoint was already trained for captioning and may have seen COCO data; this is **additional fine-tuning**, not training from scratch.
2. The 100K training subset was selected by annotation order and should not be described as a random 100K-image sample.
3. The unigram F1 analysis measures lexical similarity and does not fully capture visual correctness or semantic equivalence.
4. The 20-case human review is small, selectively sampled, and exploratory.
5. Results reflect the tested configurations and do not establish that BLIP fine-tuning is ineffective in general.

## References and acknowledgments

- [Salesforce BLIP](https://github.com/salesforce/BLIP) — upstream model and implementation.
- [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086) — original BLIP paper.
- [MS COCO](https://cocodataset.org/) — image-captioning dataset.

The original BLIP implementation, pretrained checkpoints, and COCO images are **not** authored by this project. Please follow the upstream repositories' and dataset providers' licenses and attribution requirements.
