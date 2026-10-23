# PoRef-MIL: QSM Classification with Iterative Relabeling and Posterior Decision Modeling

**English** | [简体中文](README.zh-CN.md)

This repository implements PoRef-MIL for QSM-based MRI classification with optional multimodal ablations, combining global–local feature extraction, axial self-attention, iterative slice relabeling, and a Posterior Decision Module (PDM). It supports PD/HC and MS/HC experiments.

## 🔬 Abstract

Patient-level disease labels do not describe every MRI slice. PoRef-MIL first extracts slice features using a multibranch Transformer/CNN architecture, models context along the Z axis, and updates training-slice supervision over multiple relabeling rounds. PDM then applies a single forward–backward pass to the ordered slice probabilities and combines Shannon entropy, confidence, and sliding windows to produce patient-level classifications and continuous scores.

Main components:

- **Multimodal MRI**: QSM, T1w, and other preregistered modalities specified by the experiment.
- **Global–Local Feature Extraction**: four multibranch Transformer/CNN blocks fuse features through cross-attention.
- **Iterative Relabeling**: slice predictions for training patients are generated across successive training rounds.
- **Posterior Decision Module**: bidirectional posteriors and confidence-weighted window averages determine patient-level decisions.
- **Age-conditioned FiLM**: sin/cos age encoding and an MLP generate scaling and offset parameters to modulate inputs or features before or after feature extraction.

## 🏆 Results

PoRef-MIL achieves an AUC of **0.942±0.002** and an ACC of **0.926±0.008** on PD, and an AUC of **0.864±0.0029** and an ACC of **0.825±0.018** on MS. See [Table 2](#table-2) for the full comparison and statistical protocol.

## 🏗️ Architecture

### Network Components

1. **Feature Extraction**: four multibranch blocks per modality combine global attention features and local convolutional features.
2. **Multimodal Fusion**: cross-modal cross-attention by default, with alternative fusion methods for ablation studies.
3. **Axial Self-Attention**: models slice context only in the relabeler; the diagnostic test network has no axial module.
4. **Slice Classification**: an MLP outputs a logit per slice; Sigmoid produces disease probabilities.
5. **Iterative Relabeling**: initialize diagnosis, then alternate a separate bag relabeler with freshly initialized diagnostic networks. Only the trained diagnostic encoder initializes the next relabeler.
6. **PDM**: a single forward–backward pass, entropy-based confidence, and sliding windows produce patient-level results.

```text
Training: patient labels -> initialize diagnosis
        -> copy best encoder -> Z attention -> relabeling MLP
        -> Sigmoid slice probabilities + Softmax weights -> patient Focal Loss
        -> new training-slice labels -> fresh diagnostic training -> next round

Testing: QSM -> ordered slices -> four FEBs -> max pooling -> MLP
       -> slice probabilities -> one forward/backward pass -> entropy/confidence
       -> confidence-weighted window averages -> maximum window score -> patient diagnosis
```

### Key Features

- Patient-level separation of training, validation, and test sets, with checks for overlapping patients and image paths.
- One fixed independent test set; five training/validation folds within the development set; five models evaluated separately.
- Single-, dual-, and configurable multimodal inputs.
- Support for 128-slice PD sequences, 96-slice MS sequences, and other explicitly specified lengths.
- Ablations for K=1–5 total diagnostic training rounds, fusion methods, age-conditioning positions, and PDM.
- Exported patient lists, models, pseudo-labels, slice probabilities, posteriors, ROC data, and per-fold metrics.

## 📁 Project Structure

```text
PoRef-MIL/
├── feature_ex/
│   ├── upper.py                 # Linear tokenization and Transformer branch
│   ├── middle.py                # Tokens to 2D feature maps
│   ├── lower.py                 # CNN and DenseFeatureConvBlock
│   ├── crossAttention.py        # Multihead cross-attention
│   ├── feature.py               # Four-block feature extraction backbone
│   └── film.py                  # Sin/cos age encoding and FiLM
├── classifier/
│   ├── classifier_model.py      # Shared MLP classification head
│   ├── classifierNet.py         # Legacy single-modal slice network
│   └── postProcess.py           # PDM compatibility entry point
├── relabel/
│   ├── selfAttention.py         # Chunked Z-axis self-attention
│   ├── relabelingNet.py         # Single-modal relabeling reference network
│   ├── relabelTrain.py          # Delegates to the unified training entry point
│   └── relabelTrainPP.py        # Delegates to the unified training entry point
├── poref/
│   ├── cli.py                   # Argument parsing, validation, and dispatch
│   ├── data.py                  # Patient split checks and ordered multimodal loading
│   ├── model.py                 # Main PoRef-MIL model and fusion ablations
│   ├── experiment.py            # Five-fold, K-round training and evaluation
│   ├── pdm.py                   # Forward–backward inference and window decisions
│   ├── postprocess.py           # Standalone PDM for slice-probability CSVs
│   ├── inference.py             # Unlabelled-patient inference with the main model
│   ├── preprocessing.py         # Physical-grid resampling of registered images
│   └── metrics.py               # Metrics, ROC data, and mean/std summaries
├── examples/
│   └── splits.template.json     # Fixed test set and five development folds
├── tests/                       # Module and workflow tests
├── train_val.py                 # Training script entry point
├── requirements.txt             # Dependencies
├── pyproject.toml               # Installation and poref command configuration
├── DESIGN_IMPLEMENTATION.md     # Implementation decisions and boundaries (Chinese)
├── README.zh-CN.md              # Chinese documentation
└── README.md                    # English documentation (default)
```

The tree lists files relevant to the main workflow. Legacy slice-training scripts retained in the repository are not the complete PoRef-MIL experiment. The commands below use `--mode poref`.

## 🚀 Getting Started

### Prerequisites

- Python **3.10 or later**.
- PyTorch **2.4 or later**; GPU training requires a compatible PyTorch/CUDA environment.
- NumPy, NiBabel, SciPy, scikit-learn, pandas, and tqdm.

Run the following in the project root:

```powershell
# Create and activate an environment (example)
conda create -n poref-mil python=3.10
conda activate poref-mil

# Configure PyTorch for your device, then install the project and dependencies
python -m pip install -e .

# Inspect the commands
python train_val.py --help
python -m poref --help
```

After installation, `poref train`, `poref predict`, and `poref postprocess` are also available.

### Data Preparation

#### 1. Image Modalities

| Experiment | Input modalities | Labels |
|---|---|---|
| Single-modal PD | QSM or T1w | HC=0, PD=1 |
| Dual-modal PD | QSM + T1w | HC=0, PD=1 |
| MS | T1w, FLAIR, QSM, etc., as specified by the experiment | HC=0, MS=1 |

Names supplied through `--modalities` must exactly match the keys under `images` in the split file. Modality order is part of the model configuration.

#### 2. Data Format and Preprocessing

- Input: NIfTI (`.nii` or `.nii.gz`).
- Each modality: `[1,256,256]` per slice and `[Z,1,256,256]` per sequence. The main model stacks M modalities as `[Z,M,1,256,256]` and outputs Z slice logits.
- A common PD volume shape is `[256,256,128]`; MS may use 96 slices.
- `slice_axis=2` corresponds to `[256,256,Z]`; `slice_axis=0` corresponds to `[Z,256,256]`.
- Modalities for a patient must already be spatially registered and share the same 3D shape and affine. The loader checks grid consistency; it does not register images or assess anatomical registration quality.
- Volume-level Z-score normalization is applied separately to each patient and modality, preserving the original Z order.

To change the grid of preregistered images, write to a new output directory:

```powershell
python -m poref preprocess --input-dir "D:/MRI/registered" --output "D:/MRI/resampled" --shape 256 256 128
```

This command performs linear resampling and updates the affine. The output directory must be empty and outside the input directory tree; source images are preserved. Use `--reference` to specify a shared reference grid. Resampling does not replace spatial registration.

#### 3. Fixed Test Set and Five-Fold Development Splits

The protocol is: **reserve one fixed independent test set first, then perform five-fold training/validation within the remaining development set**.

| Model | Training subsets | Validation subset | Test set |
|---|---|---|---|
| Model 1 | B + C + D + E | A | Fixed test set T |
| Model 2 | A + C + D + E | B | Fixed test set T |
| Model 3 | A + B + D + E | C | Fixed test set T |
| Model 4 | A + B + C + E | D | Fixed test set T |
| Model 5 | A + B + C + D | E | Fixed test set T |

A–E are patient-level partitions of the development set. Each model selects its weights using its validation subset, then evaluates the same T after the final training round. Test patients never participate in training.

The main training entry requires `fixed_holdout` with exactly five development train/validation folds, each evaluated on the same independent test set. Other split protocols are rejected by this entry.

#### 4. Patient Split File

Copy and edit `examples/splits.template.json` with actual patient IDs and paths. The template contains 10 development patients and 2 fixed test patients solely to illustrate the schema; **these are not the actual experimental data or cohort sizes**.

The split file uses JSON. The following illustrates fields only and is not a complete training manifest:

```json
{
  "test_design": "fixed_holdout",
  "samples": [
    {
      "patient_id": "subject_id",
      "label": 1,
      "images": {
        "QSM": "D:/MRI/QSM/subject.nii",
        "T1w": "D:/MRI/T1w/subject.nii"
      },
      "slice_axis": 2,
      "age": 65
    }
  ],
  "folds": [
    {
      "train": ["training_patient_ids"],
      "val": ["validation_patient_ids"],
      "test": ["fixed_test_patient_ids"]
    }
  ]
}
```

Requirements:

- Include every patient and all five folds. In each fold, train/val/test must cover all patients without overlap.
- Validation subsets must be mutually exclusive and cover the development set exactly once; all five test sets must be identical.
- Every subset must include both classes. The same image must not be reused across subsets under different patient IDs.
- Image paths may be absolute or relative to the JSON file.
- Age experiments require `age` for every patient; other experiments may omit it.

### Training

The following PowerShell command trains the QSM model with window length 5 and decision threshold 0.6. Supply the paths and window parameters used for the experiment. Select window parameters using training or validation data; use the test set for final evaluation.

```powershell
# QSM PoRef-MIL: fixed test set, five development folds, K=5
python train_val.py `
  --mode poref `
  --data_split_file "D:/splits/pd.json" `
  --modalities QSM `
  --save_dir "D:/results/pd_k5" `
  --batch_size 1 `
  --slice-batch-size 4 `
  --learning_rate 0.0001 `
  --num_epochs 300 `
  --num_folds 5 `
  --rounds 5 `
  --window-length 5 `
  --window-threshold 0.6 `
  --expected-slices 128 `
  --seed 2025 `
  --std-ddof 1 `
  --device auto
```

The same command can be written as `poref train ...`. Adding `--dry-run` only previews resolved arguments: it does not import the model, read images, start training, or validate actual files and split contents.

`--batch_size` is the number of **patients** accumulated per optimization step; each ordered patient sequence is processed separately for forward/backward passes. `--slice-batch-size` controls the number of slices processed at a time inside FE. Use a different empty output directory for each independent experiment.

To check the expected fixed test-set class counts, optionally add:

```text
PD: --expected-test-counts 26 31
MS: --expected-test-counts 20 20
```

The order is **HC, disease**. These options only check test-set counts; they do not describe the full cohort or create/change patient assignments.

### Testing

#### Evaluation during Training

After each model completes its final diagnostic training round, the workflow evaluates its validation subset and the fixed test set. Example output:

```text
pd_k5/
├── config.json
├── resolved_splits.json
├── fold_1_test_results.csv
├── fold_1_val_results.csv
├── fold_1/
│   ├── split.json
│   ├── best_round_1.pth
│   ├── best_round_5.pth
│   ├── best_relabel_for_round_2.pth
│   ├── best_relabel_for_round_5.pth
│   ├── pseudo_labels_for_round_2.json
│   ├── training_log.json
│   ├── patient_predictions.json
│   └── metrics.json
├── fold_2/ ... fold_5/
└── summary.json
```

The tree omits corresponding files from other rounds and folds. `summary.json` summarizes all five models on the same test set, records one independent test cohort, and indicates that predictions were not ensembled.

#### Prediction on Unlabelled Patients

```powershell
python -m poref predict `
  --checkpoint "D:/results/pd_k5/fold_1/best_round_5.pth" `
  --patients-file "D:/splits/unlabelled.json" `
  --output "D:/results/predictions_model1" `
  --device auto
```

`unlabelled.json` only needs a `samples` list. Each patient supplies `patient_id`, `images`, and `slice_axis`; age-conditioned models also require `age`. Labels and folds are unnecessary. Prediction restores the model, modalities, round, FiLM, and PDM configuration from the checkpoint and writes `predictions.json`.

This command loads one model at a time. To evaluate five models separately, use five checkpoints and separate output directories.

#### Standalone PDM

Apply the same PDM to existing ordered slice probabilities:

```powershell
python -m poref postprocess `
  --input-dir "D:/results/slice_predictions" `
  --output "D:/results/pdm" `
  --folds 5 `
  --window-length 5 `
  --window-threshold 0.6 `
  --std-ddof 1
```

Input CSVs require `patient_name,true_label,slice_probs`. For each model, the command recognizes `fold_N_test_results.csv` or `fold_N_test_slice_results.csv`; if both exist for a model, it rejects the ambiguous input. If `z_indices` is present, it must specify the original contiguous slice order.

### Key Parameters

| Parameter | Default / requirement | Description |
|---|---|---|
| `--mode` | Required | Use `poref` for the complete workflow |
| `--data_split_file` / `--split-file` | Required for the main workflow | Patient-level split JSON |
| `--save_dir` / `--output` | Required | New, empty output directory |
| `--modalities` | `QSM` | Ordered modality names, e.g. `QSM T1w` |
| `--batch_size` | `8` | Patients accumulated per optimization step; the example explicitly uses 1 |
| `--slice-batch-size` | `4` | Slice batch size inside FE |
| `--learning_rate` | `0.0001` | First-round learning rate; proportionally scales the default round schedule |
| `--num_epochs` | `300` | Maximum epochs for each diagnostic or relabeling training stage |
| `--num_folds` | `5` | The main workflow requires a fixed test set and five development train/validation folds |
| `--rounds` | `5` | K diagnostic training rounds including initialization; K−1 relabeling stages |
| `--round-lrs` | Default round schedule | Explicit list of exactly K learning rates |
| `--patience` | `50` | Early-stopping patience based on validation AUC |
| `--weight-decay` | `0.00001` | Adam weight decay |
| `--alpha` / `--gamma` | `0.5` / `2.0` | Focal Loss parameters |
| `--relabel-threshold` | `0.5` | Threshold for discrete training-slice pseudo-labels |
| `--fusion` | `cross-attention` | Fusion method |
| `--age-position` | `none` | `none`, `pre`, or `post` |
| `--supervision` | `iterative` | Iterative supervision or `traditional-mil` mean-supervision ablation |
| `--aggregation` | `pdm` | Patient-level PDM or `mean` aggregation |
| `--window-length` | Required for PDM | Sliding-window length L; cannot exceed the slice count |
| `--window-threshold` | Required for PDM | Patient decision threshold T_w |
| `--stay-hc` / `--stay-disease` | `0.9` / `0.9` | State self-transition probabilities |
| `--prior-hc` | `0.5` | Initial HC probability for PDM |
| `--expected-slices` | Unrestricted | Check sequence length, e.g. 128 or 96 |
| `--expected-test-counts` | Unrestricted | Check HC and disease test counts for each model |
| `--disease-name` | `PD` | Record `PD` or `MS` in the experiment configuration |
| `--seed` | `2025` | Seed for the first model; incremented by 1 for each subsequent model |
| `--std-ddof` | `1` | Sample standard deviation; explicitly select 0 for population standard deviation |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--dry-run` | Off | Preview resolved parameters only |

Both state self-transition probabilities default to 0.9, and the initial HC probability defaults to 0.5. Command-line arguments configure these values. Window length and decision threshold are required at runtime and are selected using training or validation data.

## 🔧 Model Architecture Details

### 1. Global–Local Feature Extraction

Each modality passes through four sequential Extraction Blocks:

- **Upper branch**: linear row tokenization produces 256 tokens of dimension 256, followed by two groups of LayerNorm, self-attention, MLP, and residual connections.
- **Middle branch**: a projection branches after the first attention feed-forward block; LayerNorm, reshaping and a 1×1 convolution produce a `[128,16,16]` feature map.
- **Lower branch**: four groups of 3×3 convolution, BatchNorm, ReLU, and MaxPool; concatenation with the middle branch, followed by DenseFeatureConvBlock processing and channel transformations.
- **Cross-Attention**: the upper branch supplies Q and the lower branch K/V. Normalization and multihead cross-attention are followed by a residual connection, LayerNorm, a GELU feed-forward MLP and a second residual connection. Output features have shape `[256,256]`.

The CNN branches in all four blocks receive the original image. Later blocks receive token inputs from the preceding block.

### 2. Axial Self-Attention

Only in the relabeling network, Z-axis attention exchanges context across all Z slices. A `1→64` projection maps scalar features to embeddings, followed by 8-head self-attention, residual connections, a `64→128→64` feed-forward network, and projection back to scalars. Input and output shapes are `[1,Z,N,256]`, where N=256. The MLP-level ablation repeats each component of its fused vector over the last dimension to preserve this shape. The implementation chunks independent feature positions while preserving the entire Z sequence and uses gradient checkpointing during training to control intermediate activation memory.

### 3. Slice Classifier

The diagnostic head receives FEB features; the relabeling head receives Z-attention features. Both max-pool each token over its feature dimension (`[Z,256,256] -> [Z,256]`) and use independently learned copies of the same MLP:

```text
256 → 512 → 1024 → 512 → 256 → 128 → 2 → 1
```

ReLU follows each hidden linear layer. The final layer outputs a logit, converted to a disease probability by Sigmoid. The MLP-level ablation repeats each vector component over the last dimension, so max pooling preserves the fused vector.

### 4. Iterative Relabeling

![Figure 1: PoRef-MIL architecture and iterative slice relabeling](docs/figures/figure-1-framework.png)

**Figure 1 | Architecture and iterative relabeling.** Four feature-extraction blocks produce slice features. Z-axis attention incorporates slice context to generate predicted labels, which update supervision in subsequent training rounds. [Original PDF](docs/figures/figure-1-framework.pdf).

1. **Round 1**: initialize slice labels from training-patient labels and train diagnosis (four FEBs -> max pooling -> MLP). Select its checkpoint using patient validation AUC.
2. **Relabeling for rounds 2–K**: copy the preceding best diagnostic encoder into a separate relabeler, with fresh Z attention and MLP parameters. The encoder remains trainable. The same slice logits feed Sigmoid probabilities and Softmax slice weights. Their dot product is the patient probability used for patient-label Focal Loss. Select the relabeler using attention-aggregated patient validation AUC.
3. Load the best relabeler and generate labels only for training slices using Sigmoid probabilities (default threshold 0.5).
4. Initialize an entirely new diagnostic network and train with the new slice labels, averaging slice Focal Loss within each patient. Diagnostic validation uses the configured patient aggregation (PDM by default). Its best encoder initializes the next relabeler. Test inference uses only the final diagnostic network and patient aggregation.

`--rounds 5` means initialization plus four relabeling/diagnosis cycles. Each stage resets Adam, ReduceLROnPlateau and early stopping. Both stages within a round use that round's learning rate and the same epoch/patience settings. Diagnosis and relabeling have separate training stages and checkpoints.

Validation and test evaluation always use actual patient labels and never enter pseudo-label updates. Slice probabilities are discretized using `>= --relabel-threshold` (default 0.5). With K=5 and `--learning_rate=1e-4`, the initial learning rates are:

```text
[1e-4, 5e-5, 3e-5, 1e-5, 5e-6]
```

For K<5, use the first K entries. Changing `--learning_rate` proportionally scales this schedule. `--round-lrs` overrides it with exactly K explicit values. The optimizer and scheduler are recreated for each stage; the best validation weights are loaded before proceeding to the next round.

### 5. Posterior Decision Module

![Figure 2: Bidirectional posterior refinement and patient-level decisions](docs/figures/figure-2-pdm.png)

**Figure 2 | Posterior decision workflow.** Ordered slice probabilities enter a two-state HC/disease model. One forward–backward pass produces bidirectional posteriors, which are combined with entropy-based confidence and sliding windows for patient-level classification. State connections show transition relationships; the implementation does not perform optimal-path decoding. [Original PDF](docs/figures/figure-2-pdm.pdf).

Apply **one** forward–backward pass to the ordered slice disease probabilities to obtain disease posteriors q_i.

```math
H_i=-q_i\log_2q_i-(1-q_i)\log_2(1-q_i)
```

```math
C_i=1-H_i
```

For each complete window of length L, confidence weights the posterior average:

```math
W_j=\frac{\sum_{i=j}^{j+L-1} C_iq_i}{\sum_{i=j}^{j+L-1} C_i}
```

```math
S=\max_j W_j
```

Windows move by one slice without padding. A zero-confidence-sum window has no defined score, is exported as `null`, and is excluded from the maximum. If every window is invalid, prediction raises an explicit error rather than assigning a healthy diagnosis.
Predict disease when `S > T_w`; otherwise predict HC. S is the patient ROC-AUC score. Validation, testing, predict and standalone postprocessing use the same formula.
PDM does not weight hard slice labels, add a low-pass filter, decode a Viterbi path or repeat forward–backward passes.

#### Slice-level Visualization

![Figure 4: Representative slices and continuous abnormal intervals identified by PDM](docs/figures/figure-4-visualization.png)

**Figure 4 | Slice and sequence decision examples.** The upper and lower groups show representative images, abnormality probabilities, and PDM-marked continuous abnormal intervals in 128-slice and 96-slice sequences, respectively, illustrating the relationship between local images and sequence decisions. [Original PDF](docs/figures/figure-4-visualization.pdf).

The heatmaps are supplied paper illustrations. The legacy heatmap script broadcasts a slice probability over the entire slice and cannot directly generate equivalent pixel-level heatmaps.

### 6. Age-conditioned FiLM

Age is introduced through **Sinusoidal Encoding (sin/cos encoding) + FiLM**, implemented in `feature_ex/film.py`. The application position is controlled by `--age-position` in `poref/model.py`.

#### Sin/Cos Age Encoding

Let a be patient age in years and d=256 the embedding dimension. Scale age as u=a/100, then use 128 fixed frequencies and interleave sin and cos values to form a 256-dimensional vector:

```math
u = \frac{a}{100},\qquad
\omega_j = \exp\!\left(-\frac{j\ln 10000}{128}\right),\qquad j=0,\ldots,127
```

```math
e_{2j}(a)=\sin(u\omega_j),\qquad
e_{2j+1}(a)=\cos(u\omega_j)
```

The order is `[sin(uω₀), cos(uω₀), …, sin(uω₁₂₇), cos(uω₁₂₇)]`. Frequencies are fixed and not trained. Dividing age by 100 is a fixed scaling operation, not normalization using dataset means or variances.

#### FiLM Parameter Generation and Modulation

A trainable MLP maps the age embedding to C scaling parameters and C offsets:

```text
256 → Linear(256, 256) → GELU → Linear(256, 2C) → split(γ, β)
```

```math
[\gamma(a),\beta(a)] = \mathrm{MLP}(e(a)),\qquad
\widetilde{F} = (1+\gamma(a))\odot F + \beta(a)
```

γ and β modulate channels and broadcast over the other dimensions. Each patient's age embedding is reused across all their slices. The last MLP layer's weights and biases are initialized to zero, giving γ=β=0 and an initial identity transformation. The MLP following the encoding is trained jointly with the classification task.

| Configuration | Actual application order | Modulated channels C |
|---|---|---|
| `--age-position none` (default) | No age encoding or FiLM | Not applicable |
| `--age-position pre` | Volume normalization → Sin/cos encoding generates γ and β → Modulate input slices per modality → FE → Modality fusion → Classifier (Z attention precedes the head only in the relabeler) | Number of input modalities M |
| `--age-position post` | FE → Modality fusion → Modulate features with γ and β generated from sin/cos age encoding → Classifier (Z attention precedes the head only in the relabeler) | Feature dimension 256 |

Pre-FE uses one scale and offset per input modality, broadcast over slice pixels. Post-FE modulates the 256 feature channels and broadcasts across tokens. With decision-level fusion, the same FiLM is applied separately to each modality's features before classification (with axial attention only in the relabeler); probabilities are then averaged.

Pre-FE and Post-FE are trained and evaluated independently. Each age-conditioned model has only one shared FiLM and does not enable both positions simultaneously. Every training, validation, test, and inference patient must supply a finite `age` in the range 0–120 years. The default training example uses `none`. For age experiments, add `--age-position pre` or `--age-position post` to the full training command and use a separate output directory per experiment.

### Loss Function

Diagnosis computes Binary Focal Loss per slice and averages it within each patient. Relabeling aggregates slice probabilities before computing patient-label Focal Loss. Both use `alpha=0.5` and `gamma=2.0`.

Each stage creates its own Adam optimizer, with weight decay `1e-5` and a maximum of 300 epochs. ReduceLROnPlateau monitors validation AUC with factor 0.5, patience 10 and minimum learning rate `1e-7`. A checkpoint is updated when validation AUC exceeds the saved best value by more than `1e-4`; early stopping follows 50 epochs without that improvement. Runtime arguments configure the exposed parameters; scheduler settings are specified in the training code.

Relabeling validation AUC uses the attention-aggregated patient probability. Diagnostic validation AUC uses the maximum confidence-weighted PDM window score. The `--aggregation mean` ablation uses mean slice probability instead.

The mean-MIL ablation uses the same backbone, averages slice probabilities into a patient prediction, and computes Focal Loss against actual training-patient labels.

## 📊 Experimental Results

### Evaluation Protocol

The workflow evaluates five training/validation models separately on the same fixed test patients. It records the following. `--std-ddof` below is a runtime configuration; Table 2 preserves the author-reported values without recalculation:

- **AUC, ACC, SEN, SPE, PRE, F1**, and ROC curve data.
- **TP, FN, FP, TN**, with confusion-matrix order `[[TP,FN],[FP,TN]]`.
- Per-model patient lists, true labels, continuous scores, predictions, and slice probabilities.
- Mean ± standard deviation using a consistent `--std-ddof`; the default is `ddof=1`.

The standard deviation describes variation across five models on the same test set, not variation across five independent test cohorts. Undefined metrics are saved as `null`.

### Performance Comparison

<a id="table-2"></a>

**Table 2. Comparison of the Proposed PoRef-MIL with Other Methods on PD and MS Datasets**

The following are actual experimental results supplied by the author, with numerical values preserved from the original Table 2.

#### PD Dataset

| Method | AUC | ACC | SEN | SPE | PRE | F1 Score |
|---|---|---|---|---|---|---|
| AG-SE-ResNeXt | 0.893±0.002 | 0.846±0.015 | 0.858±0.018 | 0.831±0.021 | 0.858±0.016 | 0.858±0.014 |
| CAAE-CRATNet | 0.901±0.002 | 0.867±0.010 | 0.865±0.014 | 0.869±0.021 | 0.888±0.015 | 0.876±0.009 |
| MMRNNet | 0.899±0.002 | 0.860±0.012 | 0.832±0.014 | 0.892±0.032 | 0.903±0.025 | 0.866±0.010 |
| DenseFormer-MoE | 0.906±0.001 | 0.856±0.015 | 0.813±0.014 | <ins>0.908±0.021</ins> | 0.913±0.019 | 0.860±0.014 |
| DA-MIDL | 0.914±0.002 | 0.867±0.010 | 0.865±0.014 | 0.869±0.021 | 0.888±0.015 | 0.876±0.009 |
| TransMIL | <ins>0.927±0.001</ins> | <ins>0.898±0.008</ins> | <ins>0.890±0.018</ins> | <ins>0.908±0.021</ins> | <ins>0.920±0.016</ins> | <ins>0.905±0.008</ins> |
| PoRef-MIL | **0.942±0.002** | **0.926±0.008** | **0.923±0.018** | **0.931±0.017** | **0.941±0.014** | **0.932±0.008** |

#### MS Dataset

| Method | AUC | ACC | SEN | SPE | PRE | F1 Score |
|---|---|---|---|---|---|---|
| AG-SE-ResNeXt | 0.774±0.0029 | 0.735±0.014 | 0.780±0.027 | 0.690±0.022 | 0.716±0.013 | 0.746±0.015 |
| CAAE-CRATNet | 0.805±0.0033 | 0.760±0.014 | 0.780±0.027 | 0.740±0.022 | 0.750±0.014 | 0.765±0.015 |
| MMRNNet | 0.795±0.0027 | 0.745±0.021 | 0.770±0.027 | 0.720±0.027 | 0.734±0.022 | 0.751±0.021 |
| DenseFormer-MoE | 0.778±0.0025 | 0.730±0.011 | 0.760±0.022 | 0.700±0.035 | 0.718±0.020 | 0.738±0.009 |
| DA-MIDL | 0.803±0.0025 | 0.760±0.014 | 0.790±0.022 | 0.730±0.027 | 0.746±0.017 | 0.767±0.013 |
| TransMIL | <ins>0.829±0.0029</ins> | <ins>0.795±0.011</ins> | <ins>0.820±0.027</ins> | <ins>0.770±0.027</ins> | <ins>0.781±0.017</ins> | <ins>0.800±0.012</ins> |
| PoRef-MIL | **0.864±0.0029** | **0.825±0.018** | **0.850±0.035** | **0.800±0.035** | **0.810±0.025** | **0.829±0.018** |


**Statistical note:** Values are mean ± standard deviation across five models trained using five-fold training/validation within the development set. All five models were evaluated separately on the same fixed test set. Bold and underlined values indicate the highest and second-highest means within each dataset, respectively; tied means receive the same formatting.

PoRef-MIL obtains the highest mean for all six reported metrics on both datasets. PD AUC and ACC are **0.942±0.002** and **0.926±0.008**; MS AUC and ACC are **0.864±0.0029** and **0.825±0.018**.

AUC: area under the ROC curve; ACC: accuracy; SEN: sensitivity; SPE: specificity; PRE: precision; F1 Score: harmonic mean of precision and sensitivity. Comparison methods are included to report baseline results.

After running the project, inspect `summary.json` for that run's aggregated metrics; `status: complete` marks completion. Table 2 retains the paper-reported values and is not generated by invocation examples or software tests.

### Ablation Studies

| Experimental factor | Configuration |
|---|---|
| Single/dual modality | `--modalities T1w`, `QSM`, `QSM T1w` |
| Fusion method | `--fusion decision-level`, `mlp-level`, `concatenation`, `cross-attention` |
| Total diagnostic training rounds | Independently train `--rounds 1` through `--rounds 5` |
| Without PDM | `--aggregation mean`, mean probability with threshold 0.5 |
| Initial labels versus iterative updates | `--rounds 1` versus multiround `iterative` configurations |
| Mean-MIL supervision | `--supervision traditional-mil --rounds 1` |
| Age-conditioning position | `--age-position pre` versus `--age-position post` |

Fusion definitions: **Decision-level** averages probabilities after each modality passes through the shared classification head (axial attention is used only by the relabeler). **MLP-level** pools each modality's tokens, concatenates the vectors, and fuses them with an MLP. **Concatenation** concatenates token features and linearly projects them to a lower dimension. **Cross-Attention** fuses modalities through attention: for two modalities, the first supplies Q and the second supplies K/V to a shared cross-attention computation; its result is added residually to each modality and normalized, followed by a second cross-attention computation with residual addition and normalization. For more than two modalities, fusion proceeds sequentially in `--modalities` order.

Train K=1–5 ablations independently with the same actual patient splits and seed configuration, using separate output directories. The main PDM experiment also saves mean-aggregation metrics from the same final model and the same slice predictions to isolate the postprocessing effect.

## 🧠 Prediction Interpretability

The main workflow exports original slice probabilities, bidirectional posteriors, Shannon entropy, confidence, confidence sums, all window scores, validity flags, and the starting index of the highest-scoring window. These outputs show which continuous slices contribute to a patient decision.

Interpretation is at the **slice-sequence level**. The legacy heatmap script broadcasts probabilities over entire slices rather than localizing pixel-level lesions. It only supports the old slice-model architecture and cannot load checkpoints from the new main model.

## 📂 Additional Resources

- [Patient-level five-fold split template](examples/splits.template.json)
- [Implementation decisions and boundaries (Chinese)](DESIGN_IMPLEMENTATION.md)
- [Dependencies](requirements.txt)
- [Installation configuration](pyproject.toml)
- [Chinese README](README.zh-CN.md)

## Figure-aligned checkpoint format

Experiment checkpoints use `format_version=2` and `role=diagnosis` or `role=relabel`.
`predict` accepts only diagnostic checkpoints. Earlier combined axial/FE1/FE2 checkpoints
cannot represent this architecture and must be retrained. `best_round_K.pth` is the
diagnostic checkpoint; `best_relabel_for_round_K.pth` initializes pseudo-label generation.
The main entry is `poref train --mode poref`; legacy modes and legacy heatmap weights
are separate compatibility interfaces, not the complete two-stage training workflow.
