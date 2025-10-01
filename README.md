# Ariel Conformer: Mu Prediction from AIRS + FGS

This repository contains a full preprocessing and modeling pipeline to predict the exoplanet transmission spectrum µ across 283 wavelengths by fusing AIRS and FGS sensor streams from the Ariel Data Challenge format. It includes robust dataset cleaning, temporal downsampling, feature tokenization, a Conformer-based encoder, and Lightning training/evaluation utilities.

Conformer model (neurips-ariel-conformer.ipynb) achieves the GLL score of 0.411, comparable to other silver medal models (late submission).  Secondary model (MLP, neurips-ariel-transit.ipynb) achieves the GLL score of 0.375 (104/861 place)

### Key features
- End-to-end data pipeline: calibration, CDS computation, hot/dead masking, linearity correction, flat-fielding, and robust temporal binning for both sensors.
- Sensor fusion: learns from AIRS (spectral) and FGS (photometric) streams with sensor-type embeddings and shared self-attention.
- Conformer encoder: macaron-style feedforward, depthwise conv blocks, and query-based wavelength head to produce µ for 283 bins.
- Reproducible training with PyTorch Lightning, mixed precision, checkpoints, and TensorBoard logging.

### Repository structure
- neurips-ariel-conformer.ipynb — main notebook with data processing, dataset/dataloader modules, model, Lightning loop, and prediction scripts.
- processed_data/ — generated arrays and artifacts, including per-observation tensors and transit_depth.csv.
- checkpoints/ — model checkpoints produced by the Trainer callbacks.
- predictions/ — saved µ predictions as CSV.
- data/ — expected input directory with train/test parquet files and CSV metadata.

## Data expectations

Place the official Ariel Data Challenge structure in ./data. The notebook expects:
- ./data/adc_info.csv
- ./data/train.csv (target spectra per planet_id for training)
- ./data/train_star_info.csv and ./data/test_star_info.csv (8 star parameters per planet_id)
- ./data/wavelengths.csv (optional; otherwise a default 283-length linspace is used)
- ./data/sigma_estimates.csv (optional; provides per-sensor sigma priors)
- For each planet_id and obs_id in train/ or test/: AIRS-CH0_signal_{obs_id}.parquet, FGS1_signal_{obs_id}.parquet, and corresponding calibration parquet files under AIRS-CH0_calibration_{obs_id}/ and FGS1_calibration_{obs_id}/ with dark, dead, flat, linear_corr matrices.

Directory sketch:
- data/
  - train/planet_id/AIRS-CH0_signal_0.parquet, FGS1_signal_0.parquet, AIRS-CH0_calibration_0/*.parquet, FGS1_calibration_0/*.parquet
  - test/planet_id/... similarly arranged for inference

## Installation

Use Python 3.10+ and install dependencies:
- pip install -r requirements.txt
- For GPU builds of torch/torchvision/torchaudio, use the official PyTorch index URL (e.g., cu121) if needed.

Required packages include numpy, pandas, scipy, astropy, pyarrow/fastparquet, tqdm, torch, torchvision, torchaudio, pytorch-lightning, tensorboard, typing-extensions, and ipywidgets. Optional: wandb for logging. See requirements.txt.

## Quickstart

Open neurips-ariel-conformer.ipynb and run the cells in order. The main configuration block controls paths and parameters:
- DATA_PATH: base dataset directory (default ./data)
- OUTPUT_DIR: where processed arrays and transit_depth.csv will be written (default ./processed_data)
- IS_TRAIN: toggles train/test folders
- SENSOR_CONFIG: shapes, binning, ROI slices, dt patterns for AIRS and FGS
- MODEL_*: phase detection slice, polynomial degree, optimization delta, etc.

### Step 1: Preprocess and compute transit depth

The ArielDataProcessor performs end-to-end cleaning and downsampling, writing fused tensors and estimating per-planet transit_depth.csv for conditioning. Example (from the notebook):
- processor = ArielDataProcessor(data_path=config["DATA_PATH"], output_path=config["OUTPUT_DIR"], is_train=config["IS_TRAIN"], n_jobs=4)
- transit_depth = processor.process_all_data(airs_factors=160, fgs_ratio=12, mode='mean')

Generated artifacts:
- processed_data/planet_{planet_id}_signal_{obs_id}_downsample_{A}.npy with shape (T, 16, 283)
- processed_data/transit_depth.csv with a single column transit_depth indexed by planet_id

### Step 2: Build DataModule

The ArielDataModule wraps the dataset, splits by planet_id for leak-free validation, and exposes train/val loaders. Parameters include batch_size, num_workers, val_split, downsample_mode (e.g., "160"), pad_to_T, and seed.

- datamodule = ArielDataModule(preprocessed_dir=config['OUTPUT_DIR'], train_csv=config['DATA_PATH'] + '/train.csv', star_info_csv=..., wavelengths_csv=..., transit_depth_csv=..., sigma_csv=..., batch_size=4, val_split=0.2, downsample_mode="160", pad_to_T=30)

### Step 3: Train the Conformer

The ArielConformerGllLightning wraps the architecture and training logic with metrics: MSE/RMSE/MAE and a GLL-style score tied to given sigma priors. Training uses AdamW, ReduceLROnPlateau, gradient clipping, mixed precision, and TensorBoard logging.[1]

- model, trainer = train_ariel_conformer(preprocessed_dir=config['OUTPUT_DIR'], data_dir=config['DATA_PATH'], batch_size=4, downsample_mode="160", max_epochs=150, learning_rate=5e-5, num_workers=0, accumulate_grad_batches=4, pad_to_T=30, lr_patience=5, input_scale=0.1)

Checkpoints are stored under ./checkpoints with val_rmse and val_gll in filenames. The notebook prints model summary and device info.

### Step 4: Predict µ

For trained checkpoints, create a DataLoader over the entire training index or test set and run inference:
- ckpts = ['./checkpoints/ariel-sensor-epoch=08-val_rmse=0.000500-val_gll=0.3594.ckpt']
- pids, mu_np = predict_mu_for_loader(ArielConformerGllLightning.load_from_checkpoint(ckpts), loader)
- save_mu_csv(pids, mu_np, './predictions/mu_predictions.csv')

The script aggregates duplicate planet observations by mean and saves planet_id-indexed CSV with 283 columns: lambda_0 ... lambda_282.

## Modeling details

- Tokenizers: CNN tokenizers for AIRS (2D) and FGS (1D) generate token grids/strips per time step; adaptive pooling controls token counts.
- Positional encodings: learnable time and token encodings to preserve structure after flattening to sequence.
- Sensor embeddings: per-sensor learned biases so the encoder can disambiguate AIRS vs FGS contributions.
- Conformer backbone: stacks of macaron feedforward, multi-head attention, and depthwise conv blocks; pre-fuse stages before the main encoder.
- Wavelength query head: a bank of learnable query vectors attends over sequence tokens to produce per-wavelength µ.
- Loss/metrics: training optimizes MSE; logs RMSE/MAE and a normalized Gaussian log-likelihood metric (GLL-like).

## Performance notes

- Mixed precision (precision='16-mixed') with gradient clipping improves throughput; set torch.set_float32_matmul_precision('high').
- Increase DataLoader num_workers for speed once the pipeline is stable; persistent_workers helps for repeated epochs.
- For larger datasets, pad_to_T allows fixed-length batches; otherwise variable-T collate remains available if custom collate is enabled.

## Reproducibility

- Random seed is set in the DataModule for planet-wise splits; set seed in all relevant frameworks for strict determinism if required.
- Checkpoints include hyperparameters via Lightning’s save_hyperparameters and can be restored with load_from_checkpoint.

## Outputs

- processed_data/planet_..._downsample_*.npy arrays for training/inference.
- processed_data/transit_depth.csv used as a conditioning feature.
- predictions/mu_predictions.csv with predicted spectra per planet.
- logs/ariel_conformer/ for TensorBoard scalars.

## Troubleshooting

- Shape mismatches: verify AIRS ROI slices [8:24, 39:321] and downsample factors; ensure AIRS factor × 12 ≈ FGS factor to align timing.
- Empty CDS: indicates too few frames; check signal parquet integrity and dt patterns.
- NaNs after calibration: sanitize with np.nan_to_num and confirm linear_corr, flat, dark, dead shapes per sensor.
- GPU issues: if FlashAttention warnings appear, it’s informational; the model runs with standard attention. Use appropriate CUDA wheels for torch.

## Citation and license

This code is intended for research use with Ariel Data Challenge format datasets. Please cite the Ariel competition/dataset where appropriate and respect their data licensing terms.

Contributions via issues and pull requests are welcome.
