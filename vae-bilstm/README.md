# Variational Autoencoder with Bidirectional Long Short-Term Memory<br>VAE-BiLSTM

This project implements a modular framework for an anomaly detector using a VAE-BiLSTM model. The framework is structured into separate modules for data parsing, data preparation, model building, evaluation (with plots and metrics exported to PDF), and workflow orchestration. The project allows you to run the training or testing workflow through a single main entry point using command-line arguments including epochs, batch size, and threshold percentile for detecting anomaly.

## Installation

   ```bash
   cd SNS_Anomaly_Detection/vae-bilstm
   pip install -r requirements.txt
   # Train:
   python driver.py train --epochs 5 --batch_size 8 --learning_rate 1e-4 --latent_dim 32 --model_path vae_bilstm_model.weights.h5 --tensorboard_logdir logs/fit
   # Predict:
   python driver.py predict --model_path vae_bilstm_model.weights.h5 --threshold_percentile 90
   ```

## Project Structure

```
vae-bilstm
├── data_preparation
│    ├── data_loader.py
│    ├── data_scaling.py
│    └── data_transformer.py
├── factories
│    └── sns_raw_prep_sep_dnn_factory.py
├── model
│    └── vae_bilstm.py
├── parser
│    └── configs.py
├── utils
│    └── logger.py
├── visualization
│    └── plots.py
├── driver.py
├── submit.sh
└── requirements.txt
```

## Data Parser Modules

- To extract the data from it's raw form, the Beam Parameter Monitor (BPM) & Differential Current Monitor (DCM) configurations is implemented in `parser/configs.py` (classes  `BPMDataConfig` & `DCMDatConfig`).

## Logs and Reports 

- **Logs are stored in individual folders (formatted as YYYYMMDD-HHMMSS) within the `logs/` directory:**
  - Training logs: `logs/YYYYMMDD-HHMMSS/train`
  - Validation logs: `logs/YYYYMMDD-HHMMSS/validation`

- **Visuals are saved in the folder once rum:**
  - Time series plot of reconstruction error: `time_plot.png`
  - Histogram of reconstruction errors: `dist_plot.png`
