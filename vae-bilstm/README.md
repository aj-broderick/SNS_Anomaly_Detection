# Variational Autoencoder with Bidirectional Long Short-Term Memory<br>*(VAE-BiLSTM)*

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
├── enterprise_tools
│    ├── beam_settings_parser_hdf5.py
│    ├── beam_settings_prep.py
│    └── data_utils.py
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

## Data Preparation

- Due to limited time and resources, the model was trained on a small subset of data based on the beam parameter settings. To remove this filter to train and test on the full dataset, the following lines of code can be removed from the `factories\sns_raw_prep_sep_dnn_factory.py` file:
```bash
merged_df = grouper.filegroup(merged_df)
merged_df = merged_df[merged_df['group'] == 0]
```

## Enterprise Tools

- This folder works under the assumption that the user is executing the code outside of the Jefferson Lab environment and needs access to custom classes that were developed for unpacking SNS data from binary format. Files include:
  - `beam_settings_parser_hdf5.py`: Parser for beam configuration parameters
  - `beam_settings_prep.py`: Beam configuration pre-processing
  - `data_utils.py`: Parser for beam tracings 
  
## Logs and Reports 

- **Logs are stored in individual folders (formatted as YYYYMMDD-HHMMSS) within the `logs/` directory:**
  - Training logs: `logs/YYYYMMDD-HHMMSS/train`
  - Validation logs: `logs/YYYYMMDD-HHMMSS/validation`

- **Visuals are saved in the folder once rum:**
  - Time series plot of reconstruction error: `time_plot.png`
  - Histogram of reconstruction errors: `dist_plot.png`
