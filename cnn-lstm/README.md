# Convolutional Neural Network with Long Short Term Memory<br>*(CNN-LSTM)*

This project implements a modular framework for an anomaly detector using a CNN+LSTM model. The framework is structured into separate modules for data parsing, data preparation, model building, evaluation (with plots and metrics exported to PDF), and workflow orchestration. The project allows you to run the training or testing workflow through a single main entry point using command-line arguments.

## Installation

   ```bash
   cd SNS_Anomaly_Detection/cnn_lstm
   pip install -r requirements.txt
   python driver.py --train
   python driver.py --test
   ```

## Project Structure

```
cnn_lstm
├── analysis
│    └── evaluation.py
├── data_preparation
│    ├── data_loader.py
│    ├── data_scaling.py
│    └── data_transformer.py
├── enterprise_tools
│    ├── beam_settings_parser_hdf5.py
│    ├── beam_settings_prep.py
│    └── data_utils.py
├── model
│    └── cnn_lstm_anomaly_model.py
├── parser
│    └── configs.py
├── testing
│    └── test.py
├── training
│    └── train.py
├── utils
│    └── logger.py
├── driver.py
├── submit.sh
└── requirements.txt
```

## Data Parser Modules

- To extract the data from it's raw form, the Beam Parameter Monitor (BPM) & Differential Current Monitor (DCM) configurations is implemented in `parser/configs.py` (classes  `BPMDataConfig` & `DCMDatConfig`).

## Enterprise Tools

- This folder works under the assumption that the user is executing the code outside of the Jefferson Lab environment and needs access to custom classes that were developed for unpacking SNS data from binary format. Files include:
  - `beam_settings_parser_hdf5.py`: Parser for beam configuration parameters
  - `beam_settings_prep.py`: Beam configuration pre-processing
  - `data_utils.py`: Parser for beam tracings 

## Logs and Reports 

- **Logs are stored in the `logs/` directory:**
  - Overall logs: `logs/app.log`
  - Training logs: `logs/train_flow.log`
  - Testing logs: `logs/test_flow.log`

- **Reports are exported as PDF:**
  - Training report: `train_metrics_report.pdf`
  - Testing report: `test_metrics_report.pdf`
