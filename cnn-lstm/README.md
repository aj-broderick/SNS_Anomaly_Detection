# Convolutional Neural Network with Long Short Term Memory<br>(CNN-LSTM)

This project implements a modular framework for an anomaly detector using a CNN+LSTM model. The framework is structured into separate modules for data parsing, data preparation, model building, evaluation (with plots and metrics exported to PDF), and workflow orchestration. The project allows you to run the training or testing workflow through a single main entry point using command-line arguments.

## Installation

   ```bash
   git clone SNS_Anomaly_Detection/cnn_lstm
   cd SNS_Anomaly_Detection/cnn_lstm
   pip install -r SNS_Anomaly_Detection/requirements.txt
   python ~/SNS_Anomaly_Detection/cnn_lstm/driver.py --train
   python ~/SNS_Anomaly_Detection/cnn_lstm/driver.py --test
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

## Logs and Reports 

- **Logs are stored in the `logs/` directory:**
  - Overall logs: `logs/app.log`
  - Training logs: `logs/train_flow.log`
  - Testing logs: `logs/test_flow.log`

- **Reports are exported as PDF:**
  - Training report: `train_metrics_report.pdf`
  - Testing report: `test_metrics_report.pdf`
