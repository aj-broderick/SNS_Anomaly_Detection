<div align="center">
<img src="https://www.odu.edu/sites/default/files/logos/univ/png-72dpi/odu-sig-noidea-fullcolor.png" style="width:225px;">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/JLab_logo_white2.jpg/250px-JLab_logo_white2.jpg" style="width:225px;"> 
<img src="https://cdn.vanderbilt.edu/vu-news/files/20190417211432/Oak-Ridge-National-Laboratory-logo.jpg" style="width:180px;">
</div>

<div align="center"> <font color=#003057>
        
# SNS Anomaly Detection

</font>

<div> 
<font size=4><i>AJ Broderick, Arun Thakur, Ashish Verma</i></font>
</div>

</div>

## Project Overview
This repository contains the codebase for an anomaly detection project at the Spallation Neutron Source (SNS), developed for the Old Dominion University Spring 2025 Capstone in collaboration with Jefferson Lab. The models aim to identify errant beam events that could damage accelerator equipment or cause downtime.

## Methods
Two deep learning architectures were implemented:
- **VAE-BiLSTM**: Combines a Variational Autoencoder with a Bidirectional LSTM to model time-series behavior.
- **CNN-LSTM**: Integrates Convolutional Neural Networks with LSTM layers for feature extraction and temporal modeling.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/aj-broderick/SNS_Anomaly_Detection.git
    cd SNS_Anomaly_Detection
    ```
2. Set up a Python virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Install dependencies:
    ```bash
    pip install -r sns_vae_bilstm/requirements.txt
    pip install -r sns_cnn_lstm/requirements.txt
    ```

## Usage
- For training and testing VAE-BiLSTM models, navigate to `sns_vae_bilstm/driver.py`.
- For CNN-LSTM models, use `sns_cnn_lstm/main.py` along with `train_flow.py` and `test_flow.py`.

## Acknowledgments
This project is a collaboration between Old Dominion University and Jefferson Lab. Special thanks to technical advisors Dr. Frank Liu and Kishansingh Rajput.
