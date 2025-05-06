<div align="center">
<img src="https://www.odu.edu/sites/default/files/logos/univ/png-72dpi/odu-sig-noidea-fullcolor.png" style="width:225px;">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/JLab_logo_white2.jpg/250px-JLab_logo_white2.jpg" style="width:225px;"> 
<img src="https://cdn.vanderbilt.edu/vu-news/files/20190417211432/Oak-Ridge-National-Laboratory-logo.jpg" style="width:180px;">
</div>

<div align="center"> <font color=#003057>
        
# Data Science Capstone Project Spring 2025 

</font>

<div> 
<font size=4 color=#828A8F><b>May 2025</b></font><br>
<font size=4><i>AJ Broderick, Arun Thakur, Ashish Verma</i></font>
</div>

</div>


## Scope
&emsp;The Spallation Neutron Source (SNS) at Oak Ridge National Laboratory is a world-leading facility for neutron scattering research, providing unprecedented insights into the structure and dynamics of materials. As a highly complex and high-throughput scientific instrument, the SNS involves numerous subsystems—ranging from high-powered accelerators to cryogenic systems and neutron detectors—all of which must operate within strict performance and safety margins. The ability to detect anomalies in these systems promptly and accurately is critical for ensuring experimental integrity, maintaining uptime, and protecting equipment.

&emsp; This report explores the application of machine learning (ML) techniques for anomaly detection within the SNS environment. By leveraging historical sensor data, waveform signals, and system logs, machine learning models—especially those using deep learning architectures—offer the potential to identify subtle, non-obvious deviations from normal operational patterns. These methods can supplement or even surpass traditional rule-based monitoring by learning complex patterns and adapting to system evolution over time. The scope of this investigation includes the selection of appropriate algorithms, data preprocessing strategies, model evaluation metrics, and integration considerations within the SNS control infrastructure.

## High level architecture 
### VAE-BiLSTM
<img src="media/vae_bilstm_architecture.png" alt="VAE-BiLSTM architecture">

### VAE-BiLSTM
<img src="media/cnn_lstm_architecture.png" alt="CNN-LSTM architecture">

## Data analysis


## Current Model Architecture 

<table> <tr> 
<th>VAE-BiLSTM</th><th>CNN-LSTM</th>
</tr>
        
<tr><td><pre>
        
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

</pre></td>

<td><pre>
        
```         
cnn-lstm
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

</pre></td></tr>
</table>

## Results 

### VAE-BiLSTM
<img src="media/vae_reconstruction_error_chart.png" alt="VAE-BiLSTM Results">

### CNN-LSTM
<img src="media/cnn_training_results.png" alt="CNN-LSTM Training Results">
<img src="media/cnn_testing_results.png" alt="CNN-LSTM Testing Results">

## Pros and Cons

### VAE-BiLSTM
<table> <tr> 
<th>Pros</th><th>Cons</th>
</tr>
        
<tr><td><pre>
+ Temporal Dependencies
+ Probabilistic Latent Space
+ Sequence Variability
+ Regularized Learning
</pre></td>

<td><pre>
- Training Instability 
- Computational Complexity
- Interpretation 
- Data Requirements
</pre></td></tr>
</table>

### CNN-LSTM
<table> <tr> 
<th>Pros</th><th>Cons</th>
</tr>
        
<tr><td><pre>
+ Feature Extraction
+ Temporal Modeling
+ Dimensionality Reduction
+ Robust to Noise

</pre></td>

<td><pre>
- Spatial Bias
- Fixed Kernal Size
-Sequence Length 
- Architecture Tuning

</pre></td></tr>
</table>

## Future Enchantments
## References
*Staffini, A., Svensson, T., Chung, U.-i., & Svensson, A. K. (2023). A Disentangled VAE-BiLSTM Model for Heart Rate Anomaly Detection. Bioengineering, 10(6), 683. https://doi.org/10.3390/bioengineering10060683*

*Zhao, Yun & Zhang, Xiuguo & Shang, Zijing & Cao, Zhiying. (2021). A Novel Hybrid Method for KPI Anomaly Detection Based on VAE and SVDD. Symmetry. 13. 2104. 10.3390/sym13112104.*

*Mahmoud Abdallah, Nhien An Le Khac, Hamed Jahromi, and Anca Delia Jurcut. 2021. A Hybrid CNN-LSTM Based Approach for Anomaly Detection Systems in SDNs. In Proceedings of the 16th International Conference on Availability, Reliability and Security (ARES '21). Association for Computing Machinery, New York, NY, USA, Article 34, 1–7. https://doi.org/10.1145/3465481.3469190*
