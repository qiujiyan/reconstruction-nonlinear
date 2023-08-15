# reconstruction-nonlinear

The accurate reconstruction of global flow fields from sparse measurements has been a longstanding challenge in which the quantity and positioning of measurements play a critical role. To address this issue, we propose a global flow field reconstruction method based on a mode decomposition autoencoder, which maintains interpretability while effectively handling arbitrary quantities and positioning of sensors, ensuring high accuracy in the reconstruction of flow fields and other modal data. An autoencoder is trained on global flow fields to capture the nonlinear modes of the flow. The backpropagation capability of the deep network is leveraged to transform the flow field reconstruction problem into an interpretable optimization problem, which is solved to obtain the complete flow field. In experiments carried out on a stable ocean surface temperature dataset and an unstable multi-cylinder airflow dataset, the proposed method consistently achieved high accuracy across various flow fields, surpassing the performance of current approaches.

# Usage

## Installation

```
pip install -r requirements.txt
```

## execution

First unzip the zip file in datasets. You can run notebooks in example. Remember to modify the path of the data set.
