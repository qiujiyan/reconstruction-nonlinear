# reconstruction-nonlinear

Reconstructing global flow fields accurately using a limited number of measurements has been a long-standing challenge in the field. The quantity and positioning of measurements plays a critical role in determining the accuracy of the reconstruction. To address this issue, this paper proposes a global flow field reconstruction method based on an unsupervised mode extraction network. This approach maintains interpretability while effectively handling arbitrary quantities and positioning of sensors, ensuring high accuracy in the reconstruction of flow fields and other modal data. Specifically, this paper first trains an unsupervised deep autoencoder on global flow fields to capture the nonlinear modes of the flow. Subsequently, this paper leverages the backpropagation capability of the deep network to transform the flow field reconstruction problem into an interpretable optimization problem. Finally, by solving this optimization problem, this paper obtains the complete flow field. Experimental results demonstrate that the method proposed in this paper consistently achieves high accuracy across various flow fields, surpassing the performance of existing approaches.


# Usage

## Installation

```
pip install -r requirements.txt
```

## execution

You can run notebooks in example

## dataset 

All dataset can be found at [here](https://drive.google.com/drive/folders/1BMb4NSLIo315ZglnBmVCtDEdCkcZtVgW?usp=sharing)
