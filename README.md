# Detect-CleaverHans-Effect-in-AD

## To run 'AD demo.ipynb':

### Data preparation:

1. Run the notebook on Colab and uploaded the datasets to the google Drive. You can also run it locally and change the path of the datsets and model config path.

![datapath](/demopicture/path3.png)

### Model preparation: 
(next few steps are already mentioned in notebook, so just run the notebook step by step if you run the notebook on Colab.)

2. Download anomalib and install it: https://github.com/openvinotoolkit/anomalib.git

3. I implemnented padim model and stfpm model with customed datsets. To do that, copy the model config and modify the copied config with customed parameters.
The customed configs are also in the github. Just download them and put it in the right folder. Pay attention: in config file, you need to change the normal data path and abnormal data path to your data location 

![model file structure1](/demopicture/path1.png) ![structure](/demopicture/path2.png)

### Inference and display
There are several types of heatmap which are indicated the anomalous parts

![prediction](/demopicture/pred1.png) ![heatmap1](/demopicture/heat1.png) ![heatmap2](/demopicture/heat2.png)

### A few things need to be considered:
1. In ped2 dataset, all test vidos are anomalous, but not all frames contain anomalous.(eg. video1:000.png and 090.png) We should consider to extract useful images during the preprocess.
2. Model might take a quite long time for the training.(I took 1.5h to train 2*5*180*0.8(?) images)
