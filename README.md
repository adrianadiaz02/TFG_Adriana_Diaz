# UNSUPERVISED LEARNING FOR TEMPORAL ACTION SEGMENTATION

## Overview
This repository contains the official implementation of the Bachelor's degree thesis "Unsupervised Learning for Temporal Action Segmentation" by Adriana Díaz Soley.

The base code was adapted from [TOT-CVPR22](https://github.com/trquhuytin/TOT-CVPR22/tree/main), which provides the official implementation of their CVPR 2022 paper ([Kumar et al., 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Kumar_Unsupervised_Action_Segmentation_by_Joint_Representation_Learning_and_Online_Clustering_CVPR_2022_paper.pdf)).


If you use this code, please cite both this work and the original paper:
```
@inproceedings{
title={Unsupervised Learning for Temporal Action Segmentation},
author={Adriana Díaz Soley},
supervisors={Javier Ruiz Hidalgo, Mariella Dimiccoli},
year={2024}
}
```

```
@inproceedings{kumar2022unsupervised,
title={Unsupervised action segmentation by joint representation learning and online clustering},
author={Kumar, Sateesh and Haresh, Sanjay and Ahmed, Awais and Konin, Andrey and Zia, M Zeeshan and Tran, Quoc-Huy},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
pages={20174--20185},
year={2022}
}
```


## Installation

#### Environment Setup
Create Conda environment 
```
 conda create -n tot
```
Install packages from requirements file
```
 pip install -r requirements.txt
```

#### Directory structure
For each dataset create separate folder (specify path --dataset_root) where the inner folders structure is as following:
> features/  
> groundTruth/  
> mapping/  
> models/

During testing will be created several folders which by default stored at --dataset_root, change if necessary 
--output_dir 
> segmentation/  
> likelihood/  
> logs/  

### Structure of the project


## Dataset: Breakfast dataset
- Breakfast features [link](https://drive.google.com/file/d/1DbYnU2GBb68CxEt2I50QZm17KGYKNR1L)
- Breakfast ground truth [link](https://drive.google.com/file/d/1RO8lrvLy4bVaxZ7C62R0jVQtclXibLXU)


## Training

- actions: 'coffee', 'cereals', 'tea', 'milk', 'juice', 'sanwich', 'scrambledegg', 'friedegg', 'salat', 'pancake'
    use 'all' to train/test on all actions in series
```
python data_utils/BF_utils/bf_train.py
```


## Testing
To Evaluate the model, first set the model path in Test.py file for dataset. 
```
 opt.loaded_model_name = 'model path'
```

- actions: 'coffee', 'cereals', 'tea', 'milk', 'juice', 'sanwich', 'scrambledegg', 'friedegg', 'salat', 'pancake'
    use 'all' to train/test on all actions in series
```
python data_utils/BF_utils/bf_test.py
```

## Number of actions (K) for each Activity

| Activity class name  | # subactions (K) |
| -------------------- | ---------------- |
|        Coffe         |        7         |
|        Cereals       |        5         |
|        Tea           |        7         |
|        Milk          |        5         |
|        Juice         |        8         |
|        Sandwich      |        9         |
|        Scrambledegg  |       12         |
|        Friedegg      |        9         |
|        Salat         |        8         |
|        Pancake       |       14         |

     
