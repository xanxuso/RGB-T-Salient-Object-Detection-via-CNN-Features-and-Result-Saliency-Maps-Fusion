# RGB-T-Salient-Object-Detection-via-CNN-Features-and-Result-Saliency-Maps-Fusion
This project provides the codes and results for 'RGB-T-Salient-Object-Detection-via-CNN-Features-and-Result-Saliency-Maps-Fusion.'

## Overview
![image](https://github.com/xanxuso/RGB-T-Salient-Object-Detection-via-CNN-Features-and-Result-Saliency-Maps-Fusion/blob/main/network.png)

## Code
### For the training of the proposed network:
download pretrained vgg model from [link](https://pan.baidu.com/s/1DDHhmjau01Oo775mi1wdgw),code：0000; put it in `model` directory.

change `img_root` in `train.py` to load train data .

use `python train.py` to start training.
### For the testing of the proposed network:
change `model_path` and `root` in `test.py` to load trained checkpoint and test data.

change `out_path` in `test.py` to save test results.

use `python test.py` to start testing.
### For Result Fusion:
change paths in `resultFusion.py`.

use `python resultFusion.py` to fuse saliency maps from RGB and T modalities, there are some examples in `asserts` for testing.

## Experimental Results
Result of VT5000 dataset: [link](https://pan.baidu.com/s/1jn6Y9vi7qhnTIpHiW8anWA),&emsp;code：1234.

Result of VT1000 dataset: [link](https://pan.baidu.com/s/1jex2q55VZeSalOYWCtC4BQ),&emsp;code：4321.

The evaluation toolbox is provided by [https://github.com/ArcherFMY/sal_eval_toolbox](https://github.com/ArcherFMY/sal_eval_toolbox)
