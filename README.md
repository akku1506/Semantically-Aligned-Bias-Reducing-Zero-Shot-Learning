# Semantically-Aligned-Bias-Reducing-Zero-Shot-Learning
Official Github repository for Semantically Aligned Bias Reducing ZSL in Tensorflow. If you find this code useful in your research, please consider citing:

```
@inproceedings{paul2019semantically,
  title={Semantically Aligned Bias Reducing Zero Shot Learning},
  author={Paul, Akanksha and Krishnan, Narayanan C and Munjal, Prateek},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={7056--7065},
  year={2019}
}
```

## Pre-Requisites
1. Python=3.5+, Tensorflow-gpu= 1.9

2. Install scipy, h5py and sklearn libraries of python

3. Download the links of the features and the models trained for SABR from:
```
  https://drive.google.com/open?id=1Xqvw8j81OhaME-dRI6WLjwVvQ1jG1xtr
```
  Copy the downloaded SABR folder from the above link inside the cloned github directory

4. Scripts in the main folder follow the following format:
```
$dataset_name_train_$setting.sh
```
where, $dataset_name can be AWA2, CUB or SUN.

and $setting can be i for inductive setting and t for transductive setting.

## Training from scratch and evaluating

1. Run the train_i.sh and train_t.sh for the dataset you wish to run the scripts for.

For each possible setting, you would obtain results in two lines like the sample below: 
```
# unseen class accuracy=  65.20058135379655
# unseen=27.9557, seen=90.5525, h=42.7221
```
The first line indicates the conventional ZSL performance while the second lines denotes the performance in the generalized ZSL setting.

## Loading the pre-trained models and evaluating

1. Comment out the lines 210-310 in cls_condwgan.py and lines 283-384 in sep_clswgan.py
2. Run the train_i.sh and train_t.sh for the dataset you wish to run the scripts for.


