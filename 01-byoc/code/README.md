# VGGish for Sound Classification

This repo contains a baseline model of sound classfication - VGGish structure trained on our own competition dataset.

* [Environments](#environments)
* [Label List](#label-list) 
* [Train VGGish](#train-vggish)
  * [Prepare train/test meta data](#prepare-train/test-meta-data)
  * [Train the model](#train-the-model)
* [Saved results](#saved-results)
* [Visualize the results](#visualize-the-results)


## Environments

The codebase is developed with Python 3.7.3. Install requirements as follows:

```bash
pip install -r requirements.txt
```
## Label List

| Label Number | Class Name |
|----------|-------|
| 0 | Barking  |
| 1 | Howling  |
| 2 | Crying  |
| 3 | COSmoke  |
| 4 | GlassBreaking  |
| 5 | Other  |

## Train VGGish

The training process consists of two parts: 1. Prepare train/test meta csv. 2. Train the model

### Prepare train/test meta data

Users need to prepare train meta csv data and follow this type of format:

| Filename | Label | Remark |
|----------|-------|--------|
| train_00001 | 0  | Barking |
| train_00002 | 0  | Barking |
| train_00601 | 3  | COSmoke |
| train_00801 | 4  | GlassBreaking |

When users train the model, they need to specify ```--data_dir```, which is the root directory of sound data. ```--csv_path```, which is metadata of training. Then, the dataset will load wav data and label from this csv data.

Users need to prepare test meta csv data and follow this type of format:

| Filename | Barking | Howling | Crying | COSmoke | GlassBreaking | Other |
|----------|-------|--------|-------|--------|-------|--------|
| public_00006 |0|1|0|0|0|0|
| public_00009 |0|0|1|0|0|0|
| public_00010 |0|0|0|0|0|1|
| public_00030 |0|0|0|1|0|0|

The test meta csv contains ground truth of testing sound data. 

### Train the model

Users can train VGGish model by executing the following commands.

```bash
python train.py --csv_path=./meta_train.csv --data_dir=./train --epochs=50 --val_split 0.1 --preload
```
```--val_split``` - the ratio of validation size split from training data

```--preload``` - whether to convert wav data to melspectrogram first before start training

The interface will be printed on screen like this:

```
Epoch 3/50
----------
8/9 [======>.] - ETA: 0s - train loss in batch: 1.2447 - train acc in batch: 0.7868
9/9 [========] - 7s 730ms/step - train loss in batch: 1.2447 - train acc in batch: 0.7868 - train epoch loss: 1.2528 - train acc: 0.7741 - train precision: 0.7805 - train recall: 0.7742 - train f1: 0.7730
0/1 [........] - ETA: 0s - val loss in batch: 0.0000e+00 - val acc in batch: 0.0000e+00
finish this epoch in 0m 0s
1/1 [========] - 0s 155ms/step - val loss in batch: 0.0000e+00 - val acc in batch: 0.0000e+00 - val epoch loss: 1.8093 - val acc: 0.2083 - val precision: 0.0347 - val recall: 0.1667 - val f1: 0.0575
```

## Saved results

The checkpoints will be saved in ```results/snapshots/[model_name]```.
The log information will be saved in ```results/log/[model_name]```.


```bash
root
├── results
│    ├── snapshots
│    |    └── model_name
│    |       └── epoch_001_valloss ... .pkl
│    |
│    └── log
│         └── model_name
│            ├── cfm.png
│            ├── events.out.tfevents.1599822797.tomorun.14975.0
│            └── classification_report.txt
│
├── losses.py
├── ops.py
├── models.py
├── config.py
├── train.py
├── test.py
├── dataset.py
├── README.md
├── utils.py
└── metrics.py
```

## Visualize the results

Running

```bash
tensorboard --logdir=results
```

from the command line and then navigating to [http://localhost:6006](http://localhost:6006)

## Test the model

Users can test the model with the following example command:

```bash
python test.py --test_csv ./meta_public_test.csv --data_dir ./private_test --model_name VGGish --model_path [path_of_models] --saved_root results/test --saved_name test_result
```

The testing results will be saved in ```--saved_root``` and the prefix of files will be ```--saved_name```. You'll get the classfication report and the confusion matrix in txt format.