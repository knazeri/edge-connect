## EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning
<p align='center'>  
  <img src='https://user-images.githubusercontent.com/1743048/50255605-916db100-03c0-11e9-8baa-6af87c2d86a9.png' width='870'/>
</p>

## Prerequisites
- Python 3
- PyTorch
- NVIDIA GPU + CUDA cuDNN

## Installation
- Clone this repo:
```bash
git clone https://github.com/knazeri/edge-connect.git
cd edge-connect
```
- Install PyTorch and dependencies from http://pytorch.org
- Install python requirements:
```bash
pip install -r requirements.txt
```

## Datasets
We use [Places2](http://places2.csail.mit.edu), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [Paris Street-View](https://github.com/pathak22/context-encoder) datasets. To train a model on the full dataset, download datasets from official websites. Our model is trained on the irregular mask dataset provided by [Liu et al.](https://arxiv.org/abs/1804.07723). You can download publically available train/test mask dataset from [their website](http://masc.cs.gmu.edu/wiki/partialconv).

After downloading, run `scripts/flist.py` to generate train, test and validation set file lists. For example, to generate the training set file list on Places2 dataset run:
```bash
mkdir datasets
python ./scripts/flist.py --path path_to_places2_traininit_set --output ./datasets/places_train.flist
```

## Getting Started
Download the pre-trained models using the following links and copy them under `./checkpoints` directory.

[Places2](https://drive.google.com/drive/folders/1KyXz4W4SAvfsGh3NJ7XgdOv5t46o-8aa) | [CelebA](https://drive.google.com/drive/folders/1nkLOhzWL-w2euo0U6amhz7HVzqNC5rqb) | [Paris-StreetView](https://drive.google.com/drive/folders/1cGwDaZqDcqYU7kDuEbMXa9TP3uDJRBR1)

Alternatively, you can run the following script to automatically download the pre-trained models:
```bash
bash ./scripts/download_model.sh
```

### Training
To train the model, create a `config.yaml` file similar to the [example config file](https://github.com/knazeri/edge-connect/blob/master/config.yml.example) and copy it under your checkpoints directory. Read the [configuration](#model-configuration) guide for more information on model configuration.

EdgeConnect is trained in three stages: 1) training the edge model, 2) training the inpaint model and 3) training the joint model. To train the model:
```bash
python train.py --model [stage] --checkpoints [path to checkpoints]
```

For example to train the edge model on Places2 dataset under `./checkpoints/places2` directory:
```bash
python train.py --model 1 --checkpoints ./checkpoints/places2
```

Convergence of the model differs from dataset to dataset. For example Places2 dataset converges in one of two epochs, while smaller datasets like CelebA require almost 40 epochs to converge. You can set the number of training iterations by changing `MAX_ITERS` value in the configuration file.


### Test
To test the model, create a `config.yaml` file similar to the [example config file](https://github.com/knazeri/edge-connect/blob/master/config.yml.example) and copy it under your checkpoints directory. Read the [configuration](#model-configuration) guide for more information on model configuration.

You can test the model on all three stages: 1) edge model, 2) inpaint model and 3) joint model. To test the model:
```bash
python test.py \
  --model [stage] \
  --checkpoints [path to checkpoints] \
  --input [path to input directory or file] \
  --mask [path to masks directory or mask file] \
  --output [path to the output directory]
```

### Model Configuration

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

Except where otherwise noted, this content is published under a [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license, which means that you can copy, remix, transform and build upon the content as long as you do not use the material for commercial purposes and give appropriate credit and provide a link to the license.

