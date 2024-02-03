## EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning
[ArXiv](https://arxiv.org/abs/1901.00212) | [BibTex](#citation)
### Introduction:
We develop a new approach for image inpainting that does a better job of reproducing filled regions exhibiting fine details inspired by our understanding of how artists work: *lines first, color next*. We propose a two-stage adversarial model EdgeConnect that comprises of an edge generator followed by an image completion network. The edge generator hallucinates edges of the missing region (both regular and irregular) of the image, and the image completion network fills in the missing regions using hallucinated edges as a priori. Detailed description of the system can be found in our [paper](https://arxiv.org/abs/1901.00212).
<p align='center'>  
  <img src='https://user-images.githubusercontent.com/1743048/50673917-aac15080-0faf-11e9-9100-ef10864087c8.png' width='870'/>
</p>
(a) Input images with missing regions. The missing regions are depicted in white. (b) Computed edge masks. Edges drawn in black are computed (for the available regions) using Canny edge detector; whereas edges shown in blue are hallucinated by the edge generator network. (c) Image inpainting results of the proposed approach.

## Prerequisites
- Python 3
- PyTorch 1.0
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
### 1) Images
We use [Places2](http://places2.csail.mit.edu), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [Paris Street-View](https://github.com/pathak22/context-encoder) datasets. To train a model on the full dataset, download datasets from official websites. 

After downloading, run [`scripts/flist.py`](scripts/flist.py) to generate train, test and validation set file lists. For example, to generate the training set file list on Places2 dataset run:
```bash
mkdir datasets
python ./scripts/flist.py --path path_to_places2_train_set --output ./datasets/places_train.flist
```

### 2) Irregular Masks
Our model is trained on the irregular mask dataset provided by [Liu et al.](https://arxiv.org/abs/1804.07723). You can download publically available Irregular Mask Dataset from [their website](http://masc.cs.gmu.edu/wiki/partialconv).

Alternatively, you can download [Quick Draw Irregular Mask Dataset](https://github.com/karfly/qd-imd) by Karim Iskakov which is combination of 50 million strokes drawn by human hand.

Please use [`scripts/flist.py`](scripts/flist.py) to generate train, test and validation set masks file lists as explained above.

## Getting Started
Download the pre-trained models using the following links and copy them under `./checkpoints` directory.

[Places2](https://drive.google.com/drive/folders/158ch9Psjop0mQEdeIp9DKjrYIGTDsZKN) | [CelebA](https://drive.google.com/drive/folders/13JgMA5sKMYgRwHBp4f7PBc5orNJ_Cv-p) | [Paris-StreetView](https://drive.google.com/drive/folders/1hMGVz6Ck3erpP3BRNzG90HNCJl85kveN)

Alternatively, you can run the following script to automatically download the pre-trained models:
```bash
bash ./scripts/download_model.sh
```

### 1) Training
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

### 2) Testing
To test the model, create a `config.yaml` file similar to the [example config file](config.yml.example) and copy it under your checkpoints directory. Read the [configuration](#model-configuration) guide for more information on model configuration.

You can test the model on all three stages: 1) edge model, 2) inpaint model and 3) joint model. In each case, you need to provide an input image (image with a mask) and a grayscale mask file. Please make sure that the mask file covers the entire mask region in the input image. To test the model:
```bash
python test.py \
  --model [stage] \
  --checkpoints [path to checkpoints] \
  --input [path to input directory or file] \
  --mask [path to masks directory or mask file] \
  --output [path to the output directory]
```

We provide some test examples under `./examples` directory. Please download the [pre-trained models](#getting-started) and run:
```bash
python test.py \
  --checkpoints ./checkpoints/places2 
  --input ./examples/places2/images 
  --mask ./examples/places2/masks
  --output ./checkpoints/results
```
This script will inpaint all images in `./examples/places2/images` using their corresponding masks in `./examples/places2/mask` directory and saves the results in `./checkpoints/results` directory. By default `test.py` script is run on stage 3 (`--model=3`).

### 3) Evaluating
To evaluate the model, you need to first run the model in [test mode](#testing) against your validation set and save the results on disk. We provide a utility [`./scripts/metrics.py`](scripts/metrics.py) to evaluate the model using PSNR, SSIM and Mean Absolute Error:

```bash
python ./scripts/metrics.py --data-path [path to validation set] --output-path [path to model output]
```

To measure the Fréchet Inception Distance (FID score) run [`./scripts/fid_score.py`](scripts/fid_score.py). We utilize the PyTorch implementation of FID [from here](https://github.com/mseitzer/pytorch-fid) which uses the pretrained weights from PyTorch's Inception model.

```bash
python ./scripts/fid_score.py --path [path to validation, path to model output] --gpu [GPU id to use]
```

### Alternative Edge Detection
By default, we use Canny edge detector to extract edge information from the input images. If you want to train the model with an external edge detection ([Holistically-Nested Edge Detection](https://github.com/s9xie/hed) for example), you need to generate edge maps for the entire training/test sets as a pre-processing and their corresponding file lists using [`scripts/flist.py`](scripts/flist.py) as explained above. Please make sure the file names and directory structure match your training/test sets. You can switch to external edge detection by specifying `EDGE=2` in the config file.

### Model Configuration

The model configuration is stored in a [`config.yaml`](config.yml.example) file under your checkpoints directory. The following tables provide the documentation for all the options available in the configuration file:

#### General Model Configurations

Option          | Description
----------------| -----------
MODE            | 1: train, 2: test, 3: eval
MODEL           | 1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model
MASK            | 1: random block, 2: half, 3: external, 4: external + random block, 5: external + random block + half
EDGE            | 1: canny, 2: external
NMS             | 0: no non-max-suppression, 1: non-max-suppression on the external edges
SEED            | random number generator seed
GPU             | list of gpu ids, comma separated list e.g. [0,1]
DEBUG           | 0: no debug, 1: debugging mode
VERBOSE         | 0: no verbose, 1: output detailed statistics in the output console

#### Loading Train, Test and Validation Sets Configurations

Option          | Description
----------------| -----------
TRAIN_FLIST     | text file containing training set files list
VAL_FLIST       | text file containing validation set files list
TEST_FLIST      | text file containing test set files list
TRAIN_EDGE_FLIST| text file containing training set external edges files list (only with EDGE=2)
VAL_EDGE_FLIST  | text file containing validation set external edges files list (only with EDGE=2)
TEST_EDGE_FLIST | text file containing test set external edges files list (only with EDGE=2)
TRAIN_MASK_FLIST| text file containing training set masks files list (only with MASK=3, 4, 5)
VAL_MASK_FLIST  | text file containing validation set masks files list (only with MASK=3, 4, 5)
TEST_MASK_FLIST | text file containing test set masks files list (only with MASK=3, 4, 5)

#### Training Mode Configurations

Option                 |Default| Description
-----------------------|-------|------------
LR                     | 0.0001| learning rate
D2G_LR                 | 0.1   | discriminator/generator learning rate ratio
BETA1                  | 0.0   | adam optimizer beta1
BETA2                  | 0.9   | adam optimizer beta2
BATCH_SIZE             | 8     | input batch size 
INPUT_SIZE             | 256   | input image size for training. (0 for original size)
SIGMA                  | 2     | standard deviation of the Gaussian filter used in Canny edge detector </br>(0: random, -1: no edge)
MAX_ITERS              | 2e6   | maximum number of iterations to train the model
EDGE_THRESHOLD         | 0.5   | edge detection threshold (0-1)
L1_LOSS_WEIGHT         | 1     | l1 loss weight
FM_LOSS_WEIGHT         | 10    | feature-matching loss weight
STYLE_LOSS_WEIGHT      | 1     | style loss weight
CONTENT_LOSS_WEIGHT    | 1     | perceptual loss weight
INPAINT_ADV_LOSS_WEIGHT| 0.01  | adversarial loss weight
GAN_LOSS               | nsgan | **nsgan**: non-saturating gan, **lsgan**: least squares GAN, **hinge**: hinge loss GAN
GAN_POOL_SIZE          | 0     | fake images pool size
SAVE_INTERVAL          | 1000  | how many iterations to wait before saving model (0: never)
EVAL_INTERVAL          | 0     | how many iterations to wait before evaluating the model (0: never)
LOG_INTERVAL           | 10    | how many iterations to wait before logging training loss (0: never)
SAMPLE_INTERVAL        | 1000  | how many iterations to wait before saving sample (0: never)
SAMPLE_SIZE            | 12    | number of images to sample on each samling interval

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

Except where otherwise noted, this content is published under a [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license, which means that you can copy, remix, transform and build upon the content as long as you do not use the material for commercial purposes and give appropriate credit and provide a link to the license.


## Citation
If you use this code for your research, please cite our papers <a href="https://arxiv.org/abs/1901.00212">EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning</a> or <a href="http://openaccess.thecvf.com/content_ICCVW_2019/html/AIM/Nazeri_EdgeConnect_Structure_Guided_Image_Inpainting_using_Edge_Prediction_ICCVW_2019_paper.html">EdgeConnect: Structure Guided Image Inpainting using Edge Prediction</a>:

```
@inproceedings{nazeri2019edgeconnect,
  title={EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning},
  author={Nazeri, Kamyar and Ng, Eric and Joseph, Tony and Qureshi, Faisal and Ebrahimi, Mehran},
  journal={arXiv preprint},
  year={2019},
}

@InProceedings{Nazeri_2019_ICCV,
  title = {EdgeConnect: Structure Guided Image Inpainting using Edge Prediction},
  author = {Nazeri, Kamyar and Ng, Eric and Joseph, Tony and Qureshi, Faisal and Ebrahimi, Mehran},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV) Workshops},
  month = {Oct},
  year = {2019}
}
```
