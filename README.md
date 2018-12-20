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
- We use [Places2](http://places2.csail.mit.edu), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [Paris Street-View](https://github.com/pathak22/context-encoder) datasets. To train a model on the full dataset, download datasets from official websites.
After downloading, run `scripts/flist.py` to generate train, test and validation set file lists. For example, to generate the training set file list on Places2 dataset run:
```bash
mkdir datasets
python ./scripts/flist.py --path path_to_places2_traininit_set --output ./datasets/places_train.flist
```

## Getting Started
Download the pre-trained models using the following links.

[Places2](https://drive.google.com/drive/folders/1KyXz4W4SAvfsGh3NJ7XgdOv5t46o-8aa) | [CelebA](https://drive.google.com/drive/folders/1nkLOhzWL-w2euo0U6amhz7HVzqNC5rqb) | [Paris-StreetView](https://drive.google.com/drive/folders/1cGwDaZqDcqYU7kDuEbMXa9TP3uDJRBR1)

Alternatively, you can run the following script to automatically download the pre-trained models:
```bash
bash ./scripts/download_model.sh
```

### Training

### Test

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

Except where otherwise noted, this content is published under a [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license, which means that you can copy, remix, transform and build upon the content as long as you do not use the material for commercial purposes and give appropriate credit and provide a link to the license.

