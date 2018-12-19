## EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning

## Prerequisites
- Python 3
- PyTorch
- NVIDIA GPU + CUDA cuDNN

## Getting Started
### Installation
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

### Datasets
- We use [Places2](http://places2.csail.mit.edu), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [Paris Street-View](https://github.com/pathak22/context-encoder) datasets. To train a model on the full dataset, download datasets from official websites.
After downloading, run `scripts/flist.py` to generate train, test and validation set file lists. For example, to generate the training set file list on Places2 dataset run:
```bash
mkdir datasets
python ./scripts/flist.py --path path_to_places2_traininit_set --output ./datasets/places_train.flist
```

### Training

### Test

## License

## Citation
