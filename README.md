# Landmark-Dectection

## Requirements

- python3
- numpy
- pytorch==1.4
- matplotlib
- sklearn
- scikit-image >= 0.16.1
- torchvision >= 0.6.1
- seaborn


## Run Demo

```
python3 demo.py
```

## Train unsupervised model(autoencoder)

First, put RHPE and RSNA dataset in your preferred folder and change local_config.ini

Train with dataset RHPE and RSNA for 1000 epochs

```
python3 train_aae_unsupervised.py --sessionname unsupervised --dataset-train rhpe rsna --epochs 1000 --with-gan --daug 4 --with-ssim-loss
```

## Train supervised model

```
python3 train_aae_landmarks.py --sessioname landmarks --dataset-train rsna --epochs 100 -r unsupervised/01000
```

You will see output images in the 'output' folder set in local_config.ini

## Train supervised model with few shots(100 images)

```
python3 train_aae_landmarks.py --sessioname landmarks --dataset rsna --epochs 100 -r unsupervised/01000 --train-count 100
```

