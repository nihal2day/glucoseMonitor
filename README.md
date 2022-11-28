# glucoseMonitor
CS7643 final project.

## Create Anaconda Environment with Cuda support

``` shell
conda env create -f environment.yml
```

## Activate Environment

``` shell
conda activate gm
```

## Run Training

``` shell
python train.py
```

## Start Tensorboard

``` shell
tensorboard --logdir=runs
```