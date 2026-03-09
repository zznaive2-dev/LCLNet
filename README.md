# Quickly Start

## Requirements

Tested environment:
- Python 3.9.25
- torch==1.13.0+cu117
- torchvision==0.14.0+cu117

### Option A: Build environment via script
```bash
bash enviorment.sh
```

### Option B: Install from requirements
```bash
pip install -r requirements.txt
```

Notes:
- Option A installs the CUDA 11.7 matching PyTorch build.
- If you already have a working PyTorch environment, Option B is usually enough.

## Preparations

Download datasets:

* [**Dresden**](https://www.kaggle.com/datasets/micscodes/dresden-image-database)
* [**Vision**](https://lesc.dinfo.unifi.it/VISION/)
* [**Daxing**](https://pan.baidu.com/s/1bXfgR1bX7qQvD5LcXyTjfA?pwd=2024)


Data preprocessing：
```bash
python ./utils/to_npy.py
```

## Run and Evaluation

Run :
```bash
python run_train.py
```

Then run evaluation scripts:
```bash
python run_eval.py
```


