# How to run Mask detect model

0. Run virtual environment
1. Install the required packages
2. Download dataset from [kaggle face mask dataset](https://www.kaggle.com/andrewmvd/face-mask-detection) into `../data/mask`
    - if you want to use another name for input data directory, Please check `config.yaml` data - directory
3. Downlaod pretrained model into `./pretrain/` directory
4. Set `config.yaml` your own config setting
    - `mode` : option for model
    - `data:visualize`  : option for visaulizing output
5. Command `python core.py` 
    - `--checkpoint True` option for saving checkpoint model during model train(default False)
