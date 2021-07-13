# How to run Mask detect model

0. Run virtual environment
1. Download [kaggle face mask dataset](https://www.kaggle.com/andrewmvd/face-mask-detection)
2. Set your own config settings from `config.yaml`
    - `pip install Django`
    - `pip install pillow`
3. Command `python manage.py migrate` & `python manage.py makemigrations convert` for migration
5. modify sample config files to real config file
    - `configs/config.py.sample` to `configs/config.py`
        - set the secret key 
        - set the program path
6. Command `run.bat`
