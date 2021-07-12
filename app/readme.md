# How to run Web service

0. Run virtual environment
1. Install the required packages
    - `pip install Django`
    - `pip install pillow`
3. Command `python manage.py migrate` & `python manage.py makemigrations convert` for migration
5. modify sample config files to real config file
    - `configs/config.py.sample` to `configs/config.py`
        - set the secret key 
        - set the program path
6. Command `run.bat`
