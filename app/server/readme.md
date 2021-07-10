# How to run Web service

0. Run virtual environment
1. Install the required packages
2. Command `python manage.py migrate` for migration
3. modify sample config files to real config file
    - `AI_Django/config.py.sample` to `AI_Django/config.py`
        - set the secret key 
    - `convert/config.py.sample` to `convert/config.py`
        - set the program path
4. Command `run.bat`
