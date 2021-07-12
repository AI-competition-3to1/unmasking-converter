import yaml
import argparse

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="config.yml", help="model.yml path"
    )
    opt = parser.parse_args()