import os
import yaml
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="config.yml", help="model.yml path")
    parser.add_argument("--checkpoint", type=bool, default=False, help="save model at every epochs")

    return parser.parse_args()


def get_config():
    args = get_args()

    FILE_IS_NOT_EXIST_MESSAGE = f"{args.data} is not exist"
    assert os.path.exists(args.data), FILE_IS_NOT_EXIST_MESSAGE

    with open(args.data) as f:
        config = yaml.safe_load(f)

    config["checkpoint"] = args.checkpoint

    return config
