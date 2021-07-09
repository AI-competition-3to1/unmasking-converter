import os
from utils.files import search_file, copy_files
from utils.preprocess import set_filters, preprocess
from utils.logger import Logger

logger = Logger().get_logger()


def main():
    # Construct your own dataset using options
    basedir = "C://Users//Handal//Downloads"
    source_dir = os.path.join(
        basedir, f"2020.AICompetition//data//05_face_verification_Accessories"
    )
    target_dir = f"./data/face_verification_accessories"

    filters = set_filters(
        person=["*"],
        accessories=["001", "003"],
        lights=["1", "2"],
        emotions=["01", "02"],
        angles=["7"],
    )

    logger.info(f"Construct Train dataset")
    preprocess(source_dir, target_dir, "train", filters, logger)

    logger.info(f"Construct Validate dataset")
    preprocess(source_dir, target_dir, "validate", filters, logger)

    logger.info("Preprocessing is Done")


if __name__ == "__main__":
    main()
