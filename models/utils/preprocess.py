import os
from utils.files import search_file, copy_files


def set_filters(
    person=["*"],
    accessories=["001", "003"],
    lights=["1", "2", "3"],
    emotions=["01", "02"],
    angles=["7"],
):
    """
    params:
        person
            * 전체
        accessories
            001 없음 002 안경 003 뿔테 004 썬글라스 005 모자 006 뿔테 모자
        lights
            1 매우 밝음 2 밝음 3 보통 4 어두움 5 매우 어두움
        emotions
            01 무표정 02 웃음 03 찡그림
        angles
            6 -15' 7 0' 8 15'
    """

    # Person options

    options = [
        f"{p}_S{s}_L{l}_E{e}_C{c}_*"
        for p in person
        for s in accessories
        for l in lights
        for e in emotions
        for c in angles
    ]

    return options


def preprocess(
    soruce_dir, target_dir, option="train", filters=set_filters(), ext_logger=None
):
    assert (
        soruce_dir is not None or target_dir is not None
    ), "Directory should be written"

    datadir = os.path.join(soruce_dir, option)
    destdir = os.path.join(target_dir, option)

    count = 0
    for regexp in filters:
        files = search_file(directory=datadir, filename=regexp)
        copy_files(files, destdir, True)

        if ext_logger:
            count += len(files)
            ext_logger.info(
                f"{count:6d} image files (+{len(files):4d} from '{regexp}')"
            )

    if ext_logger:
        ext_logger.info(f"Total {count} data files is preprocessed")
