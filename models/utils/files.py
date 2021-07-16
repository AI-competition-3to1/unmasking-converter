import os
import glob
from pathlib import Path
import shutil


def copy_files(files: list, dest_dir: Path, clear=False):
    file = ""
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    try:
        if clear:
            shutil.rmtree(dest_dir)
            os.makedirs(dest_dir)

        for file in files:
            shutil.copy(file, dest_dir)
    except:
        FILE_COPY_FAIL_MSG = f"Copying File is failed : {file}"
        raise FILE_COPY_FAIL_MSG


def check_file(file: Path):
    """
    Search for file if not found
    """
    if Path(file).is_file() or file == "":
        return file
    else:
        files = glob.glob("./**/" + file, recursive=True)  # find file
        FILE_NOT_FOUND_MSG = f"File Not Found: {file}"
        MULTIPLE_FILE_MSG = f"Multiple files match '{file}', specify exact path:{files}"

        assert len(files), FILE_NOT_FOUND_MSG  # assert file was found
        assert len(files) == 1, MULTIPLE_FILE_MSG  # assert unique
        return files[0]  # return file


def search_file(directory: Path, filename=None, recursive=True, extension=None):
    """
    Find the target path
    """
    assert Path(directory).exists(), f"Path({directory}) is not existence"
    assert Path(directory).is_dir(), f"Path({directory}) should be direcory"

    target_file = directory + "/**"
    if filename:
        target_file = target_file + "/" + filename

    if extension:
        target_file = target_file + "/*." + extension

    files = glob.glob(target_file, recursive=recursive)
    if len(files) == 1:
        return files[0]

    assert len(files), f"No files is not founded : {target_file}"
    return files

if __name__ == "__main__":
    files = glob.glob("C:\\Users\\Handal\\Downloads\\thumbnails128x128\\thumbnails128x128\\*")
    
    src_dir = "C:\\Users\\Handal\\Downloads\\thumbnails128x128\\thumbnails128x128\\"
    dst_dir = "C:\\Users\\Handal\\github\\unmasking-converter\\models\\data\\joli\\trainB\\"
    for f in files:
        basename = f.split("\\")[-1]
        shutil.move(src_dir + basename, dst_dir + basename)