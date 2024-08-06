from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List
from zipfile import ZipFile

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

from dut_fire_detect.data import ImageResult


@dataclass
class RawImage:
    filename: str
    img: np.ndarray


def read_zipfile(file_path: str):
    """Read the images from a zipfile.

    Args:
        file_path (str): The path to the zipfile.

    Returns:
        List[RawImage]: The images in the zipfile.
    """
    image_list: List[RawImage] = []
    with ZipFile(file_path, "r") as zip_ref:
        file_info_list = zip_ref.infolist()

        for item in file_info_list:
            filename = item.filename

            if filename.split(".")[-1].lower() not in ["jpg", "jpeg", "png"]:
                continue

            with zip_ref.open(item) as file:
                filename = Path(filename).name
                img = Image.open(file).convert("RGB")
                image_list.append(RawImage(filename=filename, img=np.array(img)))

    return image_list


def write_zipfile(results: List[ImageResult], df: pd.DataFrame):
    """Write the results to a zipfile.

    Args:
        results (List[ImageResult]): The results to write.

    Returns:
        zipfile_path: The path to the zipfile.
    """
    with TemporaryDirectory() as temp_dir:
        img_path = Path(temp_dir) / "images"
        img_path.mkdir()
        for result in results:
            img = Image.fromarray(result.plot)
            img_filename = f"{result.filename}_plotted.png"
            img.save(Path(img_path) / img_filename)

        df.to_csv(Path(temp_dir) / "results.csv", index=False)
        target_zipfile_path = Path("results.zip")
        if target_zipfile_path.exists():
            if target_zipfile_path.is_file():
                target_zipfile_path.unlink()
            else:
                raise gr.Error("The target zipfile is a directory.")
        with ZipFile(target_zipfile_path, "x") as zip_ref:
            for img in img_path.iterdir():
                zip_ref.write(img, f"images/{img.name}")
            zip_ref.write(Path(temp_dir) / "results.csv", "results.csv")

    return target_zipfile_path
