from dataclasses import dataclass
from typing import List, Tuple

import cv2
import gradio as gr
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ImageResult:
    filename: str
    contour: float
    area: float
    plot: np.ndarray


def get_contours(masks: np.ndarray) -> Tuple[List[np.ndarray], float]:
    """Get the contours of the masks.

    Args:
        masks (np.ndarray): The masks of the images.

    Raises:
        gr.Error: If OpenCV encounters an error.

    Returns:
        courters: The contours of the masks.
        length: The sum of the lengths of the contours.
    """
    contours = []
    for mask_item in masks:
        try:
            temp_contours, _ = cv2.findContours(
                mask_item.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
        except cv2.error as e:
            raise gr.Error(e.msg)
        contours.append(temp_contours[0])

    lengths = [cv2.arcLength(contour, True) for contour in contours]
    return contours, sum(lengths, 0.0)


def get_area(contours: List[np.ndarray]) -> float:
    """Get the area of the contours.

    Args:
        contours (List[np.ndarray]): The contours of the masks.

    Returns:
        area: The sum of the areas of the contours.
    """
    areas = []
    for contour_item in contours:
        try:
            area = cv2.contourArea(contour_item)
        except cv2.error as e:
            logger.warning(f"Failed to execute cv2.contourArea on a mask: {e}")
            continue
        areas.append(area)
    return sum(areas, 0.0)


def generate_dataframe(results: List[ImageResult]) -> pd.DataFrame:
    """Generate a DataFrame from the results.

    Args:
        results (List[ImageResult]): The results to generate the DataFrame from.

    Returns:
        pd.DataFrame: The DataFrame generated from the results.
    """
    data = []
    for result in results:
        data.append(
            {
                "filename": result.filename,
                "contour": result.contour,
                "area": result.area,
            }
        )

    df = pd.DataFrame(data)
    df.reset_index(inplace=True)
    return df
