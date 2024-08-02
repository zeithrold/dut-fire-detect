import tempfile
from io import BytesIO
from zipfile import ZipFile
from pathlib import Path
from dataclasses import dataclass

from typing import List

import cv2
import gradio as gr
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

# 在这里输入训练好的模型路径
model_path = "./weights/fire.pt"

if not Path(model_path).exists():
    logger.error(f"模型文件{model_path}不存在，请检查路径是否正确。")
    raise FileNotFoundError(f"模型文件{model_path}不存在，请检查路径是否正确。")

image_file_suffix = ["jpg", "jpeg", "png"]

logger.info("正在加载模型...")
model = YOLO(model_path)


@dataclass
class Plot:
    filename: str
    img: Image.Image


def get_area(contours: List[np.ndarray]):
    areas = []
    for contour in contours:
        try:
            area = cv2.contourArea(contour)
        except cv2.error as e:
            logger.warning(f"在一张mask上OpenCV执行contourArea失败：{e}")
            continue
        areas.append(area)
    return sum(areas)
    # return sum([cv2.contourArea(contour.astype("uint8")) for contour in contours])


def get_contours(masks: np.ndarray):
    contours = []
    for mask in masks:
        try:
            temp_contours, _ = cv2.findContours(
                mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
        except cv2.error as e:
            raise gr.Error(e)
        contours.append(temp_contours[0])

    lengths = [cv2.arcLength(contour, True) for contour in contours]
    return contours, sum(lengths)


def parse_zipfile(file_path: str):
    logger.info(f"解析压缩文件：{file_path}")
    image_list = []
    with ZipFile(file_path, "r") as zip_ref:
        file_info_list = zip_ref.infolist()
        for file_info in file_info_list:
            if file_info.filename.split(".")[-1] in image_file_suffix:
                try:
                    image = Image.open(BytesIO(zip_ref.read(file_info.filename)))
                    image.filename = file_info.filename
                    image_list.append(image)
                except Exception as e:
                    logger.error(f"解析图片{file_info.filename}失败，错误信息：{e}")
                    raise gr.Error(f"解析图片{file_info.filename}失败")
    if image_list == []:
        raise gr.Error("没有找到图片，请注意上传的文件格式。")
    return image_list


def segment(image_list: List[Image.Image], progress: gr.Progress):
    logger.info("开始分割图片...")
    results = []
    plots: List[Plot] = []
    for image in progress.tqdm(image_list, desc="正在分割图片..."):
        filename: str = image.filename
        seg: Results = model(image, verbose=False)[0]
        if not seg.masks:
            results.append(
                {
                    "length": 0,
                    "area": 0,
                    "path": seg.path,
                }
            )
            continue
        masks = seg.masks.data
        path = seg.path
        if type(masks) is not np.ndarray:
            masks = masks.cpu().numpy()
        plot_img = Image.fromarray(seg.plot())
        plots.append(
            Plot(
                filename=f"{filename}_plotted.png",
                img=plot_img,
            )
        )
        del seg
        contours, length = get_contours(masks)
        area = get_area(contours)
        results.append(
            {
                "length": length,
                "area": area,
                "path": path,
            }
        )
    return results, plots


def gradio_interface(file: str, progress=gr.Progress()):
    logger.info(f"接收到Gradio任务：{file}")
    image_list = parse_zipfile(file)
    archive_file = Path(file).name
    results, plots = segment(image_list, progress)
    df = pd.DataFrame(results)
    df.reset_index(inplace=True)
    excel_file = tempfile.NamedTemporaryFile(
        prefix=f"result_{archive_file}_", suffix=".xlsx", delete=False
    )
    zip_file = tempfile.NamedTemporaryFile(
        prefix=f"plots_{archive_file}", suffix=".zip", delete=False
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        with ZipFile(zip_file, "w") as zip_ref:
            for plot in plots:
                plot_path = Path(temp_dir) / plot.filename
                if not plot_path.parent.exists():
                    plot_path.parent.mkdir(parents=True)
                plot.img.save(plot_path)
                zip_path = Path("plots") / plot.filename
                zip_ref.write(plot_path, zip_path)

    df.to_excel(excel_file, index=False)
    logger.success(f"完成Gradio任务：{file}")
    return (
        gr.BarPlot(
            value=df,
            x="index",
            y="area",
            title="面积",
            x_title="图片",
            y_title="面积",
        ),
        gr.BarPlot(
            value=df,
            x="index",
            y="length",
            title="周长",
            x_title="图片",
            y_title="周长",
        ),
        excel_file.name,
        zip_file.name
    )


with gr.Blocks() as demo:
    with gr.Column():
        upload_title = gr.Markdown("## 上传图片")
        upload_description = gr.Markdown(
            "上传一个.zip压缩的文件，里面包含需要分割的图片。支持的图片格式有：**jpg, jpeg, png**。"
        )
        file = gr.File(label="上传图片", file_types=[".zip"])
        button = gr.Button(value="开始分割", variant="primary")
    with gr.Column():
        result_title = gr.Markdown("## 分割结果")
        result_description = gr.Markdown("分割结果将会显示在这里。")
        with gr.Row():
            with gr.Column():
                area_title = gr.Markdown("### 面积")
                area_barplot = gr.BarPlot()
            with gr.Column():
                length_title = gr.Markdown("### 周长")
                length_barplot = gr.BarPlot()
        result_file = gr.File(label="下载结果")
        plot_file = gr.File(label="下载图片")
    button.click(
        fn=gradio_interface,
        inputs=[file],
        outputs=[area_barplot, length_barplot, result_file, plot_file],
    )
demo.launch(share=True)
