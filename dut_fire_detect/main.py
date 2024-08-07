from pathlib import Path
from typing import List

import cv2
import gradio as gr
from loguru import logger
from tqdm import tqdm

from dut_fire_detect import data, file
from dut_fire_detect.model import Model

model_path = Path("weights") / "model.onnx"
if not model_path.exists():
    logger.error(
        "The model file does not exist, "
        "you should make sure 'weights/model.onnx' exists."
    )
    exit(1)

model = Model(str(model_path))


def gradio_predict(
    input_file: str,
    conf_threshold: float,
    iou_threshold: float,
    _=gr.Progress(track_tqdm=True),
):
    img_list = file.read_zipfile(input_file)
    result_list: List[data.ImageResult] = []
    for img_item in tqdm(img_list):
        boxes, segments, masks = model(img_item.img, conf_threshold, iou_threshold)
        if segments == []:
            length = 0.0
            area = 0.0
            plotted_img = img_item.img
        else:
            seg = segments[0]
            length = cv2.arcLength(seg, True)
            area = cv2.contourArea(seg)
            plotted_img = model.draw_and_visualize(img_item.img, boxes, segments)
        result_list.append(
            data.ImageResult(img_item.filename, length, area, plotted_img)
        )
    df = data.generate_dataframe(result_list)
    result_file = file.write_zipfile(result_list, df)
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
            y="contour",
            title="周长",
            x_title="图片",
            y_title="周长",
        ),
        str(result_file),
    )


def main():
    with gr.Blocks() as demo:
        with gr.Column():
            gr.Markdown("## 上传图片")
            gr.Markdown(
                (
                    "上传一个.zip压缩的文件，里面包含需要分割的图片。"
                    "支持的图片格式有：**jpg, jpeg, png**。"
                )
            )
            file = gr.File(label="上传图片", file_types=[".zip"])
            with gr.Row():
                conf_threshold = gr.Slider(
                    label="置信度阈值",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.4,
                )
                iou_threshold = gr.Slider(
                    label="IOU非极大值抑制阈值",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.45,
                )
            button = gr.Button(value="开始分割", variant="primary")
        with gr.Column():
            gr.Markdown("## 分割结果")
            gr.Markdown("分割结果将会显示在这里。")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 面积")
                    area_barplot = gr.BarPlot()
                with gr.Column():
                    gr.Markdown("### 周长")
                    length_barplot = gr.BarPlot()
            result_file = gr.File(label="下载结果")
        button.click(
            fn=gradio_predict,
            inputs=[file, conf_threshold, iou_threshold],
            outputs=[area_barplot, length_barplot, result_file],
        )
    demo.launch(share=False)
