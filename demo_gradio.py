# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gc
import glob
import os
import shutil
from datetime import datetime

import cv2
import gradio as gr
import numpy as np
import torch

from visual_util import predictions_to_glb
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import pose_encoding_to_extri_intri


def load_model(checkpoint_path: str) -> VGGTOmega:
    if not torch.cuda.is_available():
        raise gr.Error("CUDA is required to run VGGT-Omega.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to("cuda")


def run_model(target_dir: str, model: VGGTOmega, image_resolution: int) -> dict:
    print(f"Processing images from {target_dir}")

    image_names = sorted(glob.glob(os.path.join(target_dir, "images", "*")))
    if len(image_names) == 0:
        raise gr.Error("No images found. Please upload images or a video first.")

    images = load_and_preprocess_images(image_names, image_resolution=image_resolution).to("cuda")
    print(f"Preprocessed images shape: {tuple(images.shape)}")

    with torch.inference_mode():
        predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    predictions_np = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
            predictions_np[key] = value

    predictions_np["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions_np["depth"],
        predictions_np["extrinsic"],
        predictions_np["intrinsic"],
    )

    torch.cuda.empty_cache()
    return predictions_np


def unproject_depth_map_to_point_map(depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def file_path(file_data) -> str:
    if isinstance(file_data, dict) and "name" in file_data:
        return file_data["name"]
    if hasattr(file_data, "name"):
        return file_data.name
    return str(file_data)


def handle_uploads(input_video, input_images, video_sample_fps=1.0):
    gc.collect()
    torch.cuda.empty_cache()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = os.path.join("demo_outputs", f"input_images_{timestamp}")
    target_dir_images = os.path.join(target_dir, "images")
    os.makedirs(target_dir_images, exist_ok=True)

    image_paths = []
    if input_images is not None:
        for item in input_images:
            src_path = file_path(item)
            dst_path = os.path.join(target_dir_images, os.path.basename(src_path))
            shutil.copy(src_path, dst_path)
            image_paths.append(dst_path)

    if input_video is not None:
        video_path = file_path(input_video)
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        video_sample_fps = max(float(video_sample_fps), 0.1)
        frame_interval = max(int(round((fps if fps and fps > 0 else 1) / video_sample_fps)), 1)

        frame_idx = 0
        saved_idx = 0
        while True:
            ok, frame = video.read()
            if not ok:
                break
            if frame_idx % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{saved_idx:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                saved_idx += 1
            frame_idx += 1
        video.release()

    image_paths = sorted(image_paths)
    return target_dir, image_paths


def update_gallery_on_upload(input_video, input_images, video_sample_fps):
    if not input_video and not input_images:
        return None, "None", None, "Upload images or a video."
    target_dir, image_paths = handle_uploads(input_video, input_images, video_sample_fps)
    return None, target_dir, image_paths, "Upload complete. Click Reconstruct."


def gradio_demo(
    target_dir,
    model,
    image_resolution,
    conf_thres=20.0,
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    max_points_k=1000,
):
    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        raise gr.Error("Please upload images or a video first.")

    conf_thres = max(3.0, float(conf_thres))

    gc.collect()
    torch.cuda.empty_cache()

    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images))

    predictions = run_model(target_dir, model, image_resolution)
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    glbfile = glb_path(
        target_dir,
        conf_thres,
        mask_black_bg,
        mask_white_bg,
        show_cam,
        mask_sky,
        max_points_k,
    )
    scene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        max_points=int(max_points_k * 1000),
    )
    scene.export(file_obj=glbfile)

    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    return (
        glbfile,
        f"Reconstruction complete: {len(all_files)} frames.",
    )


def glb_path(
    target_dir,
    conf_thres,
    mask_black_bg,
    mask_white_bg,
    show_cam,
    mask_sky,
    max_points_k,
):
    return os.path.join(
        target_dir,
        f"scene_conf{conf_thres}_black{mask_black_bg}_white{mask_white_bg}_"
        f"cam{show_cam}_sky{mask_sky}_max{int(max_points_k)}k.glb",
    )


def update_visualization(
    target_dir,
    conf_thres,
    mask_black_bg,
    mask_white_bg,
    show_cam,
    mask_sky,
    max_points_k,
):
    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Click Reconstruct first."

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, "No reconstruction available. Click Reconstruct first."

    conf_thres = max(3.0, float(conf_thres))

    glbfile = glb_path(
        target_dir,
        conf_thres,
        mask_black_bg,
        mask_white_bg,
        show_cam,
        mask_sky,
        max_points_k,
    )
    if not os.path.exists(glbfile):
        with np.load(predictions_path) as loaded:
            predictions = {key: np.array(loaded[key]) for key in loaded.files}
        scene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=target_dir,
            max_points=int(max_points_k * 1000),
        )
        scene.export(file_obj=glbfile)

    return glbfile, "Visualization updated."


def clear_model3d():
    return None


def update_log():
    return "Loading and Reconstructing..."


def update_visual_log():
    return "Updating visualization..."


def build_ui(model: VGGTOmega, image_resolution: int):
    def reconstruct(
        target_dir,
        conf_thres,
        mask_black_bg,
        mask_white_bg,
        show_cam,
        mask_sky,
        max_points_k,
    ):
        return gradio_demo(
            target_dir,
            model,
            image_resolution,
            conf_thres,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            max_points_k,
        )

    theme = gr.themes.Ocean()
    theme.set(
        checkbox_label_background_fill_selected="*button_primary_background_fill",
        checkbox_label_text_color_selected="*button_primary_text_color",
    )

    with gr.Blocks(
        theme=theme,
        css="""
        .custom-log * {
            font-style: italic;
            font-size: 22px !important;
            background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
            -webkit-background-clip: text;
            background-clip: text;
            font-weight: bold !important;
            color: transparent !important;
            text-align: center !important;
        }

        """,
    ) as demo:
        gr.HTML(
            """
        <h1>🌀 VGGT-Ω</h1>
        <p>
        <a href="https://github.com/facebookresearch/vggt-omega">🐙 GitHub Repository</a> |
        <a href="https://vggt-omega.github.io/">Project Page</a>
        </p>

        <div style="font-size: 16px; line-height: 1.5;">
        <p>Upload a video or a set of images to create a 3D reconstruction of a scene or object. VGGT-Ω takes these images and generates a 3D point cloud, along with estimated camera poses.</p>

        <h3>Getting Started:</h3>
        <ol>
            <li><strong>Upload Your Data:</strong> Use the "Upload Video" or "Upload Images" buttons on the left to provide your input. Videos will be automatically split into individual frames using the selected sampling rate.</li>
            <li><strong>Preview:</strong> Your uploaded images will appear in the gallery on the left.</li>
            <li><strong>Reconstruct:</strong> Click the "Reconstruct" button to run camera and depth inference and build the first GLB scene.</li>
            <li><strong>Visualize:</strong> The point cloud and camera poses will appear in the viewer on the right. You can rotate, pan, zoom, and download the GLB file.</li>
            <li>
            <strong>Adjust Visualization (Optional):</strong>
            After reconstruction, adjust the visualization options and click "Update Visual" to refresh the GLB without rerunning inference.
            </li>
        </ol>
        <p><strong style="color: #0ea5e9;">Please note:</strong> <span style="color: #0ea5e9; font-weight: bold;">The demo limits Max Points by default to keep the UI responsive; increase Max Points if you need a denser point cloud. Visualizing very dense point clouds may take longer due to third-party rendering, which is independent of VGGT-Ω's processing time.</span></p>
        </div>
            """
        )

        target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

        with gr.Row():
            with gr.Column(scale=2):
                input_video = gr.Video(label="Upload Video", interactive=True)
                video_sample_fps = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Video Sampling FPS",
                    interactive=True,
                )
                input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)
                image_gallery = gr.Gallery(
                    label="Preview",
                    columns=4,
                    height="300px",
                    show_download_button=True,
                    object_fit="contain",
                    preview=True,
                )

            with gr.Column(scale=4):
                with gr.Column():
                    gr.Markdown("**Reconstruction (Point Cloud and Camera Poses)**")
                    log_output = gr.Markdown(
                        "Please upload a video or images, then click Reconstruct.",
                        elem_classes=["custom-log"],
                    )
                    reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)

                with gr.Row():
                    submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                    update_visual_btn = gr.Button("Update Visual", scale=1)
                    clear_btn = gr.ClearButton(
                        [input_video, input_images, reconstruction_output, log_output, target_dir_output, image_gallery],
                        scale=1,
                    )

                with gr.Row():
                    conf_thres = gr.Slider(
                        minimum=2,
                        maximum=100,
                        value=50,
                        step=0.1,
                        label="Confidence Threshold (%)",
                    )
                    max_points_k = gr.Slider(
                        minimum=500,
                        maximum=10000,
                        value=1000,
                        step=500,
                        label="Max Points (K points)",
                    )
                    with gr.Column():
                        show_cam = gr.Checkbox(label="Show Camera", value=True)
                        mask_sky = gr.Checkbox(label="Filter Sky", value=False)
                        mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                        mask_white_bg = gr.Checkbox(label="Filter White Background", value=False)

        input_video.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images, video_sample_fps],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
        )
        input_images.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images, video_sample_fps],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
        )
        video_sample_fps.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images, video_sample_fps],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
        )

        submit_btn.click(fn=clear_model3d, inputs=[], outputs=[reconstruction_output]).then(
            fn=update_log,
            inputs=[],
            outputs=[log_output],
        ).then(
            fn=reconstruct,
            inputs=[
                target_dir_output,
                conf_thres,
                mask_black_bg,
                mask_white_bg,
                show_cam,
                mask_sky,
                max_points_k,
            ],
            outputs=[reconstruction_output, log_output],
        )

        update_visual_btn.click(fn=update_visual_log, inputs=[], outputs=[log_output]).then(
            fn=update_visualization,
            inputs=[
                target_dir_output,
                conf_thres,
                mask_black_bg,
                mask_white_bg,
                show_cam,
                mask_sky,
                max_points_k,
            ],
            outputs=[reconstruction_output, log_output],
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT-Omega Gradio demo")
    parser.add_argument("--checkpoint", required=True, help="Local VGGT-Omega checkpoint path.")
    parser.add_argument("--image-resolution", type=int, default=512, help="Input image resolution. Default: 512.")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading checkpoint from {args.checkpoint}")
    model = load_model(args.checkpoint)
    demo = build_ui(model, args.image_resolution)
    demo.queue(max_size=20).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
