# YOLOv5 + Monocular 3D BBox video inference

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ------------------------------------------------------------
# NumPy compatibility for older codebases using deprecated types
# ------------------------------------------------------------
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

import torch
import torch.nn as nn

try:
    from torchvision.models import (
        resnet18,
        mobilenet_v3_small,
        MobileNet_V3_Small_Weights,
    )
    MOBILENET_WEIGHTS = MobileNet_V3_Small_Weights.DEFAULT
except ImportError:
    from torchvision.models import resnet18, mobilenet_v3_small
    MOBILENET_WEIGHTS = None

FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[1]
CODE_ROOT = FILE.parents[0]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

ROOT = PROJECT_ROOT

# ------------------------------------------------------------
# PyTorch 2.6+ compatibility patch for old YOLOv5 checkpoints
# ------------------------------------------------------------
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

from models.common import DetectMultiBackend
from utils.general import (
    LOGGER,
    check_img_size,
    non_max_suppression,
    print_args,
    scale_coords,
)
from utils.torch_utils import select_device
from utils.augmentations import letterbox

from script.Dataset import generate_bins, DetectedObject
from script import ClassAverages
from script.Model import ResNet, ResNet18, VGG11
from library.Math import calc_location
from library.Plotting import plot_2d_box, plot_3d_box


class MobileNetV3Regressor(nn.Module):
    def __init__(self, bins=2):
        super().__init__()
        self.bins = bins

        backbone = mobilenet_v3_small(weights=MOBILENET_WEIGHTS)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = 576

        self.fc_shared = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        self.fc_orient = nn.Linear(512, bins * 2)
        self.fc_conf = nn.Linear(512, bins)
        self.fc_dim = nn.Linear(512, 3)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_shared(x)

        orient = self.fc_orient(x).view(-1, self.bins, 2)
        conf = self.fc_conf(x)
        dim = self.fc_dim(x)

        return orient, conf, dim


model_factory = {
    "resnet": resnet18(weights=None),
    "resnet18": resnet18(weights=None),
    "mobilenetv3": None,
}

regressor_factory = {
    "resnet": ResNet,
    "resnet18": ResNet18,
    "vgg11": VGG11,
    "mobilenetv3": MobileNetV3Regressor,
}


class Bbox:
    def __init__(self, box_2d, class_, conf=0.0):
        self.box_2d = box_2d
        self.detected_class = class_
        self.conf = conf


def load_regressor(reg_weights, model_select, device):
    model_select = model_select.lower()

    if model_select not in regressor_factory:
        raise ValueError(
            f"Unsupported model_select='{model_select}'. "
            f"Choose from {list(regressor_factory.keys())}"
        )

    if model_select == "mobilenetv3":
        regressor = regressor_factory[model_select](bins=2).to(device)
    elif model_select in ["resnet", "resnet18"]:
        base_model = model_factory[model_select]
        regressor = regressor_factory[model_select](model=base_model).to(device)
    else:
        regressor = regressor_factory[model_select]().to(device)

    checkpoint = torch.load(reg_weights, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    missing, unexpected = regressor.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        LOGGER.warning(f"Missing keys while loading regressor: {missing}")
    if len(unexpected) > 0:
        LOGGER.warning(f"Unexpected keys while loading regressor: {unexpected}")

    regressor.eval()
    return regressor


def load_2d_detector(weights, data, imgsz, device):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    model.warmup(imgsz=(1, 3, *imgsz), half=False)
    return model, device, stride, names, pt, imgsz


@torch.no_grad()
def detect2d_frame(
    frame,
    model,
    device,
    stride,
    names,
    pt,
    imgsz,
    classes=None,
    conf_thres=0.25,
    iou_thres=0.45,
):
    bbox_list = []

    im0 = frame.copy()

    im = letterbox(im0, new_shape=imgsz, stride=stride, auto=pt)[0]
    im = im[:, :, ::-1].transpose(2, 0, 1)
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im).to(device)
    im = im.float() / 255.0
    if im.ndim == 3:
        im = im.unsqueeze(0)

    pred = model(im, augment=False, visualize=False)
    pred = non_max_suppression(
        pred,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        classes=classes,
        agnostic=False,
        max_det=100,
    )

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in det:
                xyxy_ = [int(x.item()) for x in xyxy]
                bbox = [(xyxy_[0], xyxy_[1]), (xyxy_[2], xyxy_[3])]
                c = int(cls.item())
                label = names[c]
                bbox_list.append(Bbox(bbox, label, float(conf.item())))

    return bbox_list


def plot3d(img, proj_matrix, box_2d, dimensions, alpha, theta_ray, img_2d=None):
    location, _ = calc_location(dimensions, proj_matrix, box_2d, alpha, theta_ray)
    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, proj_matrix, orient, dimensions, location)
    return location


@torch.no_grad()
def detect3d_video(
    det_weights,
    reg_weights,
    model_select,
    source,
    calib_file,
    data,
    imgsz,
    device,
    classes,
    conf_thres,
    iou_thres,
    show_result,
    save_result,
    output_path
):
    det_model, det_device, stride, names, pt, imgsz = load_2d_detector(
        weights=det_weights,
        data=data,
        imgsz=imgsz,
        device=device
    )

    if str(device).isdigit() and torch.cuda.is_available():
        reg_device = torch.device(f"cuda:{device}")
    elif device != "cpu" and torch.cuda.is_available():
        reg_device = torch.device("cuda")
    else:
        reg_device = torch.device("cpu")

    regressor = load_regressor(reg_weights, model_select, reg_device)

    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(2)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open input video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save_result:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create output video: {output_path}")

    frame_idx = 0

    LOGGER.info(f"Input video: {source}")
    LOGGER.info(f"2D weights : {det_weights}")
    LOGGER.info(f"3D weights : {reg_weights}")
    LOGGER.info(f"Calib file : {calib_file}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vis_frame = frame.copy()

        dets = detect2d_frame(
            frame=frame,
            model=det_model,
            device=det_device,
            stride=stride,
            names=names,
            pt=pt,
            imgsz=imgsz,
            classes=classes,
            conf_thres=conf_thres,
            iou_thres=iou_thres
        )

        for det in dets:
            (x1, y1), (x2, y2) = det.box_2d

            # Always draw 2D detection first
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            base_label = f"{det.detected_class} {det.conf:.2f}"
            cv2.putText(
                vis_frame,
                base_label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            if not averages.recognized_class(det.detected_class):
                continue

            try:
                detectedObject = DetectedObject(
                    frame,
                    det.detected_class,
                    det.box_2d,
                    str(calib_file)
                )
            except Exception as e:
                LOGGER.warning(f"Skipping object due to DetectedObject error: {e}")
                continue

            try:
                theta_ray = detectedObject.theta_ray
                input_img = detectedObject.img
                proj_matrix = detectedObject.proj_matrix
                box_2d = det.box_2d
                detected_class = det.detected_class

                if isinstance(input_img, np.ndarray):
                    input_tensor = torch.from_numpy(input_img).float()
                    if input_tensor.ndim == 3:
                        input_tensor = input_tensor.unsqueeze(0)
                    input_tensor = input_tensor.to(reg_device)
                else:
                    input_tensor = torch.zeros((1, 3, 224, 224), device=reg_device)
                    input_tensor[0] = input_img.to(reg_device)

                orient, conf, dim = regressor(input_tensor)

                orient = orient.detach().cpu().numpy()[0]
                conf = conf.detach().cpu().numpy()[0]
                dim = dim.detach().cpu().numpy()[0]

                dim += averages.get_item(detected_class)

                argmax = np.argmax(conf)
                orient = orient[argmax]
                alpha = np.arctan2(orient[1], orient[0])
                alpha += angle_bins[argmax]
                alpha -= np.pi

                location = plot3d(
                    vis_frame,
                    proj_matrix,
                    box_2d,
                    dim,
                    alpha,
                    theta_ray
                )

                if location is not None and len(location) >= 3:
                    label3d = f"{detected_class} {det.conf:.2f} | z={float(location[2]):.2f}m"
                    cv2.putText(
                        vis_frame,
                        label3d,
                        (x1, max(50, y1 - 28)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

            except Exception as e:
                LOGGER.warning(f"Skipping object due to 3D estimation error: {e}")
                continue

        cv2.putText(
            vis_frame,
            f"Frame: {frame_idx}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        if show_result:
            cv2.imshow("3D Detection Overlay", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if writer is not None:
            writer.write(vis_frame)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    LOGGER.info("Done.")
    if save_result:
        LOGGER.info(f"Saved annotated video to: {output_path}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_weights", type=str, default="/home/alien/cv_p3/ext_models/yolov5s.pt")
    parser.add_argument("--reg_weights", type=str, default="/home/alien/cv_p3/ext_models/mobilenetv3-last.pt/weights/mobilenetv3-last.pt")
    parser.add_argument("--model_select", type=str, default="resnet18", choices=["resnet", "resnet18", "vgg11", "mobilenetv3"])
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="/home/alien/cv_p3/output/annotated_3d.mp4")
    parser.add_argument("--calib_file", type=str, default="/home/alien/cv_p3/eval/camera_cal/calib_cam_to_cam.txt")
    parser.add_argument("--data", type=str, default="/home/alien/cv_p3/data/coco128.yaml")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640, 640])
    parser.add_argument("--device", default="0")
    parser.add_argument("--classes", nargs="+", type=int, default=[2, 3, 5, 7])
    parser.add_argument("--conf_thres", type=float, default=0.5)
    parser.add_argument("--iou_thres", type=float, default=0.7)
    parser.add_argument("--show_result", action="store_true")
    parser.add_argument("--save_result", action="store_true")

    opt = parser.parse_args()
    if len(opt.imgsz) == 1:
        opt.imgsz = [opt.imgsz[0], opt.imgsz[0]]
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    detect3d_video(
        det_weights=opt.det_weights,
        reg_weights=opt.reg_weights,
        model_select=opt.model_select,
        source=opt.source,
        calib_file=opt.calib_file,
        data=opt.data,
        imgsz=opt.imgsz,
        device=opt.device,
        classes=opt.classes,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        show_result=opt.show_result,
        save_result=opt.save_result,
        output_path=opt.output_path
    )


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)