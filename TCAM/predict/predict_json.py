import os
import time
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from skimage import measure

from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs


def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)
    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def process_image(model, device, img_path, category_index):
    original_img = Image.open(img_path).convert('RGB')

    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print(f"Inference+NMS time for {os.path.basename(img_path)}: {t_end - t_start}")

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if "masks" in predictions:
            predict_mask = predictions["masks"].to("cpu").numpy()
            predict_mask = np.squeeze(predict_mask, axis=1)
        else:
            predict_mask = None

        if len(predict_boxes) == 0:
            print("No targets detected!")
            return None

        plot_img = draw_objs(
            original_img.copy(),
            boxes=predict_boxes,
            classes=predict_classes,
            scores=predict_scores,
            masks=predict_mask,
            category_index=category_index,
            line_thickness=3,
            font='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            font_size=20
        )

        return plot_img, predict_boxes, predict_classes, predict_scores, predict_mask


def mask_to_ten_points(mask):
    contours = measure.find_contours(mask, 0.5)
    polygons = []
    for contour in contours:
        contour = np.flip(contour, axis=1)

        if len(contour) >= 30:
            step = len(contour) // 29
            selected_contour = contour[:30*step:step]
        else:
            selected_contour = contour

        selected_contour = np.concatenate((selected_contour,selected_contour[0:1]))
        polygons.append(selected_contour.flatten().tolist())
    return polygons


def main():
    num_classes = 4
    box_thresh = 0.8
    img_folder = " "
    results_folder = " "
    weights_path = "./model_79.pth"
    label_json_path = './TCA_indices.json'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

    assert os.path.exists(weights_path), "{} file does not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    assert os.path.exists(label_json_path), "json file {} does not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    results = {
        "images": [],
        "categories": [
            {"id": 1, "name": "Table"},
            {"id": 2, "name": "Text"},
            {"id": 3, "name": "Title"},
            {"id": 4, "name": "Image"}
        ],
        "annotations": []
    }

    process_folders(model, device, img_folder, category_index, results, results_folder)


def process_folders(model, device, img_folder, category_index, results, results_folder):
    for root, dirs, files in os.walk(img_folder):
        for dir in dirs:
            folder_path = os.path.join(root, dir)
            output_folder = os.path.join(results_folder, dir)
            os.makedirs(output_folder, exist_ok=True)
            output_json_path = os.path.join(output_folder, "results.json")

            current_results = {
                "images": [],
                "categories": [
                    {"id": 1, "name": "Table"},
                    {"id": 2, "name": "Text"},
                    {"id": 3, "name": "Title"},
                    {"id": 4, "name": "Image"}
                ],
                "annotations": []
            }

            for img_id, img_file in enumerate(os.listdir(folder_path)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, img_file)
                    try:
                        plot_img, boxes, classes, scores, masks = process_image(model, device, img_path, category_index)
                    except TypeError as e:
                        print(f"Error processing image {img_path}: {e}")
                        continue

                    if plot_img is None:
                        continue

                    for annotation_id, (box, score, label, mask) in enumerate(zip(boxes, scores, classes, masks)):
                        box = box.tolist()
                        score = float(score)
                        if score < 0.8:
                            continue
                        label = int(label)
                        mask = mask.squeeze()

                        segmentation = mask_to_ten_points(mask)
                        area = np.sum(mask)
                        area = float(area)

                        current_annotation = {
                            "id": annotation_id,
                            "image_id": img_id,
                            "category_id": label,
                            "segmentation": segmentation,
                            "bbox": box,
                            "ignore": 0,
                            "iscrowd": 0,
                            "area": area
                        }

                        current_results["annotations"].append(current_annotation)

                    if plot_img:
                        result_path = os.path.join(output_folder, f"result_{img_file}")
                        plot_img.save(result_path)

                    image_info = {
                        "width": plot_img.width,
                        "height": plot_img.height,
                        "id": img_id,
                        "file_name": img_file
                    }
                    current_results["images"].append(image_info)

            with open(output_json_path, "w") as json_file:
                json.dump(current_results, json_file, indent=2)


if __name__ == '__main__':
    main()