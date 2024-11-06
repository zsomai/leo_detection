from ultralytics import YOLO
import cv2
from glob import glob
import os
import numpy as np
from tqdm import tqdm
import yaml
import argparse

def sliding_window(image, step_size, window_size):
    h, w = window_size
    image_h, image_w = image.shape[:2]

    for y in range(0, image_h, step_size):
        for x in range(0, image_w, step_size):
            window = image[y:min(y + h, image_h), x:min(x + w, image_w)]

            pad_h = h - window.shape[0]
            pad_w = w - window.shape[1]

            if pad_h > 0 or pad_w > 0:
                window = np.pad(window, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

            yield (x, y, window)

def visualize_results(img, x,y, res, show_conf):
    for r in res:
        boxes = r.boxes.xyxy
        confidences = r.boxes.conf
        for i, b in enumerate(boxes):
            cv2.rectangle(img, (int(b[0] + x), int(b[1] + y)), (int(b[2] + x), int(b[3] + y)), (255, 0, 255), 5)
            if show_conf:
                conf_text = f"{confidences[i]:.2f}"
                text_position = (int(b[0] + x), int(b[1] + y) - 10)
                cv2.putText(img, conf_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

def main(args):
    os.makedirs(args['output_dir'], exist_ok=True)
    model = YOLO(args['model'])
    ims = glob(f'{args["img_dir"]}/*.jpg')
    for im in tqdm(ims):
        fname= os.path.basename(im)
        img = cv2.imread(im)
        img2 = img.copy()
        w, h = 640, 640
        results = []
        for (x, y, window) in sliding_window(img, step_size=600, window_size=(w, h)):
            res = model(window, conf=args['confidence'], verbose = False)
            results += [(x,y,res)]
        for (x,y,res) in results:
            visualize_results(img2, x,y,res, args['show_confidence_of_predictions'])
        cv2.imwrite(f'{args["output_dir"]}/{fname}', img2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool to detect low earth orbit satelites")
    parser.add_argument("config_file", type=str, help="Path to the config.yaml file")
    args = parser.parse_args()
    with open(args.config_file, 'r') as file:
        config_data = yaml.safe_load(file)
    main(config_data)
