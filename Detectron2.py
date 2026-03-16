import cv2
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Setup detectron2 config and model
cfg = get_cfg()

# Adjusted the path based on your directory structure
cfg.merge_from_file(r"C:\Users\musta\Documents\Weather_Effect_Generator-main\detectron2-main\configs\COCO-Detection\fast_rcnn_R_50_FPN_1x.yaml")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Set the threshold for detection (can be adjusted)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml")  # Pre-trained weights
cfg.MODEL.DEVICE = "cpu"  # Use GPU if available, or CPU

predictor = DefaultPredictor(cfg)

# Input folder with synthetic images
input_folder = r'C:\Users\musta\Documents\Weather_Effect_Generator-main\Output'  # Path to synthetic images

output_folder = r'C:\Users\musta\Documents\Weather_Effect_Generator-main\Detectron'  # Path for output images

# Iterate through all images in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):  # Assuming the images are in .jpg format
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Make predictions
        outputs = predictor(image)

        # Visualize the output
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu").pred_classes)

        # Show and save the output
        result_image = v.get_image()[:, :, ::-1]
        output_image_path = os.path.join(output_folder, f"detected_{filename}")
        cv2.imwrite(output_image_path, result_image)  # Save the output image

        print(f"Processed image: {filename}")
