from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import cv2
import os


imgs_path = "./data/test/imgs"
output_path = "./data/test/predictions"
os.makedirs(output_path, exist_ok=True)

# Load config from a config file
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = './model_final.pth'
cfg.MODEL.DEVICE = 'cpu'

#cfg.MODEL.RETINANET.NUM_CLASSES = 1

#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # for the Faster R-CNN

cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.7 # for the RetinaNet

# create predictor instance
predictor = DefaultPredictor(cfg)

threshold = 0.5

# load image
files = [f for f in os.listdir(imgs_path) if f.endswith(".jpg")]

for file in files:

    img_path = os.path.join(imgs_path,file)

    image = cv2.imread(img_path)

    # perform predictor
    outputs = predictor(image)

    # display the prediction
    preds = outputs["instances"].pred_classes.tolist()
    scores = outputs["instances"].scores.tolist()
    bboxes = outputs["instances"].pred_boxes

    for j , bbox in enumerate(bboxes):
        bbox = bbox.tolist()

        score = scores[j]
        pred = preds[j]

        if score > threshold:
            x1, y1, x2 , y2 = [int(i) for i in bbox]

            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),5)

            label = f"alpaca {score:.2f}"

            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
    save_path = os.path.join(output_path, file)
    cv2.imwrite(save_path, image)

print("Prediction finished")