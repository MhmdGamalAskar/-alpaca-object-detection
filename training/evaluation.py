import os
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator , inference_on_dataset
from detectron2.data import build_detection_test_loader

import util

if __name__ == "__main__":
    data_dir = "./data"
    class_list_file = "./class.names"
    output_dir = "./output"

    model_path = "./model_final.pth"

    # register datasets
    nmr_classes = util.register_datasets(data_dir, class_list_file)



    #get config
    cfg = util.get_cfg(
        output_dir=output_dir,
        learning_rate=0.00001,
        batch_size=4,
        iterations=6000,
        checkpoint_period=500,
        model='COCO-Detection/retinanet_R_101_FPN_3x.yaml',
        device='cpu',
        nmr_classes=nmr_classes
    )

    cfg.DATASETS.TEST =("test",)
    cfg.MODEL.WEIGHTS = model_path

    evaluator = COCOEvaluator("test",cfg,False,output_dir=output_dir)

    val_loader = build_detection_test_loader(cfg,"test")

    predictor = DefaultPredictor(cfg)

    results = inference_on_dataset(predictor.model , val_loader , evaluator)

    print("Evaluation results:")
    print(results)


