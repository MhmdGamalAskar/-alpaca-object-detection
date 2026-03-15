import os
import cv2
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.config import get_cfg as _get_cfg
from detectron2 import model_zoo

from training.loss import ValidationLoss
import base64
import plotly.graph_objects as go
import streamlit as st








def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def visualize(image, bboxes,scores):
    """
    Visualizes the image with bounding boxes using Plotly.

    Args:
        image: The input image.
        bboxes (list): A list of bounding boxes in the format [x1, y1, x2, y2].

    """
    # Get the width and height of the image

    width, height = image.size

    shapes = []
    annotations = []

    for bbox, score in zip(bboxes, scores):

        x1, y1, x2, y2 = bbox

        # bounding box
        shapes.append(dict(
            type="rect",
            x0=x1,
            y0=height - y2,
            x1=x2,
            y1=height - y1,
            line=dict(color='red', width=4),
        ))

        # label text
        annotations.append(dict(
            x=x1,
            y=height - y1,
            text=f"alpaca {score:.2f}",
            showarrow=False,
            font=dict(
                size=16,
                color="white"
            ),
            bgcolor="red",
            borderpad=3
        ))

    fig = go.Figure()

    fig.update_layout(
        images=[dict(
            source=image,
            xref="x",
            yref="y",
            x=0,
            y=height,
            sizex=width,
            sizey=height,
            sizing="stretch"
        )]
    )

    fig.update_xaxes(range=[0, width], showticklabels=False)
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        range=[0, height],
        showticklabels=False
    )

    fig.update_layout(
        height=800,
        shapes=shapes,
        annotations=annotations
    )

    st.plotly_chart(fig, use_container_width=True)





def get_cfg(output_dir, learning_rate, batch_size, iterations, checkpoint_period, model, device, nmr_classes):
    """
    Create a Detectron2 configuration object and set its attributes

    Args:
        output_dir (str): The path to the output directory where the trained model and logs will be saved
        learning_rate (float): The learning rate for the optimizer
        batch_size (int): The batch size used during training
        iterations (int): The maximum number of training iterations
        checkpoint_period (int): The number of iterations between consecutive checkpoints.
        model (str): The name of the model to use which should be one of the models available in Detectron2's model zoo
        device (str): The device to use for training, which should be 'cpu' or 'cuda'
        nmr_classes (int): The number of classes in the dataset

    Returns:
        The Detectron2 configuration object.
    """
    cfg = _get_cfg()

    # Merge the model's default configuration file with the default Detectron2 configuration file
    cfg.merge_from_file(model_zoo.get_config_file(model))

    # Set the training and validation datasets and exclude the test dataset
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.VAL = ("val",)
    cfg.DATASETS.TEST = ()

    # Set the device to use for training
    if device in ['cpu'] :
        cfg.MODEL.DEVICE = 'cpu'

    # set the number of dataloader workers
    cfg.DATALOADER.NUM_WORKERS = 2

    # set the model weight to ones pre-trained on COCO datasets
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

    # set the batch size used by the solver
    cfg.SOLVER.IMS_PER_BATCH = batch_size

    # set the checkpoint period
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period

    # set the base learning rate
    cfg.SOLVER.BASE_LR = learning_rate

    # set the Maximum number of training iterations
    cfg.SOLVER.MAX_ITER = iterations

    # set the learning rate scheduler steps to an empty list, which means the learning rate will not be decayed
    cfg.SOLVER.STEPS = []

    # set the batch size used by the ROI Heads during training
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    # set the number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = nmr_classes # for fasterR-CNN
    #cfg.MODEL.RETINANET.NUM_CLASSES = nmr_classes # for RetinaNet

    # set the output directory
    cfg.OUTPUT_DIR = output_dir

    return cfg



def get_dicts(img_dir , ann_dir):
    """
    Read the annotations for the dataset in YOLO format and create a list of dictionaries containing information for each image

    Args:
        img_dir (str): Directory containing the images
        ann_dir (str): Directory containing the annotations

    Returns:
        list[dict]: A list of dictionaries containing information for each image. Each dictionary has the following keys:
            - file_name: The path to the image file
            - image_id: The unique identifier for the image
            - height: The height of the image in pixels
            - width: The width of the image in pixels
            - annotations: A list of dictionaries, one for each object in the image, containing the following keys:
                - bbox: A list of four integers [x0, y0, w, h] representing the bounding box of the object in the image,
                        where (x0, y0) is the top-left corner and (w, h) are the width and height of the bounding box,
                        respectively
                - bbox_mode: A constant from the `BoxMode` class indicating the format of the bounding box coordinates
                             (e.g., `BoxMode.XYWH_ABS` for absolute coordinates in the format [x0, y0, w, h])
                - category_id: The integer ID of the object's class
    """
    dataset_dicts = []
    for idx , file in enumerate(os.listdir(ann_dir)):
        # annotation should be provided in yolo format
        record = {}
        filename = os.path.join(img_dir , file[:-4]+'.jpg')
        height , width = cv2.imread(filename).shape[:2]

        record['file_name'] = filename
        record['image_id'] = idx
        record['height'] = height
        record['width'] = width

        objs = []
        with open(os.path.join(ann_dir , file)) as r:
            lines = [l[:-1] for l in r.readlines()]

        for _ , line in enumerate(lines):
            if len(line) > 2:
                label , cx ,cy ,w_ , h_ = line.split(' ')

                obj = {
                    "bbox" : [
                        int((float(cx) - (float(w_) / 2)) * width) ,
                        int((float(cy) - (float(h_) / 2)) * height) ,
                        int(float(w_) * width) ,
                        int(float(h_) * height)
                    ] ,
                    "bbox_mode" : BoxMode.XYWH_ABS ,
                    "category_id" : int(label)
                }

                objs.append(obj)
        record['annotations'] = objs
        dataset_dicts.append((record))
    return  dataset_dicts





def register_datasets(root_dir , class_list_file):
    """
    Registers the train and validation datasets and returns the number of classes

    Args:
        root_dir (str): Path to the root directory of the dataset
        class_list_file (str): Path to the file containing the list of class names

    Returns:
        int: The number of classes in the dataset
    """
    # Read the list of class name from the class list file
    with open(class_list_file , 'r') as reader:
        classes_ = [l[:-1] for l in reader.readlines()  ]

    # Register the trian and validation dataset
    for d in ['train' , 'val','test']:
        DatasetCatalog.register(
            d,
            lambda d=d: get_dicts(
                os.path.join(root_dir, d, 'imgs'),
                os.path.join(root_dir, d, 'anns')
            )
        )
        # set the metadata for dataset
        MetadataCatalog.get(d).set(thing_classes=classes_)

    return len(classes_)




def train(output_dir , data_dir , class_list_file , learning_rate ,
          batch_size , iterations , checkpoint_period , device , model):
    """
    Train a Detectron2 model on a custom dataset

    Args:
        output_dir (str): Path to the directory to save the trained model and output files
        data_dir (str): Path to the directory containing the dataset
        class_list_file (str): Path to the file containing the list of class names in the dataset
        learning_rate (float): Learning rate for the optimizer
        batch_size (int): Batch size for training
        iterations (int): Maximum number of training iterations
        checkpoint_period (int): Number of iterations after which to save a checkpoint of the model
        device (str): Device to use for training (e.g., 'cpu' or 'cuda')
        model (str): Name of the model configuration to use. Must be a key in the Detectron2 model zoo

    Returns:
        None
    """
    # Register The Data Set and get the Number of classes
    nmr_classes = register_datasets(root_dir = data_dir , class_list_file = class_list_file)

    # Get the Configuration for the model
    cfg = get_cfg(output_dir = output_dir , learning_rate = learning_rate , batch_size = batch_size ,
                  iterations = iterations , nmr_classes = nmr_classes , checkpoint_period = checkpoint_period  ,
                  device = device , model = model
                  )


    # creat the output dir
    os.makedirs(cfg.OUTPUT_DIR , exist_ok=True)

    # create the trainer object
    trainer = DefaultTrainer(cfg)

    # create a custom validation loss object
    val_loss = ValidationLoss(cfg)

    # register the custom validation loss object as a hook to the trainer
    trainer.register_hooks([val_loss])

    # swap the positions of the evaluation and checkpointing hooks so that validation loss is logged correctly
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

    # resume training from a checkpoint or load the initial model weight
    trainer.resume_or_load(resume=False)

    # train the model
    trainer.train()