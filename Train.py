import random
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import metrics
from Tool import CalculateConfusionMatrix

from ultralytics import YOLO
random_color = lambda: [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]


def PrepareModel():
    model = YOLO("model/yolov8n-cls.yml")
    model = model.load("weight/yolov8n-cls.pt")
    return model


def TrainModel(model, config):
    results = model.train(**config)
    return results


if "__main__" == __name__:
    # Argümanların override sırası: args >  "cfg" > "defaults.yml"
    config = {
        "data" : "./dataset/VR01/Real",
        "epochs" : 1,
        "imgsz" : 224,
        "batch" : 32,
        "project" : "./results",
        "name" : "classification_train",
        "verbose" : True,
        "plots" : True,
        "save" : True,
        "cfg" : "cfg/classification_train.yml"
    }

    # Load Model
    model = PrepareModel()
    
    # Train Model
    results = TrainModel(model, config)
    
    # Confusion Matrix
    confusion_matrix = results.confusion_matrix.matrix
    labels = model.names
    
    RESULTS = CalculateConfusionMatrix(confusion_matrix, transpoze=True, labels=labels)




