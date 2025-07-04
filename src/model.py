import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes, model_type='resnet50'):
    if model_type == 'resnet50v2':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
