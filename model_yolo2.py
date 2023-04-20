import torch
from ultralytics import YOLO
import cv2


def load_yolo_model(weights_path, local=False, version=5):
    print("-----------------------")
    print("Loading YOLOv" + str(version) +  " model...")
    print("-----------------------")
    if version==5:
        if local:
            return torch.hub.load("../yolov5", "custom", source="local", path=weights_path, force_reload=False)
        else:
            return torch.hub.load("ultralytics/yolov5", "custom", path=weights_path, force_reload=False)
    elif version==8:
        return YOLO()




def get_yolo_prediction(img_to_predict, model):
    # Run prediction on the image
    results = model(img_to_predict)
    # Extract the predictions
    boxes = results.xyxy[0].cpu().numpy()
    return boxes


def get_boxes_position(boxes):
    placements = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        placements.append([int(x1 + (x2 - x1) / 2), int(y2), cls])
    return placements


def draw_yolo_prediction(image, boxes, model):
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        conf = float(conf)
        class_name = model.names[int(cls)]
        if conf > 0.5:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f"{class_name}: {round(conf, 2)}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def predict_image(img_to_predict, model):
    # Run prediction on the image
    results = model(img_to_predict)
    # Extract the predictions
    boxes = results.xyxy[0].cpu().numpy()
    # Draw the boxes on the image
    placements = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        placements.append([int(x1 + (x2 - x1)/2), int(y2), cls])
        class_name = model.names[int(cls)]
        if conf > 0.5:
            cv2.rectangle(img_to_predict, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img_to_predict, f"{class_name}: {conf}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img_to_predict, placements

"""
model = load_yolo_model('best4.pt')

img = cv2.imread('ChessBoard\img_test7.png', cv2.IMREAD_COLOR)


predict = predict_image(img, model)

cv2.imshow('Intersections détectées', predict)

# Attendre l'appui d'une touche pour fermer les fenêtres d'affichage
cv2.waitKey(0)
cv2.destroyAllWindows()
"""