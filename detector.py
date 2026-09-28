from ultralytics import YOLO

model = YOLO("models/best.pt")

def detect_note(image_path):

    results = model(image_path)

    for r in results:
        boxes = r.boxes

        if len(boxes) > 0:

            cls = int(boxes[0].cls)

            classes = {
                0:"10",
                1:"20",
                2:"50",
                3:"100",
                4:"200",
                5:"500",
                6:"2000"
            }

            return classes[cls]

    return "Unknown"