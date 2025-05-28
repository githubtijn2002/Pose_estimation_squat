
def preprocess_pdb(img):
    import torch
    CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.to(CTX).unsqueeze(0)
    img_tensor = img_tensor.permute(0, 2, 3, 1)
    img_tensor = img_tensor.permute(0, 3, 1, 2) 
    img_tensor = img_tensor.to(CTX)
    return img_tensor

def get_pdb_boxes(img_tensor, box_model):
    import torch
    with torch.no_grad():
        predictions = box_model(img_tensor)
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    boxes = boxes[scores > 0.5]
    return boxes

def largest_box(boxes):
    if len(boxes) == 0:
        return None
    largest = boxes[0]
    for box in boxes:
        if (box[2] - box[0]) * (box[3] - box[1]) > (largest[2] - largest[0]) * (largest[3] - largest[1]):
            largest = box
    return largest

def get_box(img, box_model):
    img_tensor = preprocess_pdb(img)
    boxes = get_pdb_boxes(img_tensor, box_model)
    largest = largest_box(boxes)
    if largest is None:
        return None
    x1, y1, x2, y2 = map(int, largest)
    return (x1, y1), (x2, y2)