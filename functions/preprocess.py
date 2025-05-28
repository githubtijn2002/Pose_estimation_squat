def preprocess_image(img_dir, img_name):
    """
    Preprocess an image for model input.
    :param img_dir: Directory containing the image.
    :param img_name: Name of the image file.
    :return: Preprocessed image as a numpy array.
    """

    import os
    import cv2
    img_path = os.path.join(img_dir, img_name)
    img_org = cv2.imread(img_path)
    img_per = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_per, (640, 480))  # Resize to a fixed size
    return img

def preprocess_hrnet(img, box):
    """
    Preprocess an image and bounding box for HRNet model input.
    :param img: Input image as a numpy array.
    :param box: Bounding box coordinates as a tuple of two tuples ((x1, y1), (x2, y2)).
    :return: Preprocessed image tensor, transformation matrix, and new bounding box.
    """
    import torch
    from functions.transform import box_transform_for_model
    from torchvision import transforms
    # Transform the image for model input
    CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # box failsafe if it is None -> whole image is box
    if box is None:
        box = ((0, 0), (img.shape[1], img.shape[0]))
    model_input, transform_matrix, new_box = box_transform_for_model(img, box)
    input_tensor = torch.from_numpy(model_input).permute(2, 0, 1).float() / 255.0
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    input_tensor = normalize(input_tensor).unsqueeze(0).to(CTX)
    return input_tensor, transform_matrix, new_box