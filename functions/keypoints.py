def extract_keypoints_from_heatmaps(heatmaps):
    """
    Extract keypoint coordinates and confidences from heatmaps
    
    Args:
        heatmaps: Model output tensor of shape [batch_size, num_joints, height, width]
    
    Returns:
        keypoints: List of [x, y, confidence] for each joint
    """
    import numpy as np
    heatmaps_np = heatmaps.detach().cpu().numpy()
    batch_size, num_joints, height, width = heatmaps_np.shape
    
    keypoints = []
    for joint_idx in range(num_joints):
        # Get heatmap for this joint
        joint_heatmap = heatmaps_np[0, joint_idx]  # Take first batch
        
        # Find the position with maximum confidence
        flat_idx = np.argmax(joint_heatmap)
        y, x = np.unravel_index(flat_idx, joint_heatmap.shape)
        
        # Get confidence value at this position
        confidence = joint_heatmap[y, x]
        
        keypoints.append([x, y, confidence])
    
    return keypoints

# Complete corrected coordinate transformation
def transform_keypoints_to_original(heatmaps, input_tensor, transform_matrix):
    """
    Transform keypoints from heatmap coordinates to original image coordinates
    Args:
        keypoints: List of keypoints in format [[x, y, confidence], ...]
        heatmaps: Model output tensor of shape [batch_size, num_joints, height, width]
        input_tensor: Input tensor to the model (used for dimensions)
        transform_matrix: Transformation matrix used to map coordinates back to original image
    Returns:
        original_keypoints: List of keypoints in original image coordinates [[x, y, confidence], ...]
        scaled_coords: Scaled coordinates in input tensor dimensions [[x, y], ...]
    """
    import numpy as np
    keypoints = extract_keypoints_from_heatmaps(heatmaps)
    # Get actual dimensions
    heatmap_height, heatmap_width = heatmaps.shape[2], heatmaps.shape[3]
    input_height, input_width = input_tensor.shape[2], input_tensor.shape[3]
    
    # Extract coordinates and confidences
    coords = np.array([[kp[0], kp[1]] for kp in keypoints])
    confidences = np.array([kp[2] for kp in keypoints])
    
    # Step 1: Scale from heatmap dimensions to input dimensions
    x_scale = input_width / heatmap_width
    y_scale = input_height / heatmap_height
    
    scaled_coords = coords.copy().astype(float)
    scaled_coords[:, 0] *= x_scale  # Scale x coordinates
    scaled_coords[:, 1] *= y_scale  # Scale y coordinates
    
    # Step 2: Apply inverse transformation to get original coordinates
    full_transform = np.vstack([transform_matrix, [0, 0, 1]])
    inverse_transform = np.linalg.inv(full_transform)
    
    original_coords = []
    for x, y in scaled_coords:
        p_hom = np.array([x, y, 1.0])
        original = inverse_transform @ p_hom
        original_coords.append([original[0], original[1]])
    
    # Combine with confidence values
    original_keypoints = []
    for (x, y), conf in zip(original_coords, confidences):
        original_keypoints.append([x, y, conf])
    
    return original_keypoints, scaled_coords

def retrieve_high_confidence_predictions(pose_model, box_model, img_dir, threshold=0.7, remove_low_confidence=False, verbose=1):
    """
    Retrieve high-confidence keypoint predictions from images in a directory.
    Args:
        pose_model: Pre-trained pose estimation model.
        box_model: Pre-trained bounding box model.
        img_dir: Directory containing images to process.
        threshold: Confidence threshold to filter predictions.
        remove_low_confidence: If False, will not remove low confidence predictions;
            elif a probability, will remove images below that probability.
    Returns:
        high_confidence_results: List of dictionaries with high-confidence keypoint predictions.
    """
    import os
    import torch
    import numpy as np
    from functions.boxmodel import get_box
    from functions.preprocess import preprocess_image, preprocess_hrnet
    high_confidence_results = []
    removed = 0
    img_list = os.listdir(img_dir)
    with torch.no_grad():
        for img_name in img_list:
            if verbose >= 2:
                print(f'Processing image: {img_name}')
            img = preprocess_image(img_dir, img_name)
            box = get_box(img, box_model)
            
            if box is None:
                continue
                
            input_tensor, transform_matrix, _ = preprocess_hrnet(img, box)
            heatmaps = pose_model(input_tensor)
            keypoints = extract_keypoints_from_heatmaps(heatmaps)
            scores = [k[2] for k in keypoints]
            confidence = np.mean(np.array(scores))
            if verbose >= 2:
                print(f'Confidence: {confidence:.2f}')
            if remove_low_confidence and confidence < remove_low_confidence:
                if verbose >= 2:
                    print(f'Removing {img_name} due to low confidence: {confidence:.2f}')
                os.remove(os.path.join(img_dir, img_name))
                removed += 1
                continue
            # Only save very high confidence predictions for finetuning
            if confidence >= threshold:
                # Transform keypoints back to original image coordinates
                original_keypoints, _ = transform_keypoints_to_original(heatmaps, input_tensor, transform_matrix            )
                
                result = {
                    "image_path": str(os.path.join(img_dir, img_name)),
                    "keypoints": original_keypoints,  # In original image coordinates
                    "scores": scores,
                    "bbox": box,
                    "confidence": confidence
                }
                high_confidence_results.append(result)
    if verbose >= 1:
        print(f'Total images processed: {len(img_list)}')
        if removed > 0:
            print(f'Removed {removed} images due to low confidence')
        print(f'Found {len(high_confidence_results)} high-confidence predictions for finetuning')
        print(f'Average confidence: {np.mean([r["confidence"] for r in high_confidence_results]):.2f}')
    return high_confidence_results