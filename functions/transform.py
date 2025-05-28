
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2


def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR
    )

    return dst_img
    

def box_to_center_scale(box, model_image_width=192, model_image_height=256):
    """
    Convert detection box to center and scale for HRNet pose estimation.
    
    Args:
        box: A tuple of ((x1, y1), (x2, y2)) defining the bounding box
        model_image_width: Width of the HRNet input (default: 192)
        model_image_height: Height of the HRNet input (default: 256)
    
    Returns:
        center (numpy.ndarray): center of box [x, y]
        scale (numpy.ndarray): scale of box [width, height]
        
    Note: Scale is adjusted to make the box slightly larger to include full person
    """
    import numpy as np
    # Unpack box coordinates
    (x1, y1), (x2, y2) = box
    
    # Calculate center point
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
    
    # Calculate width and height, with slight padding
    width = (x2 - x1) * 1.1  # Add 10% padding
    height = (y2 - y1) * 1.1  # Add 10% padding
    
    # Make box square if needed (HRNet often expects square inputs)
    if width > height:
        height = width
    else:
        width = height
        
    # Convert to scale expected by HRNet
    # The scale factor is used to map back and forth between the
    # cropped image and original image
    scale = np.array([width / model_image_width, height / model_image_height])
    
    return center, scale

def box_transform_for_model(image, box, target_size=(256, 256)):
    """
    Transform an image based on a detection box to prepare it for model input.
    
    Args:
        image: Input RGB image (numpy array)
        box: Detection box in format ((x1, y1), (x2, y2))
        target_size: Size of output image (width, height)
        
    Returns:
        transformed_img: Transformed image ready for model input
        transform_matrix: Transformation matrix for mapping coordinates back
    """
    import cv2
    import numpy as np

    # Unpack box coordinates
    (x1, y1), (x2, y2) = box
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
    
    # Calculate width and height with padding
    width = (x2 - x1) * 1.1
    height = (y2 - y1) * 1.1
    
    # Make square
    if width > height:
        height = width
    else:
        width = height
    
    # Calculate transformation parameters
    src_w = width
    dst_w = target_size[0]
    dst_h = target_size[1]
    
    # Calculate scale ratio
    scale_ratio = min(dst_w / src_w, dst_h / height)
    
    # Create transformation matrices
    rot_mat = np.eye(3)  # No rotation
    
    scale_mat = np.eye(3)
    scale_mat[0, 0] = scale_ratio
    scale_mat[1, 1] = scale_ratio
    
    src_center = center
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5])
    
    trans_mat = np.eye(3)
    trans_mat[0, 2] = -src_center[0]
    trans_mat[1, 2] = -src_center[1]
    
    trans_mat_dst = np.eye(3)
    trans_mat_dst[0, 2] = dst_center[0]
    trans_mat_dst[1, 2] = dst_center[1]
    
    # Combine transformations
    affine_mat = trans_mat_dst @ scale_mat @ rot_mat @ trans_mat
    
    # Convert to 2x3 format for OpenCV
    affine_mat_cv = affine_mat[:2]
    
    # Apply transformation
    transformed_img = cv2.warpAffine(
        image, 
        affine_mat_cv, 
        (target_size[0], target_size[1]),
        flags=cv2.INTER_LINEAR
    )
    # find the new bbox size
    new_x1 = int(affine_mat_cv[0, 2])
    new_y1 = int(affine_mat_cv[1, 2])
    new_x2 = int(new_x1 + dst_w)
    new_y2 = int(new_y1 + dst_h)
    new_box = ((new_x1, new_y1), (new_x2, new_y2))
    
    return transformed_img, affine_mat_cv, new_box
