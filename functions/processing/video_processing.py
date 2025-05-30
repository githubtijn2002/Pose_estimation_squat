def preprocess_video(vid_path_indi, img_dir, fps=10, verbose = 1):
    """
    Preprocess a video by sampling frames at a specified FPS and saving them as images.
    :param vid_path_indi: Path to the video file.
    :param img_dir: Directory where the sampled images will be saved.
    :param fps: Frames per second to sample from the video.
    :param verbose: Level of verbosity for logging.
    """
    import cv2
    import os
    cap = cv2.VideoCapture(vid_path_indi)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get the original FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    divisor = original_fps / fps
    n_samples = int(total_frames / divisor)
    print(f'Original FPS: {original_fps}')
    print(f'Total frames in video: {total_frames}')
    print(f'Sampling {n_samples} frames at {fps} FPS from video: {vid_path_indi}')
    frame_indices = [i * (total_frames // n_samples) for i in range(n_samples)]
    sampled_frames = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            sampled_frames.append(frame)
        if verbose > 1:
            print(f'Sampled frame {i + 1}/{total_frames}')

    cap.release()
    if verbose > 0:
        print(f'Sampled {len(sampled_frames)} frames from {vid_path_indi}')

    if not sampled_frames:
        if verbose > 0:
            print(f'No frames sampled from {vid_path_indi}')
        return
    sampled_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in sampled_frames]
    img_dir_indi = os.path.join(img_dir, os.path.basename(vid_path_indi).split('.')[0])
    if not os.path.exists(img_dir_indi):
        os.makedirs(img_dir_indi)
    for i, frame in enumerate(sampled_frames):
        img_path = os.path.join(img_dir_indi, f'{os.path.basename(vid_path_indi).split(".")[0]}_{i+1}.jpg')
        cv2.imwrite(img_path, frame)

def analyze_vid_2d(vid_filepath, box_model, pose_model):
    import cv2
    import os
    import numpy as np
    from functions.boxmodel import get_box
    from functions.preprocess import preprocess_hrnet
    from functions.keypoints import transform_keypoints_to_original

    cap = cv2.VideoCapture(vid_filepath)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {vid_filepath}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Total frames in video: {total_frames}')
    data_batch = {'Keypoints': [], 'Confidence': []}
    for i in range(total_frames):
        ret, frame = cap.read()
        # halving the frames
        if i%2 != 0 or not ret:
            continue
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 480))
        box = get_box(image, box_model)
        input_tensor, transformation_matrix, _ = preprocess_hrnet(image, box)
        heatmaps = pose_model(input_tensor)
        data, _ = transform_keypoints_to_original(heatmaps, input_tensor, transformation_matrix)

        data_batch['Keypoints'].append(np.array([kp[:2] for kp in data]))
        data_batch['Confidence'].append(np.array([kp[2] for kp in data]))

        if i % 50 == 0:
            print(f'Processed frame {np.ceil(i/2)}/{total_frames/2}')
    cap.release()
    return data_batch

def revise_kpts(keypoints_batch, confidence_batch, proximity=3):
    """ Replaces low confidence keypoints with nearby keypoints based on confidence thresholds, prioritizeing higher confidence keypoints.
    Args:
        keypoints_batch (np.ndarray): Array of shape (frames, joints, coords) containing keypoints.
        confidence_batch (np.ndarray): Array of shape (frames, joints) containing confidence scores for each keypoint.
        proximity (int): Number of frames to look ahead and behind for replacing low confidence keypoints.
    Returns:
        np.ndarray: Updated keypoints array with low confidence keypoints replaced.
    """
    import numpy as np
    new_kpts = []
    indices = []
    proximity = 3
    ranges = np.array(range(-proximity, proximity + 1))
    # remove proximity from ranges
    ranges = ranges[ranges != 0]
    # sort on proximity to 6
    ranges_sortes = np.argsort(np.abs(ranges) - proximity)
    order = ranges[ranges_sortes]
    for idx, confs in enumerate(confidence_batch):
        low_conf_trigger = False
        if np.any(confs < 0.3):
            low_conf_trigger = True
            keypoints = keypoints_batch[idx]
            low_conf_indices = np.where(confs < 0.3)[0]
            # only higher than joint 10 is relevant
            low_conf_indices = low_conf_indices[low_conf_indices > 9]
            for low_conf_idx in low_conf_indices:
                confs_nearby = confidence_batch[idx-5:idx+6, low_conf_idx]
                if np.any(confs_nearby >= 0.5):
                    for r in order:
                        if confidence_batch[idx + r, low_conf_idx] >= 0.5:
                            keypoints[low_conf_idx] = keypoints_batch[idx + r, low_conf_idx]
                            break
                elif np.any(confs_nearby >= 0.3):
                    for r in order:
                        if confidence_batch[idx + r, low_conf_idx] >= 0.3:
                            keypoints[low_conf_idx] = keypoints_batch[idx + r, low_conf_idx]
                            break
                else:
                    keypoints[low_conf_idx] = keypoints_batch[idx, low_conf_idx]
            
        else:
            new_kpts.append(keypoints_batch[idx])

        if low_conf_trigger:
            new_kpts.append(keypoints)
            indices.append(idx)
    new_kpts = np.array(new_kpts)
    return new_kpts, indices
