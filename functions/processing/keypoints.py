
def get_start_end_indices(indices):
    """Get start and end indices for each rep."""
    indices = sorted(indices)
    start = []
    end = []
    for i in range(len(indices)):
        if i == 0 or indices[i] != indices[i-1] + 1:
            start.append(indices[i])
        if i == len(indices) - 1 or indices[i] != indices[i+1] - 1:
            end.append(indices[i])
    return start, end

def get_squat_reps(indices, boxes_new, vid_filepath, box_model, pose_model, half_frames=False):
    import cv2
    import numpy as np
    from functions.boxmodel import get_box
    from functions.preprocess import preprocess_hrnet
    from functions.keypoints import transform_keypoints_to_original

    start, end = get_start_end_indices(indices)
    cap = cv2.VideoCapture(vid_filepath)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {vid_filepath}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reps_estimate = round(total_frames/300)
    print(f'Total frames in video: {total_frames}')
    divisor = reps_estimate * 2
    j = 0
    squat_reps = {}
    data_batch = {'Keypoints': [], 'Confidence': []}
    for i in range(total_frames):
        ret, frame = cap.read()
        if i % divisor*5 == 0:
            print(f'Processed frame {i+1}/{total_frames}')
        if half_frames:
            if i % 2 != 0:
                continue
        if i < start[j]:
            continue
        if i > end[j]:
            print(f'Processed rep {j+1}/{len(start)}, running from frame {start[j]} to {end[j]}')
            j += 1
            squat_reps[j] = data_batch.copy()
            data_batch = {'Keypoints': [], 'Confidence': []}
            if i >= total_frames or j >= len(start):
                break
            continue
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))
        if boxes_new[i] is not None:
            box = boxes_new[i]
        else:
            box = get_box(img, box_model)
        input_tensor, transformation_matrix, _ = preprocess_hrnet(img, box)
        heatmaps = pose_model(input_tensor)
        data, _ = transform_keypoints_to_original(heatmaps, input_tensor, transformation_matrix)
        data_batch['Keypoints'].append(np.array([kp[:2] for kp in data]))
        data_batch['Confidence'].append(np.array([kp[2] for kp in data]))
    return squat_reps

def get_even_indices(start, end):
    indices = {}
    for i in range(len(start)):
        indices[i+1] = list(range(start[i], end[i] + 1))
        # only keep indices that when % by 2 are 0
        indices[i+1] = [idx for idx in indices[i+1] if idx % 2 == 0]
    return indices