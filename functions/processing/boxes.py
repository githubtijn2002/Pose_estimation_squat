def analyze_boxes(vid_filepath, box_model):
    import cv2
    import numpy as np
    from functions.boxmodel import get_box
    boxes = []
    cap = cv2.VideoCapture(vid_filepath)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {vid_filepath}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reps_estimate = round(total_frames/300)
    print(f'Total frames in video: {total_frames}')
    divisor = reps_estimate * 2
    for i in range(total_frames):
        ret, frame = cap.read()
        # halving the frames
        if i%divisor != 0 or not ret:
            continue
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 480))
        box = get_box(image, box_model)
        boxes.append(box)


        if i % (10 * divisor) == 0:
            print(f'Processed frame {np.ceil(i/divisor)}/{np.ceil(total_frames/divisor)}')
    print(f'Processed frame {np.ceil(total_frames/divisor)}/{np.ceil(total_frames/divisor)}')
    cap.release()
    return boxes, divisor, total_frames

def retrieve_reps(boxes, divisor, total_frames):
    import numpy as np

    check_quick = [np.array([box[i][1] for box in boxes if box is not None]) for i in range(2)]
    box_heights = check_quick[1] - check_quick[0]
    mean_height = np.mean(box_heights)
    below_mean_indices = np.where(box_heights < 0.75 * mean_height)[0]

    extra_frames = int(np.round((total_frames // divisor) * 0.05))
    min_range = below_mean_indices - extra_frames
    max_range = below_mean_indices + extra_frames

    values = set()
    for i in range(len(min_range)):
        # add the values to a set
        values.update(range(min_range[i], max_range[i]+1))
    # remove values that are out of bounds
    values = [v for v in values if 0 <= v < (total_frames // divisor)]

    values = sorted(values)
    # Create a list to hold boxes for all frames, initialized with None
    boxes_new = [None] * total_frames
    # Assign the boxes to their corresponding frame positions
    for box_idx in values:
        frame_idx = box_idx * divisor
        if frame_idx < total_frames:
            boxes_new[frame_idx] = boxes[box_idx]

    indices = set()
    for i in range(len(min_range)):
        # add the values to a set
        indices.update(range(int(min_range[i]*10), int((max_range[i]+1)*10)))
    indices = sorted(indices)
    return boxes_new, indices, divisor, total_frames