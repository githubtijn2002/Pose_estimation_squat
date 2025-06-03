def random_sample(video_path, output_path, n_samples=5, spread='rand'):
    """
    Randomly samples frames from a video file.
    :param video_path: Path to the video file.
    :param n_samples: Number of frames to sample.
    :param spread: Method of sampling ('rand' for random, 'even' for evenly spaced).
    :return: List of sampled frames.
    """
    import cv2
    import random
    import os
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_samples > total_frames:
        raise ValueError("Number of samples exceeds total frames in the video.")
    if spread == 'rand':
        frame_indices = sorted(random.sample(range(total_frames), n_samples))
    elif spread == 'even':
        frame_indices = [i * (total_frames // n_samples) for i in range(n_samples)]
    else:
        raise ValueError("Invalid spread method. Use 'rand' or 'even'.")
    sampled_frames = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            sampled_frames.append(frame)

    cap.release()
    # save the sampled frames as .jpg files in the output path
    video_name = os.path.basename(video_path).split('.')[0]
    for i, frame in enumerate(sampled_frames):
        cv2.imwrite(os.path.join(output_path, f"{video_name}_{i+1}.jpg"), frame)
        
    print(f"Sampled {n_samples} frames from {video_path} and saved to {output_path}.")

def vid_loop(video_dir, image_dir, n_samples=10, spread='even'):
    """
    Loops through all videos in a directory and samples frames from each video.
    :param video_dir: Path to the directory containing video files.
    :param image_dir: Path to the directory where images will be saved.
    :param n_samples: Number of frames to sample from each video.
    :param spread: Method of sampling ('rand' for random, 'even' for evenly spaced).
    """    
    import os
    for video in os.listdir(video_dir):
        if not video.endswith(('.mp4', '.avi', '.mov')):
            print(f"Skipping non-video file: {video}")
            continue
        video_path = os.path.join(video_dir, video)

        # check if the output path exists, if not create it
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        # call the random_sample function
        random_sample(video_path, image_dir, n_samples, spread)

def check_vids(box_model, pose_model, vid_path, img_path, n_check=5, n_lim=3, conf_thresh=0.2, idx=(4, -1)):
    import os
    from functions.boxmodel import get_box
    from functions.preprocess import preprocess_image
    blacklist = []
    for vid_file in os.listdir(vid_path):
        vid_filepath = os.path.join(vid_path, vid_file)
        if not os.path.exists(vid_path):
            print(f"Video path {vid_path} does not exist. Skipping...")
            continue
        box_count = 0
        conf_count = 0
        random_sample(vid_filepath, img_path, n_samples=n_check, spread='even')
        for image_file in os.listdir(img_path):
            image = preprocess_image(img_path, image_file)
            box = get_box(image, box_model)
            if box is None:
                box_count += 1
                if box_count >= n_lim:
                    print(f"Too many boxes not detected in {vid_filepath}. Skipping video...")
                    blacklist.append(vid_filepath)
                    break
                continue
            if check_hrnet(box, pose_model, image, idx) < conf_thresh:
                conf_count += 1
                if conf_count >= n_lim:
                    print(f"Too many low confidence detections in {vid_path}. Skipping video...")
                    blacklist.append(vid_filepath)
                    break
        print(f"Checked {vid_filepath}: {box_count} boxes not detected, {conf_count} low confidence detections.")
        empty_imgdir(img_path)
    return blacklist

def check_hrnet(box, pose_model, image, idx=(0,-1)):
    import numpy as np
    from functions.preprocess import preprocess_hrnet
    from functions.keypoints import transform_keypoints_to_original
    start_idx, end_idx = idx
    input_tensor, transformation_matrix, _ = preprocess_hrnet(image, box)
    heatmaps = pose_model(input_tensor)
    original_keypoints, _ = transform_keypoints_to_original(heatmaps, input_tensor, transformation_matrix)
    conf = np.mean(np.array([kp[2] for kp in original_keypoints[start_idx:end_idx]]))
    return conf

def empty_imgdir(img_path):
    import os
    for file in os.listdir(img_path):
        file_path = os.path.join(img_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            
def remove_blacklist(blacklist):
    import os
    for vid_filepath in blacklist:
        if os.path.exists(vid_filepath):
            os.remove(vid_filepath)
            print(f"Removed {vid_filepath} from blacklist.")
        else:
            print(f"File {vid_filepath} does not exist.")

def test_vid_check(vid_path, img_path, box_model, pose_model, n_check=5, det_lim=3, conf_thresh=0.25, idx=(4, -1)):
    import os
    empty_imgdir(img_path)
    idx = (4, -1)
    n_check = 5
    det_lim = 3
    conf_thresh = 0.25
    for path in [img_path, vid_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    blacklist = check_vids(box_model=box_model, pose_model=pose_model, vid_path=vid_path, img_path=img_path,
                        n_check=n_check, n_lim=det_lim, conf_thresh=conf_thresh, idx=idx)
    remove_blacklist(blacklist)

