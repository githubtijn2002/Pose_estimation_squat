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
