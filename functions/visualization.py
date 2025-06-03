def show_keypoints_on_image(img, keypoints):
    """
    Visualize keypoints on the image
    Args:
        img: Input image in RGB format
        keypoints: List of keypoints in format [[x, y, confidence], ...]
    """
    import cv2
    import numpy as np
    skeleton = [[1,3],[1,0],[2,4],[2,0],[0,5],[0,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]]
    img_org = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for idx, (x, y, _) in enumerate(keypoints):
        if x > 0 and y > 0:
            cv2.circle(img_org, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(img_org, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for pair in skeleton:
        start_idx, end_idx = pair
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_x, start_y, _ = keypoints[start_idx]
            end_x, end_y, _ = keypoints[end_idx]
            if start_x > 0 and start_y > 0 and end_x > 0 and end_y > 0:
                cv2.line(img_org, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (255, 0, 0), 2)
    cv2.imshow('Keypoints', img_org)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show2Dpose(kps, img):
    import cv2

    colors = [(138, 201, 38),
              (25, 130, 196),
              (255, 202, 58)] 

    connections = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2],
 [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

    LR = [1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3]

    thickness = 3

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), colors[LR[j]-1], thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=colors[LR[j]-1], radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=colors[LR[j]-1], radius=3)

    return img


def show3Dpose(vals, ax, fix_z):
    import numpy as np
    ax.view_init(elev=15., azim=70)

    colors = [(138/255, 201/255, 38/255),
            (255/255, 202/255, 58/255),
            (25/255, 130/255, 196/255)]

    I = np.array([15, 13, 16, 14, 11, 5, 6, 5, 5, 6, 7, 8, 1, 0, 0, 1, 2, 3, 4])
    J = np.array([13, 11, 14, 12, 12, 11, 12, 6, 7, 8, 9, 10, 2, 1, 2, 3, 4, 5, 6])

    LR = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=3, color = colors[LR[i]-1])

    RADIUS = 0.72

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

    if fix_z:
        left_z = max(0.0, -RADIUS+zroot)
        right_z = RADIUS+zroot
        # ax.set_zlim3d([left_z, right_z])
        ax.set_zlim3d([0, 1.5])
    else:
        ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])

    ax.set_aspect('equal') # works fine in matplotlib==2.2.2 or 3.7.1

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)

def showimage(ax, img):
    import matplotlib.pyplot as plt
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)



def analyze_frames_2D(vid_filepath, new_kpts, indices):
    import cv2
    from functions.processing.visualization import show2Dpose
    cap = cv2.VideoCapture(vid_filepath)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {vid_filepath}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(total_frames):
        ret, frame = cap.read()
        if i%2 != 0 or not ret:
            continue
        if i not in indices:
            continue
        keypoints_relevant = new_kpts[i//2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))
        img = show2Dpose(keypoints_relevant, img)
        cv2.imshow(f'Frame {i//2}', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
def visualize_reps(boxes, divisor, total_frames):
    import matplotlib.pyplot as plt
    import numpy as np
    from functions.processing.boxes import analyze_boxes

    check_quick = [np.array([box[i][1] for box in boxes if box is not None]) for i in range(2)]
    box_heights = check_quick[1] - check_quick[0]
    mean_height = np.mean(box_heights)
    below_mean_indices = np.where(box_heights < 0.75 * mean_height)[0]
    extra_frames = int(np.round((total_frames // divisor) * 0.05))
    min_range = below_mean_indices - extra_frames
    max_range = below_mean_indices + extra_frames


    plt.figure(figsize=(10, 5))
    plt.plot(box_heights)
    plt.axhline(mean_height, color='r', linestyle='--', label='Mean Height')
    plt.scatter(below_mean_indices, box_heights[below_mean_indices], color='orange', label='Below 75% Mean Height', zorder=5)
    # plot the ranges of extra frames around the indices
    for i in range(len(min_range)):
        plt.axvspan(min_range[i], max_range[i], color='yellow', alpha=0.3, label='Extra Frame Range' if i == 0 else "")
    plt.title('Box Heights Over Frames')
    plt.xlabel('Frame Index')
    plt.ylabel('Box Height')
    plt.legend()
    plt.show()