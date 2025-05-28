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