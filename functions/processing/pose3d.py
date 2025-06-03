def create_adaptive_args():
    import argparse
    args = argparse.Namespace()
    args.layers = 8
    args.channel = 512
    args.d_hid = 1024
    args.n_joints = 17
    args.out_joints = 17
    args.frames = 243  # Adapt to your video
    args.token_num = 81  # Adapt token count
    args.layer_index = 3
    args.checkpoint = 'files\\weights'
    args.image_size = [640, 480]
    return args

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]

def camera_to_world(X, R, t):
    import numpy as np
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t

def wrap(func, *args, unsqueeze=False):
    import numpy as np
    import torch
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result

def qrot(q, v):
    import torch
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape) - 1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape) - 1)
    return (v + 2 * (q[..., :1] * uv + uuv))

def pad_sequence_to_length(sequence, target_length=243):
    """Pad sequence to target length using edge padding"""
    import torch
    current_length = sequence.shape[2]  # Assuming shape is [batch, aug, frames, joints, coords]
    
    if current_length < target_length:
        # Calculate padding needed
        pad_needed = target_length - current_length
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
        
        # Pad along the frame dimension (dim=2)
        padded = torch.nn.functional.pad(sequence, (0, 0, 0, 0, pad_left, pad_right), mode='replicate')
        return padded
    elif current_length > target_length:
        # Truncate if too long
        return sequence[:, :, :target_length, :, :]
    else:
        return sequence
