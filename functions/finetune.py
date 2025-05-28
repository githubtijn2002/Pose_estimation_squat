def create_coco_annotation(results, split, dataset_dir="finetune_dataset"):
        """Create COCO annotation format from results"""
        import os
        import shutil
        import cv2

        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{
                "id": 1,
                "name": "person",
                "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                             "left_wrist", "right_wrist", "left_hip", "right_hip",
                             "left_knee", "right_knee", "left_ankle", "right_ankle"],
                "skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],
                           [6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],
                           [2,4],[3,5],[4,6],[5,7]]
            }]
        }
        
        for idx, result in enumerate(results):
            img_path = result["image_path"]
            img_name = os.path.basename(img_path)
            
            # Copy image to appropriate directory
            dest_path = f"{dataset_dir}/images/{split}/{img_name}"
            shutil.copy2(img_path, dest_path)
            
            # Read image to get dimensions
            img = cv2.imread(img_path)
            height, width = img.shape[:2]
            
            # Add image info
            image_info = {
                "id": idx,
                "file_name": f"{split}/{img_name}",
                "width": width,
                "height": height
            }
            coco_data["images"].append(image_info)
            
            # Add annotation
            keypoints_flat = []
            for kp in result["keypoints"]:
                keypoints_flat.extend([kp[0], kp[1], 2])  # x, y, visibility (2 = visible)
            
            bbox = result["bbox"]
            x1, y1 = bbox[0]
            x2, y2 = bbox[1]
            bbox_coco = [x1, y1, x2-x1, y2-y1]  # COCO format: [x, y, width, height]
            
            annotation = {
                "id": idx,
                "image_id": idx,
                "category_id": 1,
                "keypoints": keypoints_flat,
                "num_keypoints": 17,
                "bbox": bbox_coco,
                "area": (x2-x1) * (y2-y1),
                "iscrowd": 0
            }
            coco_data["annotations"].append(annotation)
        
        return coco_data

def create_finetune_dataset(results, dataset_dir="finetune_dataset"):
    """
    Create a finetune dataset in COCO format from the given results.
    Args:
        results: List of dictionaries containing keypoints and image paths.
                 Each dictionary should have the format:
                 {
                     "image_path": "path/to/image.jpg",
                     "keypoints": [[x1, y1, confidence1], [x2, y2, confidence2], ...],
                     "bbox": [[x1, y1], [x2, y2]]  # Bounding box coordinates
                    }
        dataset_dir: Directory to save the finetune dataset.
    Returns:
        dataset_dir: Path to the created finetune dataset directory.
    """
    import os
    import json
    from sklearn.model_selection import train_test_split
    
    # Create directories
    os.makedirs(f"{dataset_dir}/images/train", exist_ok=True)
    os.makedirs(f"{dataset_dir}/images/val", exist_ok=True)
    os.makedirs(f"{dataset_dir}/annotations", exist_ok=True)
    
    # Split into train/val
    train_results, val_results = train_test_split(results, test_size=0.2, random_state=42)
    
    # Create train and val annotations
    train_coco = create_coco_annotation(train_results, "train", dataset_dir)
    val_coco = create_coco_annotation(val_results, "val", dataset_dir)
    
    # Save annotation files
    with open(f"{dataset_dir}/annotations/train.json", 'w') as f:
        json.dump(train_coco, f, indent=2)
    
    with open(f"{dataset_dir}/annotations/val.json", 'w') as f:
        json.dump(val_coco, f, indent=2)
    
    print(f"Created finetune dataset with {len(train_results)} train and {len(val_results)} val samples")
    return dataset_dir

def finetune_model(original_model, finetune_cfg, verbose=1):
    """
    Finetune the PoseHighResolutionNet model with the provided configuration.
    Args:
        finetune_cfg: Configuration object containing finetuning parameters.
    Returns:
        model: The finetuned PoseHighResolutionNet model.
    """
    import torch.optim as optim
    import os
    import torch


    from functions.models.pose_hrnet import PoseHighResolutionNet

    CTX = finetune_cfg.MODEL.DEVICE
    # Create output directories
    os.makedirs(finetune_cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(finetune_cfg.LOG_DIR, exist_ok=True)
    
    # Initialize model
    model = PoseHighResolutionNet(finetune_cfg)
    
    # Load your current weights (the pretrained model)
    model.load_state_dict(original_model.state_dict())
    model = model.to(CTX)
    
    # Freeze early layers, only finetune final layers
    for name, param in model.named_parameters():
        if 'final_layer' in name or 'stage4' in name:
            param.requires_grad = True  # Finetune these layers
        else:
            param.requires_grad = False  # Freeze these layers
    
    # Optimizer for unfrozen parameters only
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=finetune_cfg.TRAIN.LR,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=finetune_cfg.TRAIN.LR_STEP,
        gamma=finetune_cfg.TRAIN.LR_FACTOR
    )
    
    # Loss function (you'll need to implement this based on your needs)
    criterion = torch.nn.MSELoss()
    
    print("Starting finetuning...")
    model.train()
    
    # You would load your dataset here and create data loaders
    # For now, this is a simplified training loop structure
    for epoch in range(finetune_cfg.TRAIN.END_EPOCH):
        scheduler.step()
        if verbose >= 2:
            print(f"Epoch {epoch+1}/{finetune_cfg.TRAIN.END_EPOCH}")
        for batch_idx in range(finetune_cfg.TRAIN.BATCH_SIZE):
            if verbose >= 2:
                print(f"Processing batch {batch_idx + 1}")

        # Your training loop would go here
        # for batch in train_loader:
        #     optimizer.zero_grad()
        #     outputs = model(batch['input'])
        #     loss = criterion(outputs, batch['target'])
        #     loss.backward()
        #     optimizer.step()
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(finetune_cfg.OUTPUT_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(finetune_cfg.OUTPUT_DIR, 'finetuned_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Finetuning complete! Saved final model: {final_model_path}")
    
    return model


import torch.nn as nn
class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints