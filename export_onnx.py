# ==============================================================================
# Export best trained model to ONNX format for deployment
# Run this AFTER training is complete
# ==============================================================================

import os
import json
import torch
import torch.nn as nn
from pathlib import Path


def export_best_model_to_onnx(checkpoint_path, output_onnx_path, model_type="efficientnet_b4"):
    """
    Export a trained model checkpoint to ONNX format.
    
    Args:
        checkpoint_path: Path to best_model.pth
        output_onnx_path: Where to save the .onnx file
        model_type: One of 'efficientnet_b4', 'resnet50', 'vit_base'
    """
    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    num_classes = ckpt.get('num_classes', len(ckpt.get('class_names', [])))
    class_names = ckpt.get('class_names', [])
    
    print(f"Loading {model_type} with {num_classes} classes...")
    
    # Build model architecture
    if model_type == "efficientnet_b4":
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_name("efficientnet-b4")
        in_features = model._fc.in_features
        model._fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )
    elif model_type == "resnet50":
        from torchvision import models
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )
    elif model_type == "vit_base":
        try:
            import timm
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
        except ImportError:
            from torchvision import models
            model = models.vit_b_16(weights=None)
            model.heads.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(model.heads.head.in_features, num_classes),
            )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load trained weights
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Export to ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    
    output_dir = Path(output_onnx_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        input_names=["image"],
        output_names=["prediction"],
        dynamic_axes={"image": {0: "batch"}, "prediction": {0: "batch"}},
        opset_version=14,
    )
    
    # Save class labels alongside ONNX
    labels_path = output_dir / "class_labels.json"
    labels = {str(i): name for i, name in enumerate(class_names)}
    labels_path.write_text(json.dumps(labels, indent=2))
    
    file_size = os.path.getsize(output_onnx_path) / (1024 * 1024)
    print(f"✅ ONNX model exported: {output_onnx_path} ({file_size:.1f} MB)")
    print(f"✅ Class labels saved: {labels_path}")
    
    return output_onnx_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    parser.add_argument("--output", required=True, help="Path for output .onnx file")
    parser.add_argument("--model_type", default="efficientnet_b4",
                        choices=["efficientnet_b4", "resnet50", "vit_base"])
    args = parser.parse_args()
    
    export_best_model_to_onnx(args.checkpoint, args.output, args.model_type)
