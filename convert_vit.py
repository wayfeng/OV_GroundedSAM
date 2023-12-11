import torch
import openvino as ov
from pytorch_pretrained_vit import ViT
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert ViT model to OpenVINO IR")
    parser.add_argument("model_id", default="B_32", help="Model configuration ID")
    parser.add_argument("-o", "--output_path", default="/tmp/vit", help="Path to save converted IR")
    args = parser.parse_args()
    model_id = args.model_id
    path = args.output_path
    # Load model
    model = ViT(model_id, pretrained=True)
    model.eval()

    # Convert and save OV model
    img_size = 384 if model_id.endswith("imagenet1k") else 224
    img = torch.randn([1,3,img_size,img_size])
    ov_model = ov.convert_model(model, example_input=img, input=(1,3,img_size,img_size))
    ov_path = f"{path}/vit_{model_id.lower()}_fp16.xml"
    ov.save_model(ov_model, ov_path, compress_to_fp16=True)
    ov_path = f"{path}/vit_{model_id.lower()}_fp32.xml"
    ov.save_model(ov_model, ov_path, compress_to_fp16=False)
