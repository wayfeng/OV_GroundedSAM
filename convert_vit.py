import torch
import openvino as ov
from pytorch_pretrained_vit import ViT
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        exit(-1)
    model_id = sys.argv[1].strip()
    #save_as_fp16 = bool(int(sys.argv[2]) == 0)

    # Load model
    model = ViT(model_id, pretrained=True)
    model.eval()

    # Convert and save OV model
    img_size = 384 if model_id.endswith("imagenet1k") else 224
    img = torch.randn([1,3,img_size,img_size])
    ov_model = ov.convert_model(model, example_input=img)
    #ov_path = f"/tmp/vit_{model_id.lower()}_fp{16 if save_as_fp16 else 32}.xml"
    ov_path = f"/tmp/vit_{model_id.lower()}_fp16.xml"
    ov.save_model(ov_model, ov_path, compress_to_fp16=True)
    ov_path = f"/tmp/vit_{model_id.lower()}_fp32.xml"
    ov.save_model(ov_model, ov_path, compress_to_fp16=False)
