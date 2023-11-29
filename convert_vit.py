import torch
from pytorch_pretrained_vit import ViT

from pathlib import Path
import openvino as ov

import sys


if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_id = str(sys.argv[1]).strip()
    else:
        exit(-1)
    as_fp16 = True if len(sys.argv) < 2 else int(sys.argv[2]) == 0
    output_path = Path('/tmp/') if len(sys.argv) < 3 else Path(sys.argv[3].strip())

    model = ViT(model_id, pretrained=True)
    model.eval()

    img_sz = 384 if model_id.endswith('imagenet1k') else 224
    img = torch.randn([1, 3, img_sz, img_sz])
    ov_model = ov.convert_model(model, example_input=img, input=[1,3,img_sz,img_sz])
    if output_path.exists() and output_path.is_dir():
        output_path += f"vit_{model_id}_fp{16 if as_fp16 else 32}.xml"
        ov.save_model(ov_model, str(output_path), compress_to_fp16=as_fp16)
    else:
        ov.save_model(ov_model, f"/tmp/vit_{model_id}_fp{16 if as_fp16 else 32}.xml")