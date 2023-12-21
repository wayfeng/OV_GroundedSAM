# OV_GroundedSAM
Run Grounded SAM with OpenVINO on Intel dGPU.

## To convert models

### GroundingDINO
Check https://github.com/wenyi5608/GroundingDINO for more details.

### SAM
```bash
pip install -e git+https://github.com/facebookresearch/segment-anything.git#egg=segment_anything
python convert_sam.py vit_l models/sam
```

### CLIP
```bash
pip install -e git+https://github.com/mlfoundations/open_clip.git#egg=open_clip_torch
python convert_clip.py ViT-L-14 -o models/clip
```

### ViT
```bash
pip install -e git+https://github.com/lukemelas/PyTorch-Pretrained-ViT.git#egg=pytorch_pretrained_vit
python convert_vit.py L_32 -o models/vit
```

## To benchmark models

### GroundingDINO

```bash
benchmark_app -m models/dino/groundingdino_512.xml -d GPU -data_shape img[1,3,512,512],input_ids[1,6],attention_mask[1,6],position_ids[1,6],token_type_ids[1,6],text_token_mask[1,6,6] -hint latency
```


### SAM
```bash
benchmark_app -m models/sam/sam_image_encoder_vit_l.xml -hint latency -d GPU
benchmark_app -m models/sam/sam_mask_predictor_vit_b.xml -d GPU -data_shape "[1,256,64,64]" -hint latency
```

### CLIP
```bash
benchmark_app -m models/clip_static/vit_l_14_visual.xml -hint latency -d GPU -data_shape "x[1,3,224,224]"
benchmark_app -m models/clip_static/vit_l_14_text.xml -hint latency -d GPU -data_shape "text[1,77]"
```

### ViT
```bash
benchmark_app vit/vit_b32_fp16.xml -hint latency -d GPU
```
