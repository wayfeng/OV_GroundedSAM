import gradio as gr
import openvino as ov
import numpy as np
import supervision as sv
import argparse
from typing import Dict, List
from PIL import Image
from transformers import AutoTokenizer

import warnings
warnings.filterwarnings("ignore")

def normalize(arr, mean=(0,0,0), std=(1,1,1)):
    arr = arr.astype(np.float32)
    arr /= 255.0
    for i in range(3):
        arr[...,i] = (arr[...,i] - mean[i]) / std[i]
    return arr

def preprocess_image(input_image, shape=[512,512]):
    img = input_image.resize(shape, Image.Resampling.NEAREST)
    img = np.asarray(img)
    img = normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return img.transpose(2,0,1)

def load_model(model_checkpoint_path='./models/groundingdino_512.xml', device='GPU'):
    core = ov.Core()
    model_read = core.read_model(model_checkpoint_path)
    model = core.compile_model(model_read, device.upper())
    model.tokenizer = AutoTokenizer.from_pretrained('./models/tokenizer_pretrained_pytorch')
    model.max_text_len = 256
    return model

def generate_masks(tokenized, special_tokens_list):
    """Generate attention mask between each pair of special tokens
    Args:
        input_ids (np.ndarray): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.
    Returns:
        np.ndarray: attention mask between each special tokens.
    """
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = np.zeros((bs, num_token)).astype(np.bool_)
    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = np.transpose(np.nonzero(special_tokens_mask))

    # generate attention mask and positional ids
    attention_mask = np.repeat(np.eye(num_token).astype(np.bool_)[None], bs, axis=0)
    position_ids = np.zeros((bs, num_token), dtype=np.int64)
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = np.arange(0, col - previous_col)
        previous_col = col

    return attention_mask, position_ids

def get_phrases_from_posmap(
            posmap: np.ndarray,
            tokenized: Dict,
            tokenizer: AutoTokenizer,
            left_idx: int = 0, right_idx: int = 255):
    if posmap.ndim == 1:
        posmap[0: left_idx + 1] = False
        posmap[right_idx:] = False
        non_zero_idx = list(np.nonzero(posmap)[0])
        token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
        return tokenizer.decode(token_ids)
    else:
        raise NotImplementedError("posmap must be 1-dim")

# warning of np.exp(-x) overflow can be ignored
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def get_grounding_output(model, image, caption, box_threshold, text_threshold):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."

    tokenized = model.tokenizer(caption, padding="longest", return_tensors="np")
    specical_tokens = model.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    
    text_self_attention_masks, position_ids = generate_masks(tokenized, specical_tokens)

    if text_self_attention_masks.shape[1] > model.max_text_len:
        text_self_attention_masks = text_self_attention_masks[
            :, : model.max_text_len, : model.max_text_len]
        
        position_ids = position_ids[:, : model.max_text_len]
        tokenized["input_ids"] = tokenized["input_ids"][:, : model.max_text_len]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, : model.max_text_len]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : model.max_text_len]

    inputs = {}
    inputs["img"] = image[None]
    inputs["input_ids"] = tokenized["input_ids"]
    inputs["attention_mask"] = tokenized["attention_mask"]
    inputs["token_type_ids"] = tokenized["token_type_ids"]
    inputs["position_ids"] = position_ids
    inputs["text_token_mask"] = text_self_attention_masks

    outputs = model.infer_new_request(inputs)

    prediction_logits_ = sigmoid(np.squeeze(outputs["logits"], 0)) # prediction_logits.shape = (nq, 256)
    prediction_boxes_ = np.squeeze(outputs["boxes"], 0) # prediction_boxes.shape = (nq, 4)

    # filter output
    mask = prediction_logits_.max(axis=1) > box_threshold
    logits = prediction_logits_[mask]  # num_filt, 256
    boxes = prediction_boxes_[mask]  # num_filt, 4

    # get phrase
    tokenized = model.tokenizer(caption)
    phrases = [get_phrases_from_posmap(logit > text_threshold, tokenized, model.tokenizer) for logit in logits]

    return boxes, logits.max(axis=1), phrases

def box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Converts bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.
    (cx, cy) refers to center of bounding box
    (w, h) are width and height of bounding box
    Args:
        boxes (Tensor[N, 4]): boxes in (cx, cy, w, h) format which will be converted.

    Returns:
        boxes (Tensor(N, 4)): boxes in (x1, y1, x2, y2) format.
    """
    cx, cy, w, h = np.transpose(boxes)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.transpose(np.array([x1, y1, x2, y2]))

def annotate(image_source: np.ndarray, boxes: np.ndarray, logits: np.ndarray, phrases: List[str]) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * np.array([w, h, w, h])
    xyxy = box_cxcywh_to_xyxy(boxes=boxes)
    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    box_annotator = sv.BoxAnnotator()
    box_annotator.text_scale=0.5
    box_annotator.text_padding=0
    annotated_frame = box_annotator.annotate(scene=image_source, detections=detections, labels=labels)
    return annotated_frame

def run_grounding(input_image, grounding_caption, box_threshold, text_threshold):
    image = preprocess_image(input_image)
    boxes, logits, phrases = get_grounding_output(model, image, grounding_caption, box_threshold, text_threshold)
    annotated_frame = annotate(image_source=np.asarray(input_image), boxes=boxes, logits=logits, phrases=phrases)
    image_with_box = Image.fromarray(annotated_frame)
    return image_with_box

model = load_model(device='GPU')

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO demo", add_help=True)
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--share", action="store_true", help="share the app")
    args = parser.parse_args()


    block = gr.Blocks().queue()
    with block:
        gr.Markdown("# [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)")
        gr.Markdown("### Open-World Detection with Grounding DINO")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Image", sources="upload", type="pil")
                grounding_caption = gr.Textbox(label="Detection Prompt")
                run_button = gr.Button(value="Run")
                with gr.Accordion("Advanced options", open=False):
                    box_threshold = gr.Slider(
                        label="Box Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                    )
                    text_threshold = gr.Slider(
                        label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                    )
            with gr.Column():
                gallery = gr.Image(type="pil")
        run_button.click(
            fn=run_grounding,
            inputs=[input_image, grounding_caption, box_threshold, text_threshold],
            outputs=[gallery])
    block.launch(server_name='0.0.0.0', server_port=7579, debug=args.debug, share=args.share)
