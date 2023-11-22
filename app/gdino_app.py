import gradio as gr
import openvino as ov
import numpy as np
import supervision as sv
import argparse
import torch
import cv2
from typing import Dict, List
from PIL import Image
from transformers import AutoTokenizer
from torchvision.ops import box_convert
import torchvision.transforms as T

def preprocess_image(input_image, shape=[512,512]):
    transform = T.Compose([
        T.Resize(shape),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = transform(input_image)
    return image

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
        input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.
    Returns:
        torch.Tensor: attention mask between each special tokens.
    """
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((bs, num_token)).bool()
    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = (torch.eye(num_token).bool().unsqueeze(0).repeat(bs, 1, 1))
    position_ids = torch.zeros((bs, num_token))
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = torch.arange(0, col - previous_col)
        previous_col = col

    return attention_mask, position_ids.to(torch.long)

def get_phrases_from_posmap(
            posmap: torch.BoolTensor,
            tokenized: Dict,
            tokenizer: AutoTokenizer,
            left_idx: int = 0, right_idx: int = 255):
    if posmap.dim() == 1:
        posmap[0: left_idx + 1] = False
        posmap[right_idx:] = False
        non_zero_idx = posmap.nonzero(as_tuple=True)[0].tolist()
        token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
        return tokenizer.decode(token_ids)
    else:
        raise NotImplementedError("posmap must be 1-dim")

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def get_grounding_output(model, image, caption, box_threshold, text_threshold):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."

    tokenized = model.tokenizer(caption, padding="longest", return_tensors="pt")
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
    input_img = np.expand_dims(image, 0)
    inputs["img"] = input_img
    inputs["input_ids"] = tokenized["input_ids"]
    inputs["attention_mask"] = tokenized["attention_mask"]
    inputs["token_type_ids"] = tokenized["token_type_ids"]
    inputs["position_ids"] = position_ids
    inputs["text_token_mask"] = text_self_attention_masks 

    outputs = model.infer_new_request(inputs)

    prediction_logits_ = sigmoid(np.squeeze(outputs["logits"], 0)) # prediction_logits.shape = (nq, 256)
    prediction_boxes_ = np.squeeze(outputs["boxes"], 0) # prediction_boxes.shape = (nq, 4)
    logits = torch.from_numpy(prediction_logits_)
    boxes = torch.from_numpy(prediction_boxes_)

    # filter output
    mask = logits.max(dim=1)[0] > box_threshold
    logits = logits[mask]  # num_filt, 256
    boxes = boxes[mask]  # num_filt, 4

    # get phrase
    tokenized = model.tokenizer(caption)
    phrases = [get_phrases_from_posmap(logit > text_threshold, tokenized, model.tokenizer) for logit in logits]

    return boxes, logits.max(dim=1)[0], phrases

def find_index(string, lst):
    # if meet string like "lake river" will only keep "lake"
    # this is an hack implementation for visualization which will be updated in the future
    string = string.lower().split()[0]
    for i, s in enumerate(lst):
        if string in s.lower():
            return i
    return 0

def phrases2classes(phrases: List[str], classes: List[str]) -> np.ndarray:
    class_ids = []
    for phrase in phrases:
        try:
            class_ids.append(find_index(phrase, classes))
        except ValueError:
            class_ids.append(None)
    return np.array(class_ids)

def post_process_result(
        source_h: int,
        source_w: int,
        boxes: torch.Tensor,
        logits: torch.Tensor,
        phrases, classes
) -> sv.Detections:
    boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    confidence = logits.numpy()
    class_ids = phrases2classes(phrases, classes)
    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id = class_ids)

def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    box_annotator = sv.BoxAnnotator()
    box_annotator.text_scale=0.5
    box_annotator.text_padding=0
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame

def run_grounding(input_image, grounding_caption, box_threshold, text_threshold):
    sw, sh = input_image.size
    image = preprocess_image(input_image)
    classes = grounding_caption.split('.')
    boxes, logits, phrases = get_grounding_output(model, image, grounding_caption, box_threshold, text_threshold)
    annotated_frame = annotate(image_source=np.asarray(input_image), boxes=boxes, logits=logits, phrases=phrases)
    image_with_box = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    return image_with_box
    #detections = post_process_result(source_h=sh, source_w=sw, boxes=boxes, logits=logits, phrases=phrases, classes=classes)
    #box_annotator = sv.BoxAnnotator()
    #box_annotator.text_scale=0.5
    #box_annotator.text_padding=0
    #labels = [f"{p} {c:.2f}" for p, c in zip(phrases, detections.confidence.tolist())]
    #annotated_image = box_annotator.annotate(scene=img.copy(), detections=detections, labels=labels)
    #return annotated_image

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