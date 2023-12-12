from segment_anything import sam_model_registry
import torch
from pathlib import Path
import openvino as ov

ckpts = {'vit_b': './checkpoints/sam_vit_b_01ec64.pth',
         'vit_l': './checkpoints/sam_vit_l_0b3195.pth',
         'vit_h': './checkpoints/sam_vit_h_4b8939.pth'}

from typing import Tuple

class SamPredictModel(torch.nn.Module):
    def __init__(
        self,
        model,
        return_single_mask: bool
    ) -> None:
        super().__init__()
        self.mask_decoder = model.mask_decoder
        self.model = model
        self.img_size = model.image_encoder.img_size
        self.return_single_mask = return_single_mask

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.model.prompt_encoder.not_a_point_embed.weight * (
            point_labels == -1
        )

        for i in range(self.model.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.model.prompt_encoder.point_embeddings[
                i
            ].weight * (point_labels == i)

        return point_embedding

    def t_embed_masks(self, input_mask: torch.Tensor) -> torch.Tensor:
        mask_embedding = self.model.prompt_encoder.mask_downscaling(input_mask)
        return mask_embedding

    def mask_postprocessing(self, masks: torch.Tensor) -> torch.Tensor:
        masks = torch.nn.functional.interpolate(
            masks,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        return masks

    def select_masks(
        self, masks: torch.Tensor, iou_preds: torch.Tensor, num_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Determine if we should return the multiclick mask or not from the number of points.
        # The reweighting is used to avoid control flow.
        score_reweight = torch.tensor(
            [[1000] + [0] * (self.model.mask_decoder.num_mask_tokens - 1)]
        ).to(iou_preds.device)
        score = iou_preds + (num_points - 2.5) * score_reweight
        best_idx = torch.argmax(score, dim=1)
        masks = masks[torch.arange(masks.shape[0]), best_idx, :, :].unsqueeze(1)
        iou_preds = iou_preds[torch.arange(masks.shape[0]), best_idx].unsqueeze(1)

        return masks, iou_preds

    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor = None,
    ):
        sparse_embedding = self._embed_points(point_coords, point_labels)
        if mask_input is None:
            dense_embedding = self.model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                point_coords.shape[0], -1, image_embeddings.shape[0], 64
            )
        else:
            dense_embedding = self._embed_masks(mask_input)

        masks, scores = self.model.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
        )

        if self.return_single_mask:
            masks, scores = self.select_masks(masks, scores, point_coords.shape[1])
        upscaled_masks = masks #self.mask_postprocessing(masks)
        return upscaled_masks, scores


import warnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type')
    parser.add_argument('-o', '--output_path', default='/tmp/sam')

    args = parser.parse_args()

    core = ov.Core()
    model_type = args.model_type.lower()
    if model_type not in ckpts:
        print(f"Available model types are {[k for k in ckpts.keys()]}")
        exit(0)
    model_path = args.output_path

    ov_encoder_path = Path(f"{model_path}/sam_image_encoder_{model_type}.xml")
    ov_model_path = Path(f"{model_path}/sam_mask_predictor_{model_type}.xml")
    if ov_encoder_path.exists() or ov_model_path.exists():
        print(f"model files exist.")
        exit(0)

    sam = sam_model_registry[model_type](checkpoint=ckpts[model_type])
    # convert encoder
    ov_encoder = ov.convert_model(sam.image_encoder, example_input=torch.randn(1,3,1024,1024), input=(1,3,1024,1024))
    ov.save_model(ov_encoder, str(ov_encoder_path), compress_to_fp16=True)
    # convert predictor
    pred_model = SamPredictModel(sam, return_single_mask=True)
    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 2, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 2), dtype=torch.float),
    }
    predict_model = ov.convert_model(pred_model, example_input=dummy_inputs)
    ov.save_model(predict_model, str(ov_model_path), compress_to_fp16=True)

    print(f"{ov_encoder_path} and {ov_model_path} created.")