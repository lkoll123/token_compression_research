from transformers import CLIPVisionModel, CLIPImageProcessor
import torch


class FastV(torch.nn.Module):
    def __init__(self, model_name" str="openai/clip-vit-large-patch14-336") -> None:
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        processed_images = self.processor(images=images, return_tensors="pt")


        outputs = self.model(**processed_images)