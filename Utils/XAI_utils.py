import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from ..Models.ViT_LRP.ViT_explanation_generator import LRP


def generate_visualization(attribution_generator, original_image, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(
        original_image.unsqueeze(0).cuda(),
        method="transformer_attribution",
        index=class_index,
    ).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(
        transformer_attribution, scale_factor=16, mode="bilinear"
    )
    transformer_attribution = (
        transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    )
    transformer_attribution = (
        transformer_attribution - transformer_attribution.min()
    ) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (
        image_transformer_attribution - image_transformer_attribution.min()
    ) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def show_explanation_sample(model, dataloader):
    attribution_generator = LRP(model)
    sample = next(iter(dataloader))
    image = sample["image"]
    label = sample["high"]

    vis = generate_visualization(attribution_generator, image, class_index=label)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image)
    axs[0].axis("off")
    axs[1].imshow(vis)
    axs[1].axis("off")
