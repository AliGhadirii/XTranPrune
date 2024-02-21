import numpy as np
from Explainability.ViT_CX.cam.base_cam import BaseCAM


class get_feature_map(BaseCAM):
    def __init__(
        self,
        model,
        target_layers,
        use_cuda=False,
        reshape_transform=None,
        num_classes=3,
    ):
        super(get_feature_map, self).__init__(
            model, target_layers, use_cuda, reshape_transform, num_classes=num_classes
        )

    def get_cam_weights(
        self, input_tensor, target_layer, target_category, activations, grads
    ):
        return np.mean(grads, axis=(2, 3))
