
import torch
import os
from preprocess.data_preprocess import PredictionTransform
from postprocess.postprocessing import *

root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config.
    The predictor takes an BGR image, resizes it to the specified resolution,
    runs the model and produces a dict of predictions.

    This predictor takes care of model loading and input preprocessing for you.
    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.

    Examples:

    .. code-block:: python

        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, model, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = model
        self.model.eval()
        self.device = torch.device(self.cfg.DEVICE if torch.cuda.is_available() else "cpu")
        checkpoint_file = os.path.join(root_path, self.cfg.DETECTOR.CHECKPOINT)
        checkpoint = torch.load(checkpoint_file, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.input_format = cfg.INPUT.FORMAT
        self.transform = PredictionTransform(self.cfg.INTENSOR_SHAPE[0], self.cfg.DATASET.MEAN)
        assert self.input_format in ["RGB", "BGR"], self.input_format
        self.model.to(self.device)

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict): the output of the model
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width, _ = original_image.shape
            image = self.transform(original_image)
            inputs = torch.stack([image], dim=0)
            predictions = self.model(inputs)
            processed_predictions = []
            for predictions_per_image in predictions:
                r = detector_postprocess(predictions_per_image, height, width)
                processed_predictions.append({"instances": r})
            return processed_predictions
