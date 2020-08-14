
import torch
import os
from preprocess.data_preprocess_3d import PredictionTransform
from postprocess.postprocessing import *
from Archs_3D.MultiBin import MultiBin
from Archs_3D.DetectedObjects import DetectedObject

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
        self.proj_matrix = torch.tensor(self.cfg.CAMERA.KMAT, dtype=torch.float32, device=self.device)
        self.proj_matrix = torch.reshape(self.proj_matrix, (3, 4))
        assert self.input_format in ["RGB", "BGR"], self.input_format
        self.model.to(self.device)

    def __call__(self, original_image, bboxes, classes):
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
            objects = DetectedObject(original_image, classes, bboxes, self.proj_matrix,
                                     input_size=self.cfg.INTENSOR_SHAPE)
            inputs = objects.croped_imgs.to(self.device)
            classes = objects.classes.to(self.device)
            theta_rays = objects.theta_rays.to(self.device)
            bboxes = objects.bboxes
            orient, conf, dim = self.model(inputs)
            alpha, dimension = self.model.binCoder.decode(orient, conf, dim, classes)
            num = list(classes.size())[0]
            locations = []
            orients = []
            dimensions = []
            for i in range(num):
                location, _ = calc_regressed_bbox_3d(alpha[i].cpu().numpy(),
                                                     theta_rays[i].cpu().numpy(),
                                                     dimension[i].cpu().numpy(),
                                                     bboxes[i].cpu().numpy(),
                                                     self.proj_matrix.cpu().numpy())
                locations.append(location)
                orients.append(alpha[i].cpu().numpy() + theta_rays[i].cpu().numpy())
                dimensions.append(dimension[i].cpu().numpy())
            return locations, orients, dimensions
