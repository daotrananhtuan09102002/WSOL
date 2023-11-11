from PIL import Image
import torch
from torchvision import transforms
import wsod
import numpy as np
import cv2


_IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
_IMAGE_STD_VALUE = [0.229, 0.224, 0.225]


def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam


def t2n(t):
    return t.detach().cpu().numpy().astype(np.float32)



def transform_image_pil(image_path):
    # Open the image using PIL
    image_pil = Image.open(image_path)

    my_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # Apply the transformations to the PIL Image
    transformed_image = my_transforms(image_pil).unsqueeze(0)

    return transformed_image


def plot_cam(model, image_path, label):
    # read image
    img = transform_image_pil(image_path)
    img = img.cuda()

    # get cam
    output = model.forward(img, labels=label, return_cam=True)
    
    cam = output['cams'][0].cpu().detach().numpy()

    cam_resized = cv2.resize(t2n(output['cams'][0]), img.shape[2:],
                            interpolation=cv2.INTER_CUBIC)
    cam_normalized = normalize_scoremap(cam_resized)
    image_orig = img[0].cpu().detach() * torch.tensor(_IMAGE_STD_VALUE).view(3, 1, 1) + torch.tensor(_IMAGE_MEAN_VALUE).view(3, 1, 1)


    heatmap = cv2.applyColorMap(np.uint8(255*cam_normalized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[...,::-1]

    overlayed = 0.5 * image_orig.permute(1, 2, 0).numpy() + 0.3 * heatmap

    return overlayed