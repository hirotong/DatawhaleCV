# !/usr/bin/env python
# !-*-coding:utf-8-*-
# @ Date: 2020/5/27 9:46
# @ Author: hiro.tong
# @ Description:

from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def detect(model, original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    normalize = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    image = normalize(original_image)

    image = image.to(device)
    with torch.no_grad():
        predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
        max_overlap=max_overlap, top_k=top_k)

    det_boxes = det_boxes[0].to('cpu')

    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = original_dims * det_boxes

    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    if det_labels == ['background']:
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("MONACO.TTF")

    for i in range(det_boxes.size(0)):
        if suppress is not None and det_labels[i] in suppress:
            continue

        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[det_labels[i]])

        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
            box_location[1]]

        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white', font=font)
    del draw
    return annotated_image

if __name__ == '__main__':
    img_path = '../Dataset/mchar_train/000020.png'
    checkpoint = './checkpoint_epoch_14.pth.tar'
    checkpoint = torch.load(checkpoint, map_location=device)
    start_epoch = checkpoint['epoch'] + 1
    print('\nLoaded checkpoint from epoch %d.\n' %start_epoch)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    detect(model, original_image, min_score=0.05, max_overlap=0.4, top_k=5).show()