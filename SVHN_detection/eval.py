from utils import *
from datasets import SVHNDataset
from tqdm import tqdm
from pprint import PrettyPrinter
import torch

pp = PrettyPrinter()

data_folder = '../Dataset/'
batch_size=32
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = './checkpoint_epoch_2.pth.tar'

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint, map_location=device)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

val_dataset = SVHNDataset(data_folder, split='val')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
    collate_fn=val_dataset.collate_fn, num_workers=workers, pin_memory=True)

def evaluate(val_loader, model, n_classes):
    """
    Evaluate
    :param test_loader: DataLoader for evaluate data
    :param model: model
    """

    model.eval()

    det_boxes = []
    det_labels = []
    det_scores = []
    true_boxes = []
    true_labels = []

    with torch.no_grad():
        for images, boxes, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)

            predicted_locs, predicted_scores = model(images)

            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs,
                predicted_scores, min_score=0.01, max_overlap=0.45, top_k=200)

            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)

        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, n_classes)

    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)

if __name__ == '__main__':
    evaluate(val_loader, model, 11)