from scipy import stats
import torchvision
import torch
import numpy as np


def accuracy(dataloaders_x, cgan, BATCH_SIZE=64, device="cuda"):
    total = 0
    count = len(dataloaders_x["test"])
    images = []
    gt = []
    defense_preds = []
    actual_preds = []
    for i, batch in enumerate(dataloaders_x["test"]):
        if i == count - 1:
            break

        inputs, classes = batch[0], batch[1]

        inputs = inputs.to(device)
        generated, orig = cgan.inference(inputs, batch_size=BATCH_SIZE)
        modes = stats.mode(generated)
        defense_pred = modes.mode
        # for j in range(BATCH_SIZE):
        #     # m = stats.mode(
        #     #     [
        #     #         generated[0][j],
        #     #         generated[1][j],
        #     #         generated[2][j],
        #     #         generated[3][j],
        #     #         generated[4][j],
        #     #     ]
        #     # )
        #     m = stats.mode(generated[:, j])
        #     if classes[j] == torch.tensor(m[0]):
        #         total += 1
        #     if i == 1:
        #         predicted.append((int(m[0])))
        # datapoints collection to plot
        images.append(inputs.detach().cpu().numpy())
        # generated_images.append(generated[0])
        gt.append(classes.detach().cpu().numpy())
        defense_preds.append(defense_pred.reshape(-1))
        actual_preds.append(orig)
    groundtruth = np.hstack(gt)
    defense_preds = np.hstack(defense_preds)
    actual_preds = np.hstack(actual_preds)
    images = np.vstack(images)

    classifier_accuracy = (groundtruth == actual_preds).sum() / groundtruth.shape[0]
    defense_accuracy = (groundtruth == defense_preds).sum() / groundtruth.shape[0]

    print("Classifier accuracy:", classifier_accuracy)
    print("Defense Accuracy:", defense_accuracy)
    return groundtruth, actual_preds, defense_preds
    # return accuracy, image[0], generated_image[0], np.array(actual[0])


# accuracy(images, generated_images,actual, predicted=accuracy(dataloaders,cgan))# real
