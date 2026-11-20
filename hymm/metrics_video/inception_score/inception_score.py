import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

import numpy as np

from torchvision.models.inception import inception_v3
from scipy.stats import entropy

def load_inception_pretrained(device, pretrained_model_base_path):
    inception_model = inception_v3(pretrained=False, transform_input=False).to(device=device)
    # self.inception_model = inception_v3(pretrained=False, transform_input=True).to(device=self.device)

    pretrained_model_path = f"{pretrained_model_base_path}/inception_score/inception_v3_google-0cc3c7bd.pth"
    inception_model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device("cpu")))
    inception_model = inception_model.eval()
    return inception_model

def get_inception_score(x, inception_model, inception_resize):

    if inception_resize is not None:
        x = inception_resize(x)
    x = inception_model(x)
    return F.softmax(x, dim=-1).cpu().detach()

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=False, transform_input=False).type(dtype)
    # inception_model = inception_v3(pretrained=False, transform_input=True).type(dtype) # 比较重要

    pretrained_model_path = "assets/local/resource_9505c09baa.pth"
    inception_model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device("cpu")))

    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=-1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

