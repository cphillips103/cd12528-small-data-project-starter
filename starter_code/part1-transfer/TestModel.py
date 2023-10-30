import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.cpu()
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(2.0)

def test_model(test_loader, trained_model, class_names):
    print('showing test results')
    with torch.no_grad():
        #Get correct device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trained_model = trained_model.to(device)
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            inp = torchvision.utils.make_grid(inputs)
            outputs = trained_model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(len(inputs)):
                inp = inputs.data[i]
                imshow(inp, class_names[preds[i]])