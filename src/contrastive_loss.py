import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function based on Video lectures from TUM :
    https://youtu.be/6e65XfwmIWE?t=1093
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean(label * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.max((torch.pow(self.margin, 2) - torch.pow(euclidean_distance, 2)), 0))


        return loss_contrastive


class TripletLoss(torch.nn.Module):
    """ TO BE IMPLEMENTED """
    def __init__(self):
        pass