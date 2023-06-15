import torch 
import torch.nn as nn
import torch.nn.functional as F

from models.base_models import ResNet, VGG, CNN, IN_base_learner


def get_base_learner(name: str, num_classes: int):
    if name in ['ResNet18', 'ResNet34']:
        return ResNet(name, num_classes)
    elif name in ['VGG']:
        return VGG(num_classes)
    elif name in ['CNN']:
        return CNN(num_classes)
    elif name.endswith('IN'):
        return IN_base_learner(name) 
    else:
        raise ValueError(f'Unknown base learner: {name}')


class Ensemble(nn.Module):
    """Ensemble of base learners
    
    Args:
        base_learner: (str) name of base learner
        num_learners: (int) number of learners in ensemble
        num_classes: (int) number of classes in dataset
        probs: (bool) whether to aggregate at output probability level
        drop_p: (float) dropout probability (learrner level dropout)
        device: (str) device to run on
    """
    def __init__(self, base_learner, num_learners, num_classes, probs=True, drop_p=0, device='cuda'):
        super(Ensemble, self).__init__()
        self.num_learners = num_learners
        self.probs = probs
        self.device = device
        self.binomial = torch.distributions.binomial.Binomial(probs=1 - drop_p)

        self.ensemble_learners = nn.ModuleList()
        for _ in range(self.num_learners):
            learner_network = get_base_learner(base_learner, num_classes)
            self.ensemble_learners.append(learner_network)

    def forward(self, x):
        ind_pred = []
        for i, learner in enumerate(self.ensemble_learners):
            pred_ens_i = self.ensemble_learners[i](x)

            if self.probs:
                # check dims
                pred_ens_i = F.softmax(pred_ens_i, dim=1)

            ind_pred.append(pred_ens_i)
        ind_pred = torch.stack(ind_pred, dim=1).squeeze()

        # for ensembles with only one learner
        if len(ind_pred.shape) < 2:
            ind_pred = ind_pred.unsqueeze(1)
        if self.training:
            binary_mask = self.binomial.sample(ind_pred.shape[:2]).unsqueeze(2).repeat(1, 1, ind_pred.shape[-1]).to(
                self.device)
        else:
            binary_mask = torch.ones(ind_pred.shape).to(self.device)

        convex_weights = binary_mask / binary_mask.sum(dim=1, keepdim=True)
        convex_weights = torch.nan_to_num(convex_weights)  # deal with ensembles with all zero
        ensemble_pred = (ind_pred * convex_weights).sum(dim=1, keepdim=True)

        return ensemble_pred, ind_pred, convex_weights