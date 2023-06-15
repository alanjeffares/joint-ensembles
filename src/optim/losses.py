import torch
import torch.nn as nn

class CustomCE(nn.Module):
    """
    Expects inputs of shapes:
    pred - (batch_size, num_outputs)
    label - (num_outputs)

    Note that in the case of multiple learners in an ensemble we can pass
    their predictions with pred of shape (batch_size * ensemble_size, num_outputs)
    and corresponding labels of shape (batch_size * ensemble_size).

    Args:
        probs (bool): apply cross entropy if applying to logits and NLL if acting
            applying to probabilities.
    """

    def __init__(self, probs=None, weights=None) -> None:
        super(CustomCE, self).__init__()
        self.probs = probs
        if self.probs:
            self.CE = nn.NLLLoss(weight=weights, reduction='none')
        else:
            self.CE = nn.CrossEntropyLoss(weight=weights, reduction='none')

    def forward(self, pred, label):
        if self.probs:
            eps = 1e-12  # prevents log(0) error
            return self.CE(torch.log(pred + eps), label)
        else:
            return self.CE(pred, label)


class CustomMSE(nn.Module):
    def __init__(self) -> None:
        super(CustomMSE, self).__init__()
        self.MSE = nn.MSELoss(reduction='none')

    def forward(self, pred, label):
        if len(label.shape) != 2:
            label = label.unsqueeze(1)
        return self.MSE(pred, label)

class DecomposedLoss(nn.Module):
    """ Decomposed loss function for ensembles of learners
    Args:
        num_learners (int): number of learners in ensemble
        setting (str): 'classification' or 'regression'
        probs (bool): ensemble the scores or the probabilities
        weights (tensor): weights for learners in the ensemble. If None,
            uniform weighting is applied.
    """
    def __init__(self, num_learners, setting, probs, weights=None):
        super(DecomposedLoss, self).__init__()
        self.num_learners = num_learners
        self.setting = setting  # 'classification' or 'regression'
        self.probs = probs  # ensemble the scores or the probabilities
        if self.setting == 'classification':
            self.loss_fn = CustomCE(probs=self.probs, weights=weights)
        else:
            self.loss_fn = CustomMSE()

    def weighted_ensemble_error(self, ind_pred, label, convex_weights):
        num_outputs = ind_pred.shape[2]
        ensemble_size = ind_pred.shape[1]
        batch_size = ind_pred.shape[0]
        effective_batch_size = batch_size - (convex_weights.sum(1) == 0).sum()  # dont count cases where full ensemble is downweighted

        # reshape dimensions from (batch_size, ensemble_size, *) to
        # (batch_size * ensemble_size, *) for predictions and expand dimensions
        # of labels correspondingly.
        label = label.repeat_interleave(ensemble_size)
        ind_preds = ind_pred.reshape(-1, num_outputs)
        ind_loss = self.loss_fn(ind_preds, label)  # individual learner loss
        ind_loss = ind_loss.reshape(-1, ensemble_size)  # reshape loss back to original shape

        # apply convex weights
        ind_loss = ind_loss * (convex_weights > 0)  # only get loss of learners from binary mask
        w_ind_loss = (ind_loss * convex_weights).sum(dim=1, keepdim=True)  # weighted individual learner loss
        w_ind_loss = w_ind_loss.sum()/effective_batch_size  # average over minibatch

        ind_loss = ind_loss.sum(dim=0) / (convex_weights > 0).sum(dim=0, keepdim=True)  # possible division by zero
        ind_loss[ind_loss != ind_loss] = 0  # replace nan values (due to division by 0) by 0
        ind_loss = ind_loss.squeeze()

        return w_ind_loss, ind_loss

    def forward(self, ens_pred, ind_pred, label, convex_weights):
        """
        Input dimensions:
        ens_pred - (batch_size, num_outputs)
        ind_pred - (batch_size, ensemble_size, num_outputs)
        label - (batch_size)
        convex_weights - (batch_size, ensemble_size)

        """
        # calculate ensemble loss
        ens_losses = self.loss_fn(ens_pred, label)#.mean()
        effective_batch_size = ind_pred.shape[0] - (convex_weights.sum(1) == 0).sum()
        effective_ens_losses = ens_losses[convex_weights.sum(1) > 0]
        ens_loss = effective_ens_losses.sum()/effective_batch_size

        # calculate individual loss
        w_ind_loss, ind_loss = self.weighted_ensemble_error(ind_pred, label, convex_weights)

        div = w_ind_loss - ens_loss  # diversity

        return w_ind_loss, div, ens_loss, ind_loss