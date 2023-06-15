import torch

def evaluate(model, loss_fn, data_loader, beta: float, device: str):
    """Evaluate model on a data_loader.
    
    Args:
        model: (nn.Module) model to evaluate
        loss_fn: (nn.Module) loss function
        data_loader: (DataLoader) data loader to evaluate on
        beta: (float) beta parameter for loss function
        device: (str) device to run on

    Returns:
        eval_loss: (float) evaluation loss
        ens_loss: (float) ensemble loss
        div_loss: (float) diversity loss
        w_ind_loss: (float) weighted individual loss
        accs_epoch: (list) list of individual learner accuracies
        acc_ens: (float) ensemble accuracy
        top_5_acc: (float) top 5 accuracy
    """    
    running_loss = 0.0
    running_ens_loss = 0.0
    running_div_loss = 0.0
    running_w_ind_loss = 0.0

    with torch.no_grad():
        ind_preds_full = []
        ens_pred_full = []
        targets_full = []
        model.eval()
        for i, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)

            ens_pred, ind_pred, convex_weights = model(X)
            ens_pred = ens_pred.squeeze(1)
            convex_weights = convex_weights[:,:,0]

            w_ind_loss, div, ens_loss, ind_loss = loss_fn(ens_pred, ind_pred, y, convex_weights)

            loss = w_ind_loss - beta * div

            ens_pred_full.append(ens_pred)
            ind_preds_full.append(ind_pred)
            targets_full.append(y)
            
            running_loss += loss.item() * X.shape[0]  # multiply mean loss by size of mini-batch
            running_ens_loss += ens_loss.item() * X.shape[0]
            running_div_loss += div.item() * X.shape[0]
            running_w_ind_loss += w_ind_loss.item() * X.shape[0]
    
    eval_loss = running_loss/data_loader.split_size  # normalise by size of dataset
    ens_loss = running_ens_loss/data_loader.split_size
    div_loss = running_div_loss/data_loader.split_size
    w_ind_loss = running_w_ind_loss/data_loader.split_size

    # unpack results
    ind_preds = torch.cat(ind_preds_full, dim=0).detach().cpu().numpy().argmax(2)
    ens_preds = torch.cat(ens_pred_full, dim=0).detach().cpu().numpy().argmax(1)
    test_labels = torch.cat(targets_full, dim=0).detach().cpu().numpy()
    ens_probs = torch.cat(ens_pred_full, dim=0).detach().cpu().numpy()

    # get ind learners acc
    accs_epoch = []
    ensemble_size = ind_preds.shape[1]
    for i in range(ensemble_size):
        acc_i = (ind_preds[:,i] == test_labels).sum()/data_loader.split_size
        accs_epoch.append(acc_i)

    # get ens acc
    acc_ens = (ens_preds == test_labels).sum()/data_loader.split_size 
    top_5_acc = top_n_acc(ens_probs, test_labels, 5)

    return eval_loss, ens_loss, div_loss, w_ind_loss, accs_epoch, acc_ens, top_5_acc


def top_n_acc(ens_probs, test_labels, n):
    """Get top n accuracy.
    
    Args:
        ens_probs: (np.array) ensemble probs of shape (num_samples, num_classes)
        test_labels: (np.array) test labels of shape (num_samples,)"""
    top_n = ens_probs.argsort(axis=1)[:,-n:]
    top_n_acc = (top_n == test_labels.reshape(-1,1)).sum()/test_labels.shape[0]
    return top_n_acc