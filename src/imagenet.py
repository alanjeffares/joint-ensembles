import torch
from datetime import datetime
import numpy as np
import time
import argparse
from tqdm import tqdm

from optim.losses import DecomposedLoss
from models.ensemble import Ensemble
from data.utils import get_dataloaders
from optim.optimiser import optimiser, scheduler, update_scheduler
from helpers.model_utils import evaluate
import helpers.utils as utils


EXPERIMENT = 'imagenet'

parser = argparse.ArgumentParser()
parser.add_argument('-tag', default='notag', help='Tag for this experiment')
args = parser.parse_args()

params, data_config, optim_config, path_config = utils.get_configs(EXPERIMENT)

num_classes = data_config[params['dataset']]['num_classes']
device = params['device']

# create results folder
now = datetime.now()
results_folder_path = utils.init_results_folder(params, args.tag, data_config, optim_config, path_config)

loss_fn = DecomposedLoss(num_learners=params['num_learners'], setting='classification',
                         probs=params['probs'])

for seed in params['seeds']:
    train_loader, val_loader = get_dataloaders(params['dataset'],
                                               batch_size=optim_config['batch_size'],
                                               seed=seed)

    for beta in params['beta_vals']:
        print(f'\n\nStarting experiment - Seed: {seed}, Beta: {beta}')

        model = Ensemble(params['base_learner'], params['num_learners'],  num_classes).to(device)
        optim = optimiser(model, optim_config)
        lr_scheduler = scheduler(optim, optim_config)

        train_loss_ls = []
        val_loss_ls = []
        val_acc_ind_ls = []
        val_acc_ens_ls = []
        val_div_ls = []
        val_ind_loss_ls = []
        val_ens_loss_ls = []
        val_top_5_acc_ls = []

        best_loss = np.inf
        best_ens_loss = np.inf
        breakflag = False
        patience = 0
        train_loss = -1

        for epoch in range(optim_config['num_epochs']):

            loop_start = time.time()
            
            running_loss = 0.0
            running_ens_loss = 0.0
            running_div_loss = 0.0
            running_w_ind_loss = 0.0
            
            iterator = tqdm(enumerate(train_loader))
            for i, (X, y) in iterator:
                model.train()
                X, y = X.to(device), y.to(device)

                optim.zero_grad()
                ens_pred, ind_pred, convex_weights = model(X)
                ens_pred = ens_pred.squeeze(1)
                convex_weights = convex_weights[:, :, 0]
                w_ind_loss, div, ens_loss, ind_loss = loss_fn(ens_pred, ind_pred, y, convex_weights)

                loss = w_ind_loss - beta * div
                loss.backward()
                optim.step()

                running_loss += loss.item() * X.shape[0]  # multiply mean loss by size of mini-batch
                running_ens_loss += ens_loss.item() * X.shape[0]
                running_div_loss += div.item() * X.shape[0]
                running_w_ind_loss += w_ind_loss.item() * X.shape[0]

                if (i + 1) % 2000 == 0:  # validate every 2000 mini-batches
                    print('\nStarting Validation')
                    val_results = evaluate(model, loss_fn, val_loader, beta, device)
                    [eval_loss, ens_loss, div_loss, w_ind_loss, accs_epoch, acc_ens, top_5_acc] = val_results

                    val_acc_ens_ls.append(acc_ens)
                    val_acc_ind_ls.append(accs_epoch)
                    val_div_ls.append(div_loss)
                    val_ind_loss_ls.append(w_ind_loss)
                    val_ens_loss_ls.append(ens_loss)
                    val_loss_ls.append(eval_loss)
                    val_top_5_acc_ls.append(top_5_acc)

                    print(f'epoch {epoch} [beta: {beta} seed: {seed}]')
                    print(f'train loss: {train_loss:.4f} val loss: {eval_loss:.4f} ' \
                          f'ens loss: {ens_loss:.4f} div: {div_loss:.4f} acc: {acc_ens:.4f} top 5 acc: {top_5_acc:.4f}')

                    val_results = [val_loss_ls, val_div_ls, val_ind_loss_ls, val_ens_loss_ls,
                        val_acc_ens_ls, val_acc_ind_ls, val_top_5_acc_ls]

                    utils.save_results_IN(results_folder_path, beta, seed, lr_scheduler, train_loss_ls, val_results)

                    # early stopping check
                    if ens_loss <= best_ens_loss:
                        best_ens_loss = ens_loss
                        patience = 1
                        print('New best validation loss!')
                        # save this model
                        best_model_time = now.strftime("%Y%m%d_%H:%M:%S")
                        best_model_path = f'{path_config["models"]}/model_{best_model_time}'
                        torch.save(model.state_dict(), best_model_path)
                    else:
                        patience += 1
                        if patience > optim_config['max_patience'] or lr_scheduler.get_last_lr()[0] < 1e-8:
                            print(f'No improvement for {optim_config["max_patience"]} epochs - applying early stopping')
                            breakflag = True
                            iterator.close()
                            break
                    
                    update_scheduler(lr_scheduler, optim_config['lr_switch'], patience, len(val_loss_ls))

            if breakflag:
                utils.store_final_model(results_folder_path, best_model_path, seed, beta)
                break

            train_loss = running_loss / len(train_loader.dataset)  # normalise by size of dataset
            ens_loss = running_ens_loss / len(train_loader.dataset)
            div_loss = running_div_loss / len(train_loader.dataset)
            w_ind_loss = running_w_ind_loss / len(train_loader.dataset)

            train_loss_ls.append(train_loss)

            loop_end = time.time()
            print(f'Epoch time: {(loop_end - loop_start):.4f} seconds')
            