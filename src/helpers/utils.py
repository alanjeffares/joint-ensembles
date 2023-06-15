import json
from datetime import datetime
import os
import csv
from pathlib import Path

COL_NAMES = ['beta', 'run', 'train_loss', 'val_loss', 'val_div', 'val_ind_loss', 'val_ens_loss',
             'val_acc_ens', 'val_acc_ind', 'test_loss', 'test_div', 'test_ind_loss', 'test_ens_loss',
             'test_acc_ens', 'test_acc_ind']

def get_configs(experiment):
    """Load configs and test they are valid"""
    # first load the implementation check config
    with open('src/tests/implemented.json') as json_file:
        implemented = json.load(json_file)

    # get experiment config
    with open(f'src/configs/{experiment}/experiment.json') as json_file:
        params = json.load(json_file)

    test_experiment_config(params, implemented)

    # get data config
    with open('src/configs/data.json') as json_file:
        data_config = json.load(json_file)

    # get optim config
    with open(f'src/configs/{experiment}/optim.json') as json_file:
        optim_config = json.load(json_file)
    
    test_optim_config(optim_config, implemented)

    # get paths config
    with open('src/configs/paths.json') as json_file:
        path_config = json.load(json_file)
    
    test_paths_config(path_config)

    return params, data_config, optim_config, path_config


def test_experiment_config(params, implemented):
    """Test experiment config is valid"""
    assert params['dataset'] in implemented['datasets'], 'Dataset not supported'
    assert params['base_learner'] in implemented['base_learners'], 'Base learner not supported'
    assert params['probs'] in implemented['probs'], 'Probs not supported'

def test_optim_config(optim_config, implemented):
    """Test optim config is valid"""
    assert optim_config['optimiser'] in implemented['optimisers'], 'Optimiser not supported'
    assert optim_config['scheduler'] in implemented['schedulers'], 'Scheduler not supported'

def test_paths_config(path_config):
    """Test all paths exist"""
    assert os.path.isdir(path_config['data']), 'Data path does not exist'
    assert os.path.isdir(path_config['results']), 'Results path does not exist'
    if not os.path.isdir(path_config['models']):
        print('Warning: models path does not exist, creating directory...')
        init_models_folder(path_config['models'])


def init_results_folder(params, tag, data_config, optim_config, path_config):
    """Create results folder, dump configs, and return path"""
    # create results folder
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H:%M:%S")
    results_folder_path = f'{path_config["results"]}/{tag}_{params["dataset"]}_{params["base_learner"]}_{dt_string}'
    os.mkdir(results_folder_path)

    # store experiment configurations
    with open(f'{results_folder_path}/experiment.json', 'w') as outfile:
        json.dump(params, outfile)

    with open(f'{results_folder_path}/data.json', 'w') as outfile:
        json.dump(data_config, outfile)
    
    with open(f'{results_folder_path}/optim.json', 'w') as outfile:
        json.dump(optim_config, outfile)

    with open(f'{results_folder_path}/paths.json', 'w') as outfile:
        json.dump(path_config, outfile)

    # init the results file
    init_results_file(params['dataset'], results_folder_path)
    
    return results_folder_path


def init_results_file(dataset, results_folder_path):
    """instatiate the results csv file"""
    if dataset == 'ImageNet':
        pass
    else:
        with open(f'{results_folder_path}/results.csv', 'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(COL_NAMES)

def save_results(results_folder_path, beta, seed, train_loss_ls, val_results, test_results):
    """Save results for beta-diversity experiments"""
    val_loss_ls, val_div_ls, val_ind_loss_ls, val_ens_loss_ls, val_acc_ens_ls, val_acc_ind_ls = val_results
    eval_loss, ens_loss, div_loss, w_ind_loss, accs_epoch, acc_ens, _ = test_results
    
    results = {
                'beta': beta,
                'run': seed,
                'train_loss': train_loss_ls,
                'val_loss': val_loss_ls,
                'val_div': val_div_ls,
                'val_ind_loss': val_ind_loss_ls,
                'val_ens_loss': val_ens_loss_ls,
                'val_acc_ens': val_acc_ens_ls,
                'val_acc_ind': val_acc_ind_ls,
                'test_loss': eval_loss,
                'test_div': div_loss,
                'test_ind_loss': w_ind_loss,
                'test_ens_loss': ens_loss,
                'test_acc_ens': acc_ens,
                'test_acc_ind': accs_epoch,
            }
    with open(f'{results_folder_path}/results.csv', 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=COL_NAMES)
        writer.writerow(results)

def save_results_IN(results_folder_path, beta, seed, lr_scheduler, train_loss_ls, val_results):
    """Save live results for IN experiments"""
    val_loss_ls, val_div_ls, val_ind_loss_ls, val_ens_loss_ls, val_acc_ens_ls, val_acc_ind_ls, top_5_acc = val_results
    
    results = {
                'beta': beta,
                'run': seed,
                'lr_switch': lr_scheduler.switch_iter,
                'train_loss': train_loss_ls,
                'val_loss': val_loss_ls,
                'val_div': val_div_ls,
                'val_ind_loss': val_ind_loss_ls,
                'val_ens_loss': val_ens_loss_ls,
                'val_acc_ens': val_acc_ens_ls,
                'val_acc_ind': val_acc_ind_ls,
                'val_top_5_acc': top_5_acc
            }
    with open(f'{results_folder_path}/results_seed:{seed}_beta:{beta}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def init_models_folder(path):
    """Create models folder"""
    Path(path).mkdir(parents=True, exist_ok=True)

def store_final_model(results_folder_path, best_model_path, seed, beta):
    """Store final model in results folder"""
    final_model_path = f'{results_folder_path}/model_seed:{seed}_beta:{beta}'
    os.rename(best_model_path, final_model_path)