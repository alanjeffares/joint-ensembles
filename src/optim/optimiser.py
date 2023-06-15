import torch

def optimiser(model, config):
    """Construct the appropriate optimiser based on string input from config"""
    if config['optimiser'] == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=config['learning_rate'],
                               momentum=config['momentum'], weight_decay=config['weight_decay'])
    elif config['optimiser'] == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                weight_decay=config['weight_decay'])
    elif config['optimiser'] == 'RMSprop':
        return torch.optim.RMSprop(model.parameters(), lr=config['learning_rate'],
                                   momentum=config['momentum'], weight_decay=config['weight_decay'])
    else:
        raise ValueError('Optimiser not recognised')
    
def scheduler(optimiser, config):
    """Construct the appropriate scheduler based on string input from config"""
    if config['scheduler'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=config['gamma'])
    elif config['scheduler'] == 'Linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimiser, start_factor=1, end_factor=0.0, total_iters=1/config['factor'])
    elif config['scheduler'] == None:
        # dummy scheduler that does nothing
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=1)
    else:
        raise ValueError('scheduler not recognised')
    scheduler.type = config['scheduler']
    scheduler.switch_iter = []
    return scheduler

def update_scheduler(scheduler, lr_switch, count, curr_iter):
    """Update scheduler. lr_switch can be an int or a list of ints. If it is a list, 
    the scheduler will be updated at each of the epochs in the list if it is an int,
    the scheduler will be updated every lr_switch epochs."""
    if scheduler.type == 'ExponentialLR':
        if isinstance(lr_switch, int):  # update schedular after count iters of no improvement
            if count % lr_switch == 0:
                scheduler.step()
                scheduler.switch_iter.append(curr_iter)
                print(f'Updating scheduler, new lr: {scheduler.get_last_lr()[0]:.8f}')
        elif isinstance(lr_switch, list):  # update scheduler at each iter in lr_switch
            if curr_iter in lr_switch:
                scheduler.step()
                scheduler.switch_iter.append(curr_iter)
                print(f'Updating scheduler, new lr: {scheduler.get_last_lr()[0]:.8f}')
    elif scheduler.type == 'Linear':  # subract factor from lr every lr_switch iters
        if curr_iter % lr_switch == 0:
            scheduler.step()
            scheduler.switch_iter.append(curr_iter)
            print(f'Updating scheduler, new lr: {scheduler.get_last_lr()[0]:.8f}')