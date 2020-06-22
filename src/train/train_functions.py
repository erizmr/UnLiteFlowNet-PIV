import random
import numpy as np
import json
from src.model.loss_functions import *
from src.model.utils import realEPE
from torch.utils.data import DataLoader
from livelossplot import PlotLosses
import GPUtil
import time
import datetime
from src.model.models import estimate, device


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True  # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms
    torch.backends.cudnn.enabled = True

    return True


def train(model, optimizer, criterion, data_loader):
    """Train function """
    model.train()
    train_loss, flow2_EPE = 0, 0
    iter_num = 0
    total_time = 0
    for x, y in data_loader:
        start = time.clock()
        x1 = x[:, 0, ...].view(-1, 1, 256, 256)
        x2 = x[:, 1, ...].view(-1, 1, 256, 256)
        y = y.view(-1, 2, 256, 256)

        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()

        output_forward = estimate(x1, x2, model, train=True)
        output_backward = estimate(x2, x1, model, train=True)
        loss = criterion(output_forward, output_backward, x1, x2)
        real_timeEPE = realEPE(output_forward[-1], y).item()
        flow2_EPE += real_timeEPE * x.size(0)

        loss.backward()
        train_loss += loss.item() * x.size(0)
        optimizer.step()

        ##-----------------print info-----------------------
        end = time.clock()
        time_used = end - start
        total_time += time_used
        iter_num += 1
        percent = 100 * iter_num * x1.shape[0] / len(data_loader.dataset)
        print(
            "Finished this epoch %1.3f %%, real time EPE loss %1.3f, time used(seconds) %1.3f, expected time to finish %1.3f"
            % (percent, real_timeEPE, total_time,
               (100 - percent) * total_time / percent))

    return train_loss / len(data_loader.dataset), flow2_EPE / len(
        data_loader.dataset)


def validate(model, criterion, data_loader):
    """Validate functions"""
    model.eval()
    validation_loss = 0.
    for x, y in data_loader:
        with torch.no_grad():
            x1 = x[:, 0, ...].view(-1, 1, 256, 256)
            x2 = x[:, 1, ...].view(-1, 1, 256, 256)
            y = y.view(-1, 2, 256, 256)

            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            output_forward = estimate(x1, x2, model, train=True)
            loss = criterion(output_forward[-1], y)
            validation_loss += loss.item() * x.size(0)

    return validation_loss / len(data_loader.dataset)


def train_model(model,
                train_dataset,
                validate_dataset,
                test_dataset,
                batch_size,
                test_batch_size,
                lr,
                n_epochs,
                optimizer=None,
                epoch_trained=0,
                seed=42):
    """The train function """
    set_seed(seed)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=1e-5,
                                     eps=1e-3,
                                     amsgrad=True)
    else:
        optimizer = optimizer
    criterion_train = multiscaleUnsupervisorError
    criterion_validate = realEPE

    # Prepare data loader
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    validation_loader = DataLoader(validate_dataset,
                                   batch_size=test_batch_size,
                                   shuffle=False,
                                   num_workers=4,
                                   pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=test_batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)

    liveloss = PlotLosses()
    para_dict = {}
    total_time = 0
    for epoch in range(epoch_trained, n_epochs):
        start_time = time.clock()
        print("Total epoch %d" % n_epochs)
        print("Epoch %d starts! " % epoch)
        print("Memory allocated: ",
              torch.cuda.memory_allocated() / 1024 / 1024 / 1024)

        GPUtil.showUtilization()

        logs = {}
        train_loss, train_loss_epe = train(model, optimizer, criterion_train,
                                           train_loader)
        validation_loss_epe = validate(model, criterion_validate,
                                       validation_loader)

        end_time = time.clock()

        logs['' + 'multiscale loss'] = train_loss
        logs['' + 'EPE loss'] = train_loss_epe
        logs['val_' + 'EPE loss'] = validation_loss_epe
        liveloss.update(logs)
        liveloss.draw()

        total_time += end_time - start_time

        print(
            "Epoch: ", epoch, ", Avg. Train EPE Loss: %1.3f" % train_loss_epe,
            " Avg. Validation EPE Loss: %1.3f" % validation_loss_epe,
            "Time used this epoch (seconds): %1.3f" % (end_time - start_time),
            "Time remain(hrs) %1.3f" % (total_time / (epoch + 1) *
                                        (n_epochs - epoch) / 3600))

        # Every 5 epoach, checkpoint
        if (epoch + 1) % 5 == 0:
            test_loss_epe = validate(model, criterion_validate, test_loader)
            # Fill in the parameters into the dict
            para_dict['epoch'] = epoch
            para_dict['dataset size'] = len(train_loader.dataset)
            para_dict['train EPE'] = train_loss_epe
            para_dict['validation EPE'] = validation_loss_epe
            para_dict['learning rate'] = lr
            para_dict['time used(seconds)'] = total_time

            # There is no actual test loss, so use validation loss here
            para_dict['test EPE'] = test_loss_epe

            # Do the save
            save_model(model, optimizer, train_loss, para_dict,
                       "UnLiteFlowNet_checkpoint_%d_" % epoch)

    test_loss_epe = validate(model, criterion_validate, test_loader)
    print(" Avg. Test EPE Loss: %1.3f" % test_loss_epe,
          "Total time used(seconds): %1.3f" % total_time)
    print("")

    # Fill in the parameters into the dict
    para_dict = {}
    para_dict['epoch'] = n_epochs
    para_dict['dataset size'] = len(train_loader.dataset)
    para_dict['batch_size'] = batch_size
    para_dict['train EPE'] = train_loss_epe
    para_dict['validation EPE'] = validation_loss_epe
    para_dict['learning rate'] = lr
    para_dict['time used(seconds)'] = total_time
    para_dict['test EPE'] = test_loss_epe
    save_model(model, optimizer, train_loss, para_dict,
               "UnLiteFlowNet_%d_" % epoch)

    return model


def save_model(model, optimizer, train_loss, para_dict, save_name):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
    model_save_name = save_name + st + '.pt'
    PATH = F"./{model_save_name}"
    epoch = para_dict['epoch']
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, PATH)

    # Serialize data into file:
    json.dump(para_dict, open("./" + save_name + st + '.json', 'w'))
