# -*- coding: utf-8 -*-
"""
UnLiteFlowNet-PIV

"""
import argparse
import matplotlib.pyplot as plt
from src.model.models import *
from src.data_processing.read_data import *
from src.train.train_functions import *

data_path = "./sample_data"
result_path = "./output"


def test_train():
    # Read data
    img1_name_list, img2_name_list, gt_name_list = read_all(data_path)
    flow_img1_name_list, flow_img2_name_list, flow_gt_name_list, flow_dir = read_by_type(
        data_path)

    print([f_dir for f_dir in flow_dir])
    img1_len = [len(f_dir) for f_dir in flow_img1_name_list]
    img2_len = [len(f_dir) for f_dir in flow_img2_name_list]
    gt_len = [len(f_dir) for f_dir in flow_gt_name_list]

    for img1_num, img2_num in zip(img1_len, img2_len):
        assert img1_num == img2_num
    for img1_num, gt_num in zip(img1_len, gt_len):
        assert img1_num == gt_num

    train_dataset, validate_dataset, test_dataset = construct_dataset(
        img1_name_list, img2_name_list, gt_name_list)

    # Set hyperparameters
    lr = 1e-4
    batch_size = 8
    test_batch_size = 8
    n_epochs = 100
    new_train = True

    # Load the network model
    model = Network().to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=1e-5,
                                 eps=1e-3,
                                 amsgrad=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    if new_train:
        # New train
        model_trained = train_model(model, train_dataset, validate_dataset,
                                    test_dataset, batch_size, test_batch_size,
                                    lr, n_epochs, optimizer)
    else:
        model_save_name = 'UnsupervisedLiteFlowNet_pretrained.pt'
        PATH = F"./models/{model_save_name}"
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        model_trained = train_model(model,
                                    train_dataset,
                                    validate_dataset,
                                    test_dataset,
                                    batch_size,
                                    test_batch_size,
                                    lr,
                                    n_epochs,
                                    optimizer,
                                    epoch_trained=epoch + 1)
    return model_trained


def test_estimate():

    flow_img1_name_list, flow_img2_name_list, flow_gt_name_list, flow_dir = read_by_type(
        data_path)
    assert len(flow_dir) == len(flow_img1_name_list)
    flow_dataset = {}

    for i, f_name in enumerate(flow_dir):
        total_index = np.arange(0, len(flow_img1_name_list[i]), 1)
        flow_dataset[f_name] = FlowDataset(
            total_index, [flow_img1_name_list[i], flow_img2_name_list[i]],
            targets_index_list=total_index,
            targets=flow_gt_name_list[i])

    flow_type = [f_dir for f_dir in flow_dir]
    print("Flow cases: ", flow_type)

    # Load pretrained model
    model_save_name = 'UnsupervisedLiteFlowNet_pretrained.pt'
    PATH = F"./models/{model_save_name}"
    unliteflownet = Network()
    unliteflownet.load_state_dict(torch.load(PATH)['model_state_dict'])
    unliteflownet.eval()
    unliteflownet.to(device)
    print('unliteflownet load successfully.')

    # Visualize results, random select a flow type
    f_type = random.randint(0, len(flow_type) - 1)
    print("Selected flow scenario: ", flow_type[f_type])
    test_dataset = flow_dataset[flow_type[f_type]]
    test_dataset.eval()

    resize = False
    save_to_disk = False

    # random select a sample
    number_total = len(test_dataset)
    number = random.randint(0, number_total - 1)
    input_data, label_data = test_dataset[number]
    h_origin, w_origin = input_data.shape[-2], input_data.shape[-1]

    if resize:
        input_data = F.interpolate(input_data.view(-1, 2, h_origin, w_origin),
                                   (256, 256),
                                   mode='bilinear',
                                   align_corners=False)
    else:
        input_data = input_data.view(-1, 2, 256, 256)

    h, w = input_data.shape[-2], input_data.shape[-1]
    x1 = input_data[:, 0, ...].view(-1, 1, h, w)
    x2 = input_data[:, 1, ...].view(-1, 1, h, w)

    # Visualization
    fig, axarr = plt.subplots(1, 2, figsize=(16, 8))

    # ------------Unliteflownet estimation-----------
    b, _, h, w = input_data.size()
    y_pre = estimate(x1.to(device), x2.to(device), unliteflownet, train=False)
    y_pre = F.interpolate(y_pre, (h, w), mode='bilinear', align_corners=False)

    resize_ratio_u = h_origin / h
    resize_ratio_v = w_origin / w
    u = y_pre[0][0].detach() * resize_ratio_u
    v = y_pre[0][1].detach() * resize_ratio_v

    color_data_pre = np.concatenate((u.view(h, w, 1), v.view(h, w, 1)), 2)
    u = u.numpy()
    v = v.numpy()
    # Draw velocity magnitude
    axarr[1].imshow(fz.convert_from_flow(color_data_pre))
    # Control arrow density
    X = np.arange(0, h, 8)
    Y = np.arange(0, w, 8)
    xx, yy = np.meshgrid(X, Y)
    U = u[xx.T, yy.T]
    V = v[xx.T, yy.T]
    # Draw velocity direction
    axarr[1].quiver(yy.T, xx.T, U, -V)
    axarr[1].axis('off')
    color_data_pre_unliteflownet = color_data_pre

    # ---------------Label data------------------
    u = label_data[0].detach()
    v = label_data[1].detach()

    color_data_label = np.concatenate((u.view(h, w, 1), v.view(h, w, 1)), 2)
    u = u.numpy()
    v = v.numpy()
    # Draw velocity magnitude
    axarr[0].imshow(fz.convert_from_flow(color_data_label))
    # Control arrow density
    X = np.arange(0, h, 8)
    Y = np.arange(0, w, 8)
    xx, yy = np.meshgrid(X, Y)
    U = u[xx.T, yy.T]
    V = v[xx.T, yy.T]

    # Draw velocity direction
    axarr[0].quiver(yy.T, xx.T, U, -V)
    axarr[0].axis('off')
    color_data_pre_label = color_data_pre

    if save_to_disk:
        fig.savefig('./output/frame_%d.png' % number, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test')
    parser.add_argument('--train', action='store_true', help='train the model')
    parser.add_argument('--test', action='store_true', help='train the model')

    args = parser.parse_args()
    isTrain = args.train
    isTest = args.test

    if isTrain:
        test_train()
    if isTest:
        test_estimate()
