import glob
import json
from sklearn.model_selection import ShuffleSplit
from .custom_dataset import *


def read_all(data_path):
    # Read the whole dataset
    try:
        img1_name_list = json.load(
            open(data_path + "/img1_name_list.json", 'r'))
        img2_name_list = json.load(
            open(data_path + "/img2_name_list.json", 'r'))
        gt_name_list = json.load(open(data_path + "/gt_name_list.json", 'r'))
    except:
        data_dir = glob.glob(data_path + "/*")
        print(data_dir)
        gt_name_list = []
        img1_name_list = []
        img2_name_list = []

        for dir in data_dir:
            gt_name_list.extend(glob.glob(dir + '/*flow.flo'))
            img1_name_list.extend(glob.glob(dir + '/*img1.tif'))
            img2_name_list.extend(glob.glob(dir + '/*img2.tif'))
        gt_name_list.sort()
        img1_name_list.sort()
        img2_name_list.sort()
        assert (len(gt_name_list) == len(img1_name_list))
        assert (len(img2_name_list) == len(img1_name_list))

        # Serialize data into file:
        json.dump(img1_name_list, open(data_path + "/img1_name_list.json",
                                       'w'))
        json.dump(img2_name_list, open(data_path + "/img2_name_list.json",
                                       'w'))
        json.dump(gt_name_list, open(data_path + "/gt_name_list.json", 'w'))
    return img1_name_list, img2_name_list, gt_name_list


def read_by_type(data_path):
    # Read the data by flow type
    data_dir = glob.glob(data_path + "/*[!json]")
    flow_dir = [dir.split('/')[-1] for dir in data_dir]
    flow_img1_name_list = []
    flow_img2_name_list = []
    flow_gt_name_list = []

    try:
        for f_dir in flow_dir:
            flow_img1_name_list.append(
                json.load(
                    open(data_path + "/" + f_dir + "_img1_name_list.json",
                         'r')))
            flow_img2_name_list.append(
                json.load(
                    open(data_path + "/" + f_dir + "_img2_name_list.json",
                         'r')))
            flow_gt_name_list.append(
                json.load(
                    open(data_path + "/" + f_dir + "_gt_name_list.json", 'r')))

    except:
        for dir in data_dir:
            flow_gt_name_list.extend(glob.glob(dir + '/*flow.flo'))
            flow_img1_name_list.extend(glob.glob(dir + '/*img1.tif'))
            flow_img2_name_list.extend(glob.glob(dir + '/*img2.tif'))
            assert (len(flow_gt_name_list) == len(flow_img1_name_list))
            assert (len(flow_img2_name_list) == len(flow_img1_name_list))
            flow_gt_name_list.sort()
            flow_img1_name_list.sort()
            flow_img2_name_list.sort()
            # Serialize data into file:
            json.dump(
                flow_img1_name_list,
                open(
                    data_path + "/" + dir.split('/')[-1] +
                    "_img1_name_list.json", 'w'))
            json.dump(
                flow_img2_name_list,
                open(
                    data_path + "/" + dir.split('/')[-1] +
                    "_img2_name_list.json", 'w'))
            json.dump(
                flow_gt_name_list,
                open(
                    data_path + "/" + dir.split('/')[-1] +
                    "_gt_name_list.json", 'w'))
    return flow_img1_name_list, flow_img2_name_list, flow_gt_name_list, flow_dir


def construct_dataset(img1_name_list,
                      img2_name_list,
                      gt_name_list,
                      ratio=1.0,
                      test_size=0.1):
    """Construct dataset
    Args:
        img1_name_list: path list of the image1 in the pair
        img2_name_list: path list of the image2 in the pair
        gt_name_list: path list of the ground truth field
        ratio: Use how much of the data
        test_size: portion of test data (default 0.1)
    """

    amount = len(gt_name_list)
    total_data_index = np.arange(0, amount, 1)
    total_label_index = np.arange(0, amount, 1)

    # Divide train/validation and test data ( Default: 1:9)
    shuffler = ShuffleSplit(n_splits=1, test_size=test_size,
                            random_state=2).split(total_data_index,
                                                  total_label_index)
    indices = [(train_idx, test_idx) for train_idx, test_idx in shuffler][0]
    # Divide train and validation data ( Default: 1:9)
    shuffler_tv = ShuffleSplit(n_splits=1, test_size=test_size,
                               random_state=2).split(indices[0], indices[0])
    indices_tv = [(train_idx, validation_idx)
                  for train_idx, validation_idx in shuffler_tv][0]

    train_data = indices_tv[0][:int(ratio * len(indices_tv[0]))]
    validate_data = indices_tv[1][:int(ratio * len(indices_tv[1]))]
    test_data = indices[1][:int(ratio * len(indices[1]))]
    print("Check training data: ", len(train_data))
    print("Check validate data: ", len(validate_data))
    print("Check test data: ", len(test_data))

    train_dataset = FlowDataset(train_data, [img1_name_list, img2_name_list],
                                targets_index_list=train_data,
                                targets=gt_name_list)
    validate_dataset = FlowDataset(validate_data,
                                   [img1_name_list, img2_name_list],
                                   validate_data, gt_name_list)
    test_dataset = FlowDataset(test_data, [img1_name_list, img2_name_list],
                               test_data, gt_name_list)

    return train_dataset, validate_dataset, test_dataset
