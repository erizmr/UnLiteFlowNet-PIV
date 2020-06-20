import glob
import json

def read_all(data_path):
    # Read the whole dataset
    try:
      img1_name_list = json.load(open(data_path+"/img1_name_list.json",'r'))
      img2_name_list = json.load(open(data_path+"/img2_name_list.json",'r'))
      gt_name_list = json.load(open(data_path+"/gt_name_list.json",'r'))
    except:
      data_dir = glob.glob(data_path+"/*")
      print(data_dir)
      gt_name_list = []
      img1_name_list = []
      img2_name_list = []

      for dir in data_dir:
        gt_name_list.extend(glob.glob(dir+'/*flow.flo'))
        img1_name_list.extend(glob.glob(dir+'/*img1.tif'))
        img2_name_list.extend(glob.glob(dir+'/*img2.tif'))
      gt_name_list.sort()
      img1_name_list.sort()
      img2_name_list.sort()
      assert(len(gt_name_list) == len(img1_name_list))
      assert(len(img2_name_list) == len(img1_name_list))

      # Serialize data into file:
      json.dump(img1_name_list, open( data_path+"/img1_name_list.json", 'w' ))
      json.dump(img2_name_list, open( data_path+"/img2_name_list.json", 'w' ))
      json.dump(gt_name_list, open( data_path+"/gt_name_list.json", 'w' ))
    return img1_name_list, img2_name_list, gt_name_list


def read_by_type(data_path):
    # Read the data by flow type
    data_dir = glob.glob(data_path+"/*[!json]")
    flow_dir = [dir.split('/')[-1] for dir in data_dir]
    try:
      flow_img1_name_list = []
      flow_img2_name_list = []
      flow_gt_name_list = []

      for f_dir in flow_dir:
        flow_img1_name_list.append(json.load(open(data_path+"/" + f_dir + "_img1_name_list.json",'r')))
        flow_img2_name_list.append(json.load(open(data_path+"/" + f_dir + "_img2_name_list.json",'r')))
        flow_gt_name_list.append(json.load(open(data_path+"/" + f_dir + "_gt_name_list.json",'r')))

    except:
      for dir in data_dir:
        flow_gt_name_list = []
        flow_img1_name_list = []
        flow_img2_name_list = []
        flow_gt_name_list.extend(glob.glob(dir+'/*flow.flo'))
        flow_img1_name_list.extend(glob.glob(dir+'/*img1.tif'))
        flow_img2_name_list.extend(glob.glob(dir+'/*img2.tif'))
        assert(len(flow_gt_name_list) == len(flow_img1_name_list))
        assert(len(flow_img2_name_list) == len(flow_img1_name_list))
        flow_gt_name_list.sort()
        flow_img1_name_list.sort()
        flow_img2_name_list.sort()
        # Serialize data into file:
        json.dump(flow_img1_name_list, open( data_path+"/" +  dir.split('/')[-1] + "_img1_name_list.json", 'w' ))
        json.dump(flow_img2_name_list, open( data_path+"/" +  dir.split('/')[-1] + "_img2_name_list.json", 'w' ))
        json.dump(flow_gt_name_list, open( data_path+"/"  +  dir.split('/')[-1] + "_gt_name_list.json", 'w' ))
    return flow_img1_name_list, flow_img2_name_list, flow_gt_name_list, flow_dir