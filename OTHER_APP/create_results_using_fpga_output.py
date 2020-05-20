import numpy as np

import sys
import os
import json
from tqdm import tqdm

json_dir = './' #output directory
path_to_test_image_dir = './dump_detect/'
if (len(sys.argv) > 1) :
    path_to_test_image_dir = sys.argv[1]
    print(path_to_test_image_dir)

# Define the label
label_dict = {}
label_dict[0] = 'Pedestrian'
label_dict[1] = 'Bicycle'
label_dict[2] = 'Car'
label_dict[3] = 'Truck'
label_dict[4] = 'Signal'
label_dict[5] = 'Signs'

def detect_json():
    print('object detection')
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    if os.path.exists(json_dir+'results.json'):
        os.remove(json_dir+'results.json')
    res_file = open(json_dir+'results.json', 'w')
    results = {}
    for file_name in tqdm(sorted(os.listdir(path_to_test_image_dir))):
        detect = np.reshape(np.fromfile(path_to_test_image_dir+file_name, np.float32), [1, 1, -1, 6])

        # Parse the outputs.
        det_label = detect[0,0,:,0]
        det_conf = detect[0,0,:,1]
        det_xmin = detect[0,0,:,2]
        det_ymin = detect[0,0,:,3]
        det_xmax = detect[0,0,:,4]
        det_ymax = detect[0,0,:,5]

        # Define buffer of results
        num_boxes = len(det_label)
        tmp_res = {}
        res_per_ped = []
        res_per_bic = []
        res_per_car = []
        res_per_truck = []
        res_per_signal = []
        res_per_signs = []
        res_per_image = {}
        # Post Processing for object detection & Save results
        for jj in range(num_boxes):
            label = label_dict[int(det_label[jj])]
            post_det_xmin = det_xmin[jj]
            post_det_ymin = det_ymin[jj]
            post_det_xmax = det_xmax[jj]
            post_det_ymax = det_ymax[jj]

            if label == 'Pedestrian':
                    res_per_ped.append([int(post_det_xmin),int(post_det_ymin),int(post_det_xmax),int(post_det_ymax)])
            elif label == 'Bicycle':
                    res_per_bic.append([int(post_det_xmin),int(post_det_ymin),int(post_det_xmax),int(post_det_ymax)])
            elif label == 'Car':
                    res_per_car.append([int(post_det_xmin),int(post_det_ymin),int(post_det_xmax),int(post_det_ymax)])
            elif label == 'Truck':
                    res_per_truck.append([int(post_det_xmin),int(post_det_ymin),int(post_det_xmax),int(post_det_ymax)])
            elif label == 'Signal':
                    res_per_signal.append([int(post_det_xmin),int(post_det_ymin),int(post_det_xmax),int(post_det_ymax)])
            else:
                    res_per_signs.append([int(post_det_xmin),int(post_det_ymin),int(post_det_xmax),int(post_det_ymax)])

        tmp_res["Pedestrian"] = res_per_ped
        tmp_res["Bicycle"] = res_per_bic
        tmp_res["Car"] = res_per_car
        tmp_res["Truck"] = res_per_truck
        tmp_res["Signal"] = res_per_signal
        tmp_res["Signs"] = res_per_signs
        # Merge results
        results[file_name.split('.')[0]+'.jpg'] = tmp_res
    
    # Write results(.json) 
    sorted_results = sorted(results.items(), key=lambda x:x[0])
    results.clear()
    results.update(sorted_results)
    json.dump(results,res_file,indent=1)
    res_file.close

detect_json()

