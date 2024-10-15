import cv2
import argparse
import os
import numpy as np

import rosbag
from cv_bridge import CvBridge

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    args = parser.parse_args()

    bag = rosbag.Bag(args.bag_path)
    bridge = CvBridge()

    os.makedirs('demo_data/' + args.dataset_name, exist_ok=True)
    for fd in ['depth', 'rgb', 'masks']:
        os.makedirs('demo_data/' + args.dataset_name + '/' + fd, exist_ok=True)

    depth_dir = 'demo_data/' + args.dataset_name + '/depth/'
    rgb_dir = 'demo_data/' + args.dataset_name + '/rgb/'

    count = 0
    for topic, msg, t in bag.read_messages(topics=['/device_0/sensor_0/Depth_0/image/data']):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        cv2.imwrite(os.path.join(depth_dir, "%04i.png" % count), cv_img)
        print("Wrote depth image %i" % count)

        count += 1

    count = 0
    for topic, msg, t in bag.read_messages(topics=['/device_0/sensor_1/Color_0/image/data']):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        cv2.imwrite(os.path.join(rgb_dir, "%04i.png" % count), cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR))
        print("Wrote rgb image %i" % count)

        count += 1

    # topics = bag.get_type_and_topic_info().topics

    # for t,v in topics.items():
    #     print(t,v[1])

    for topic, msg, t in bag.read_messages(topics=['/device_0/sensor_1/Color_0/info/camera_info']):
        print('RGB intrinsics: ', msg)

        K = np.array(msg.K)

        with open('demo_data/' + args.dataset_name + "/cam_K.txt", "w") as f:
            for i in range(3):
                f.write(f'{K[i]} {K[i+1]} {K[i+2]}\n')

    for topic, msg, t in bag.read_messages(topics=['/device_0/sensor_0/Depth_0/info/camera_info']):
        print('Depth intrinsics: ', msg)

        


    bag.close()