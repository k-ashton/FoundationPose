import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse
import os

from mobile_sam import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import glob

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
# function to display the coordinates of 
# of the points clicked on the image 
def click_event(event, x, y, flags, params): 

    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 

        # displaying the coordinates 
        # on the Shell 
        print(x, ' ', y) 
        image_copy = image.copy()

        # displaying the coordinates 
        # on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(image_copy, str(x) + ',' +
                    str(y), (x,y), font, 
                    1, (255, 0, 0), 2) 

        input_point = np.array([[x, y]])
        input_label = np.array([1])

        predictor = SamPredictor(mobile_sam)
        predictor.set_image(image)
        # plt.figure(figsize=(10,10))
        # plt.imshow(image)
        # show_points(input_point, input_label, plt.gca())
        # plt.axis('on')
        # plt.show()  
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        mask_idx = args.mask_idx
        mask_img = np.zeros(masks[mask_idx].shape)
        mask_img[masks[mask_idx]] = 255.0
        image_copy[masks[mask_idx]] = np.array([30, 144, 255])
        cv2.imshow('image', cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)) 
        

        # for i, (mask, score) in enumerate(zip(masks, scores)):
        #     plt.figure(figsize=(10,10))
        #     plt.imshow(image)
        #     show_mask(mask, plt.gca())
        #     show_points(input_point, input_label, plt.gca())
        #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        #     plt.axis('off')
        #     plt.show()  

        cv2.imwrite(fn.replace('rgb','masks'), mask_img)





if __name__=='__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--folder', type=str, default=f'{code_dir}/demo_data/gastank/')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--mask_idx', type=int, default=1)
    parser.add_argument('--do_all', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.folder + 'masks', exist_ok=True)

    for fn in sorted(glob.glob(args.folder + 'rgb/*.png')):

        image = cv2.imread(fn)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        model_type = "vit_t"
        mobile_sam = sam_model_registry[model_type](checkpoint=f'{code_dir}/../weights/mobile_sam.pt')
        mobile_sam.to(device=device)
        mobile_sam.eval()

        if not args.interactive:
            mask_generator = SamAutomaticMaskGenerator(mobile_sam)
            masks = mask_generator.generate(image)
            max_size = 0
            best_mask = None
            for mask in masks: #glasses 1 and 2 (lower, upper)
                n = np.sum(mask['segmentation'])

                if n > max_size:
                    best_mask = mask['segmentation']
                    max_size = n

            print(max_size)

            mask_img = np.zeros(best_mask.shape)
            mask_img[best_mask] = 255.0

            cv2.imwrite(fn.replace('rgb','masks'), mask_img)

        else:
            # displaying the image 
            cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR)) 

            # setting mouse handler for the image 
            # and calling the click_event() function 
            cv2.setMouseCallback('image', click_event) 

            # wait for a key to be pressed to exit 
            cv2.waitKey(0) 

            # close the window 
            cv2.destroyAllWindows() 

        if not args.do_all:
            exit(0)

        

