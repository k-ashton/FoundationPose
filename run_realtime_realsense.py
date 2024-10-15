# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse

import pyrealsense2 as rs
import numpy as np
import cv2

import torch
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert

import sys
sys.path.append("Grounded-SAM-2")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import groundingdino.datasets.transforms as T

from PIL import Image

SAM2_CHECKPOINT = "Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DEPTH = 10.0

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--object_to_track', type=str, default='mustard bottle')
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/ycb/mustard_tsdf.obj')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    parser.add_argument('--repose_freq', type=int, default=-1)
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    logging.info("realsense setup done")

    # build SAM2 image predictor
    sam2_checkpoint = SAM2_CHECKPOINT
    model_cfg = SAM2_MODEL_CONFIG
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino model
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG, 
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE
    )

    img_transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    logging.info("groundingSAM setup done")

    i =0

    # Streaming loop
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
            K = np.array([[intrinsics.fx, 0.0, intrinsics.ppx], [0.0, intrinsics.fy, intrinsics.ppy], [0.0,0.0,1.0]])

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            color = cv2.cvtColor(color_image.copy(), cv2.COLOR_BGR2RGB)
            depth = depth_image.copy().astype(np.float64)*depth_scale 


            print("DEPTH: ", np.min(depth), np.max(depth), np.min(depth_image), np.max(depth_image))

            remask = i==0 or (args.repose_freq>0 and not i%args.repose_freq)

            if remask:
                text = args.object_to_track + ' . object .'

                image_source = np.asarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
                image_pil = Image.fromarray(image_source)
                image_transformed, _ = img_transform(image_pil, None)


                sam2_predictor.set_image(image_source)

                boxes, confidences, labels = predict(
                    model=grounding_model,
                    image=image_transformed,
                    caption=text,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                )

                # process the box prompt for SAM 2
                h, w, _ = image_source.shape
                boxes = boxes * torch.tensor([w, h, w, h],device='cpu')
                input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

                # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

                # if torch.cuda.get_device_properties(0).major >= 8:
                #     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                #     torch.backends.cuda.matmul.allow_tf32 = True
                #     torch.backends.cudnn.allow_tf32 = True

                masks, scores, logits = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )

                if masks.ndim == 4:
                    masks = masks.squeeze(1)

                #filter 
                correct_label = []
                class_names = []
                for j in range(len(labels)):
                    if (labels[j]==args.object_to_track):
                        correct_label += [j]
                        class_names += [labels[j]]

                confidences = confidences.numpy()[correct_label].tolist()

                if len(confidences)==0:
                    print("No mask detections!")
                    continue

                class_ids = np.array(list(range(len(class_names))))

                labels = [
                    f"{class_name} {confidence:.2f}"
                    for class_name, confidence
                    in zip(class_names, confidences)
                ]

                """
                Visualize image with supervision useful API
                """
                detections = sv.Detections(
                    xyxy=input_boxes[correct_label],  # (n, 4)
                    mask=masks[correct_label].astype(bool),  # (n, h, w)
                    class_id=class_ids
                )

                box_annotator = sv.BoxAnnotator()
                annotated_frame = box_annotator.annotate(scene=color_image.copy(), detections=detections)

                label_annotator = sv.LabelAnnotator()
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

                mask_annotator = sv.MaskAnnotator()
                annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

                #get most confident mask
                best_idx = np.argmax(confidences)

                mask = masks[correct_label][best_idx].astype(bool)

                if debug>=2:
                    cv2.imwrite(os.path.join(f'{debug_dir}/grounded_sam2_{i}.jpg'), annotated_frame)

                pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
                
                if debug>=3:
                    m = mesh.copy()
                    m.apply_transform(pose)
                    m.export(f'{debug_dir}/model_tf.obj')
                    xyz_map = depth2xyzmap(depth, reader.K)
                    valid = depth>=0.001
                    pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                    o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)

            else:
                pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter)

            os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
            np.savetxt(f'{debug_dir}/ob_in_cam/{i}.txt', pose.reshape(4,4))

            center_pose = pose@np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)

            # Render images:
            #   depth align to color on left
            #   depth on right
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((annotated_frame, depth_colormap, vis[...,::-1]))

            if debug>=2:
                os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
                cv2.imwrite(f'{debug_dir}/track_vis/{i}.png', images)
                imageio.imwrite(f'{debug_dir}/color_x.png', color)

            i += 1

            cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            cv2.imshow('Align Example', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            
    finally:
        pipeline.stop()



