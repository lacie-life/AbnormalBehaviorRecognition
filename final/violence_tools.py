import argparse
import cv2
import mmcv
import numpy as np
import os
import os.path as osp
import shutil
import torch
import warnings
from scipy.optimize import linear_sum_assignment
import tqdm
from pyskl.apis import inference_recognizer, init_recognizer

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    def inference_detector(*args, **kwargs):
        pass

    def init_detector(*args, **kwargs):
        pass
    warnings.warn(
        'Failed to import `inference_detector` and `init_detector` from `mmdet.apis`. '
        'Make sure you can successfully import these if you want to use related features. '
    )

try:
    from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
except (ImportError, ModuleNotFoundError):
    def init_pose_model(*args, **kwargs):
        pass

    def inference_top_down_pose_model(*args, **kwargs):
        pass

    def vis_pose_result(*args, **kwargs):
        pass

    warnings.warn(
        'Failed to import `init_pose_model`, `inference_top_down_pose_model`, `vis_pose_result` from '
        '`mmpose.apis`. Make sure you can successfully import these if you want to use related features. '
    )


try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')


FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 1.5
FONTCOLOR = (0, 0, 0)  # BGR, white
THICKNESS = 2
LINETYPE = 1
BATCH_SIZE = 20
def parse_args():
    config = {}

    args = argparse.Namespace(
        video='test.avi',
        out_filename='test2.mp4',
        config='configs/posec3d/slowonly_r50_ntu120_xsub/joint.py',
        checkpoint='https://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_ntu120_xsub/joint.pth',
        det_config='faster_rcnn_r50_fpn_1x_coco-person.py',
        det_checkpoint='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth',
        pose_config='hrnet_w32_coco_256x192.py',
        pose_checkpoint='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
        det_score_thr=0.9,
        label_map='tools/data/label_map/nturgbd_120.txt',
        device='cuda:0',
        short_side=480)
    return args



def dist_ske(ske1, ske2):
    dist = np.linalg.norm(ske1[:, :2] - ske2[:, :2], axis=1) * 2
    diff = np.abs(ske1[:, 2] - ske2[:, 2])
    return np.sum(np.maximum(dist, diff))


def frame_extraction(video_path, short_side):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    while flag:
        if new_h is None:
            h, w, _ = frame.shape
            new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))

        frame = mmcv.imresize(frame, (new_w, new_h))

        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return frame_paths, frames

def detection_inference(args, frame_paths):
    """Detect human boxes given frame paths.

    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model is not None, ('Failed to build the detection model. Check if you have installed mmcv-full properly. '
                               'You should first install mmcv-full successfully, then install mmdet, mmpose. ')
    assert model.CLASSES[0] == 'person', 'We require you to use a detector trained on COCO'
    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()
    return results


def pose_inference(args, frame_paths, det_results):
    model = init_pose_model(args.pose_config, args.pose_checkpoint,
                            args.device)
    ret = []
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f, d in zip(frame_paths, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
    return ret

def pose_tracking(pose_results, max_tracks=2, thre=30):
    tracks, num_tracks = [], 0
    num_joints = None
    for idx, poses in enumerate(pose_results):
        if len(poses) == 0:
            continue
        if num_joints is None:
            num_joints = poses[0].shape[0]
        track_proposals = [t for t in tracks if t['data'][-1][0] > idx - thre]
        n, m = len(track_proposals), len(poses)
        scores = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                scores[i][j] = dist_ske(track_proposals[i]['data'][-1][1], poses[j])

        row, col = linear_sum_assignment(scores)
        for r, c in zip(row, col):
            track_proposals[r]['data'].append((idx, poses[c]))
        if m > n:
            for j in range(m):
                if j not in col:
                    num_tracks += 1
                    new_track = dict(data=[])
                    new_track['track_id'] = num_tracks
                    new_track['data'] = [(idx, poses[j])]
                    tracks.append(new_track)
    tracks.sort(key=lambda x: -len(x['data']))
    # print(pose_results, num_joints)
    if num_joints == None:
        return None
    result = np.zeros((max_tracks, len(pose_results), num_joints, 3), dtype=np.float16)
    for i, track in enumerate(tracks[:max_tracks]):
        for item in track['data']:
            idx, pose = item
            result[i, idx] = pose
    return result[..., :2], result[..., 2]


args = parse_args()

frame_paths_l, original_frames_l = frame_extraction(args.video,
                                                    args.short_side)
result_frames = []

videos_cnt = len(frame_paths_l) // BATCH_SIZE

for ef in tqdm.tqdm(range(videos_cnt)):
    frame_paths = frame_paths_l[ef*BATCH_SIZE:(ef+1)*BATCH_SIZE]
    original_frames = original_frames_l[ef*BATCH_SIZE:(ef+1)*BATCH_SIZE]

    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape

    config = mmcv.Config.fromfile(args.config)
    config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']
    # Are we using GCN for Infernece?
    GCN_flag = 'GCN' in config.model.type
    GCN_nperson = None
    if GCN_flag:
        format_op = [op for op in config.data.test.pipeline if op['type'] == 'FormatGCNInput'][0]
        # We will set the default value of GCN_nperson to 2, which is
        # the default arg of FormatGCNInput
        GCN_nperson = format_op.get('num_person', 2)

    model = init_recognizer(config, args.checkpoint, args.device)


    # Load label_map
    label_map = [x.strip() for x in open(args.label_map).readlines()]

    # Get Human detection results
    det_results = detection_inference(args, frame_paths)
    torch.cuda.empty_cache()

    pose_results = pose_inference(args, frame_paths, det_results)
    torch.cuda.empty_cache()

    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)


    if len(pose_results[-1]) > 0:
        if GCN_flag:
            # We will keep at most `GCN_nperson` persons per frame.
            tracking_inputs = [[pose['keypoints'] for pose in poses] for poses in pose_results]
            keypoint, keypoint_score = pose_tracking(tracking_inputs, max_tracks=GCN_nperson)
            fake_anno['keypoint'] = keypoint
            fake_anno['keypoint_score'] = keypoint_score
        else:
            num_person = max([len(x) for x in pose_results])
            # Current PoseC3D models are trained on COCO-keypoints (17 keypoints)
            num_keypoint = 17
            keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                                dtype=np.float16)
            keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                                      dtype=np.float16)
            for i, poses in enumerate(pose_results):
                for j, pose in enumerate(poses):
                    pose = pose['keypoints']
                    keypoint[j, i] = pose[:, :2]
                    keypoint_score[j, i] = pose[:, 2]
            fake_anno['keypoint'] = keypoint
            fake_anno['keypoint_score'] = keypoint_score

    else:
        num_keypoint = 17
        keypoint = np.zeros((0, num_frame, num_keypoint, 2),
                                dtype=np.float16)
        keypoint_score = np.zeros((0, num_frame, num_keypoint),
                                  dtype=np.float16)
        fake_anno['keypoint'] = keypoint
        fake_anno['keypoint_score'] = keypoint_score

    results = inference_recognizer(model, fake_anno)


    action_label = label_map[results[0][0]]

    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
                                 args.device)
    vis_frames = [
        vis_pose_result(pose_model, frame_paths[i], pose_results[i])
        for i in range(num_frame)
    ]
    for frame in vis_frames:
        if action_label == "fight":
            REDCOLOR = (0, 0, 255)  # BGR, white
            cv2.putText(frame, "fight", (10, 30), FONTFACE, FONTSCALE,
                        REDCOLOR, THICKNESS, LINETYPE)
        else:
            cv2.putText(frame, action_label, (10, 30), FONTFACE, FONTSCALE,
                        FONTCOLOR, THICKNESS, LINETYPE)


    result_frames += vis_frames

vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in result_frames], fps=24)
vid.write_videofile(args.out_filename, remove_temp=True)

tmp_frame_dir = osp.dirname(frame_paths[0])
shutil.rmtree(tmp_frame_dir)
