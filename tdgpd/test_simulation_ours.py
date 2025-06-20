#!/usr/bin/env python#!/usr/bin/env python
import argparse
import os
import os.path as osp
import logging
import time
import sys
import open3d
import numpy as np
from tqdm import tqdm

sys.path.insert(0, osp.dirname(__file__) + '/..')

import torch
import torch.nn as nn
import torch.nn.functional as F

from tdgpd.config import load_cfg_from_file
from tdgpd.utils.io_utils import mkdir
from tdgpd.utils.logger import setup_logger
from tdgpd.models.build_model import build_model
from tdgpd.utils.checkpoint import CheckPointer
from tdgpd.dataset import build_data_loader
from tdgpd.utils.metric_logger import MetricLogger

from tdgpd.eval_experiment.eval_point_cloud import EvalExpCloud
from tdgpd.eval_experiment.torch_scene_point_cloud import TorchScenePointCloud


def parse_args():
    parser = argparse.ArgumentParser(description="3D Grasp Detection Testing")
    parser.add_argument(
        "--cfg",
        dest="config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def eval_preds(preds, data_batch, k=10, output_dir=None):
    eval_results = {}

    view_point_cloud = data_batch["scene_points"][0].cpu().numpy().T  # (N, 3)
    scene_point_cloud = data_batch["complete_cloud"][0].cpu().numpy().T  # (M, 3)
    label_array = data_batch['scene_label'][0].cpu().numpy()
    scene_normal = data_batch['scene_normal'][0].cpu().numpy()

    scene_cloud = open3d.geometry.PointCloud()
    scene_cloud.points = open3d.utility.Vector3dVector(scene_point_cloud)
    scene_cloud.normals = open3d.utility.Vector3dVector(scene_normal)
    view_cloud = open3d.geometry.PointCloud()
    view_cloud.points = open3d.utility.Vector3dVector(view_point_cloud)

    scene = TorchScenePointCloud(scene_cloud, label_array)
    evaluation_point_cloud = EvalExpCloud(view_cloud)

    # grasp_logits = F.softmax(preds["local_search_logits"][0], dim=0).squeeze(-1).cpu().numpy().T
    grasp_logits = F.softmax(preds["scene_score_logits"][0], dim=0).cpu().numpy().T
    score_classes = grasp_logits.shape[-1]
    # pred_class = np.argmax(grasp_logits, axis=-1)
    # descenting_inds = np.where(pred_class == (score_classes - 1))[0]
    # np.random.shuffle(descenting_inds)
    #
    score = np.linspace(0, 1, score_classes + 1)[:-1][np.newaxis, :]
    pred_score = np.sum(grasp_logits * score, axis=1)
    descenting_inds = np.argsort(-pred_score)

    frame_R = preds["frame_R"][0].cpu().numpy().T
    frame_t = preds["frame_t"][0].transpose(0, 1).detach().cpu().numpy()
    T_STRIDE = 0.1

    topk = []
    antipodal_score = []
    collision = []
    multi_objects = []
    frames = []
    inds = []
    for i in descenting_inds:
        R = frame_R[i].reshape(3, 3)
        t = frame_t[i]

        # if t[-1] < 0.80:
        #     continue

        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = t
        T_global_to_local = np.linalg.inv(H)
        T_global_to_local = torch.tensor(T_global_to_local, device=evaluation_point_cloud.device).float()

        # detect collision with view point cloud
        if evaluation_point_cloud.view_non_collision(T_global_to_local):
            eval_frame_results = evaluation_point_cloud.eval_frame(T_global_to_local, scene)
            antipodal_score.append(eval_frame_results["antipodal_score"])
            collision.append(eval_frame_results["collision"])
            multi_objects.append(eval_frame_results["multi_objects"])
            frames.append(H.reshape(-1))
            inds.append(i)

        if len(antipodal_score) == 10:
            break

    eval_results["antipodal_score"] = np.stack(antipodal_score)
    eval_results["collision"] = np.stack(collision)
    eval_results["multi_objects"] = np.stack(multi_objects)
    eval_results["frames"] = np.stack(frames)
    eval_results["inds"] = np.stack(inds)

    if output_dir is not None:
        if not osp.exists(output_dir):
            os.mkdir(output_dir)
        np.savetxt(osp.join(output_dir, "view_point_cloud.txt"), view_point_cloud, fmt="%.4f")
        np.savetxt(osp.join(output_dir, "scene_point_cloud.txt"), scene_point_cloud, fmt="%.4f")
        np.savetxt(osp.join(output_dir, "antipodal_score.txt"), eval_results["antipodal_score"], fmt="%.4f")
        np.savetxt(osp.join(output_dir, "collision.txt"), eval_results["collision"], fmt="%d")
        np.savetxt(osp.join(output_dir, "multi_objects.txt"), eval_results["multi_objects"], fmt="%d")
        np.savetxt(osp.join(output_dir, "frames.txt"), eval_results["frames"], fmt="%.4f")
        np.savetxt(osp.join(output_dir, "inds.txt"), eval_results["inds"], fmt="%.4f")

    return eval_results


def test_model(model,
               data_loader,
               log_period=10,
               output_dir="",
               TOPK=10,
               ):
    logger = logging.getLogger("tdgpd.test_simul_ours")
    meters = MetricLogger(delimiter="  ")
    model.eval()
    # model.train()
    end = time.time()

    multi_objects = []
    collision = []
    antipodal_score = []
    with torch.no_grad():
        for iteration, data_batch in enumerate(data_loader):
            data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items() if isinstance(v, torch.Tensor)}
            data_time = time.time() - end
            preds = model(data_batch)
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            if iteration % log_period == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "iter: {:05d}",
                            "{meters}",
                        ]
                    ).format(
                        iteration, meters=str(meters),
                    )
                )
            eval_results = eval_preds(preds, data_batch, k=TOPK,
                                      output_dir=osp.join(output_dir, "step_{:06d}".format(iteration))
                                      if iteration % 1 == 0 else None)
            multi_objects.append(eval_results["multi_objects"])
            collision.append(eval_results["collision"])
            antipodal_score.append(eval_results["antipodal_score"])

    multi_objects = np.concatenate(multi_objects)
    collision = np.concatenate(collision)
    antipodal_score = np.concatenate(antipodal_score)

    np.savetxt(osp.join(output_dir, "multi_objects.txt"), multi_objects, fmt="%d")
    np.savetxt(osp.join(output_dir, "collision.txt"), collision, fmt="%d")
    np.savetxt(osp.join(output_dir, "antipodal_score.txt"), antipodal_score, fmt="%.4f")

    logger.info("Total evaluated grasps: {}".format(multi_objects.shape[0]))
    logger.info("multi_objects: {:.2f}%, collision: {:.2f}%, antipodal_score: {:.4f}".format(
        multi_objects.astype(np.float).sum() / np.logical_not(collision).astype(np.float).sum() * 100.0,
        collision.astype(np.float).mean() * 100.0,
        antipodal_score.sum() / (np.logical_not(np.logical_or(multi_objects, collision)).astype(np.float).sum())
    ))


def test(cfg, output_dir):
    logger = logging.getLogger("tdgpd.tester_simul_ours")
    # build model
    model, _, _ = build_model(cfg)
    model = nn.DataParallel(model).cuda()
    # build checkpointer
    checkpointer = CheckPointer(model, save_dir=output_dir)

    if cfg.TEST.WEIGHT:
        weight_path = cfg.TEST.WEIGHT.replace("@", output_dir)
        checkpointer.load(weight_path, resume=False)
    else:
        checkpointer.load(None, resume=True)

    # build data loader
    test_data_loader = build_data_loader(cfg, mode="test")
    start_time = time.time()
    test_model(model,
               test_data_loader,
               output_dir=output_dir,
               TOPK=cfg.TEST.TOPK)
    test_time = time.time() - start_time
    logger.info("Test forward time: {:.2f}s".format(test_time))


def main():
    args = parse_args()
    num_gpus = torch.cuda.device_count()

    cfg = load_cfg_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    # assert cfg.TEST.BATCH_SIZE == 1

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace("configs", "outputs")
        output_dir = output_dir.replace('@', config_path)
        mkdir(output_dir)

    logger = setup_logger("tdgpd", output_dir, prefix="test")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    test(cfg, output_dir)


if __name__ == "__main__":
    main()

import argparse
import os
import os.path as osp
import logging
import time
import sys
import open3d
import numpy as np
from tqdm import tqdm

sys.path.insert(0, osp.dirname(__file__) + '/..')

import torch
import torch.nn as nn
import torch.nn.functional as F

from tdgpd.config import load_cfg_from_file
from tdgpd.utils.io_utils import mkdir
from tdgpd.utils.logger import setup_logger
from tdgpd.models.build_model import build_model
from tdgpd.utils.checkpoint import CheckPointer
from tdgpd.dataset import build_data_loader
from tdgpd.utils.metric_logger import MetricLogger
from tdgpd.utils.file_logger import file_logger

from tdgpd.eval_experiment.eval_point_cloud import EvalExpCloud
from tdgpd.eval_experiment.torch_scene_point_cloud import TorchScenePointCloud


def parse_args():
    parser = argparse.ArgumentParser(description="3D Grasp Detection Testing")
    parser.add_argument(
        "--cfg",
        dest="config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def eval_preds(preds, data_batch, k=10, output_dir=None):
    eval_results = {}

    view_point_cloud = data_batch["scene_points"][0].cpu().numpy().T  # (N, 3)
    scene_point_cloud = data_batch["complete_cloud"][0].cpu().numpy().T  # (M, 3)
    label_array = data_batch['scene_label'][0].cpu().numpy()
    scene_normal = data_batch['scene_normal'][0].cpu().numpy()

    scene_cloud = open3d.geometry.PointCloud()
    scene_cloud.points = open3d.utility.Vector3dVector(scene_point_cloud)
    scene_cloud.normals = open3d.utility.Vector3dVector(scene_normal)
    view_cloud = open3d.geometry.PointCloud()
    view_cloud.points = open3d.utility.Vector3dVector(view_point_cloud)

    scene = TorchScenePointCloud(scene_cloud, label_array)
    evaluation_point_cloud = EvalExpCloud(view_cloud)

    # grasp_logits = F.softmax(preds["local_search_logits"][0], dim=0).squeeze(-1).cpu().numpy().T
    grasp_logits = F.softmax(preds["scene_score_logits"][0], dim=0).cpu().numpy().T
    score_classes = grasp_logits.shape[-1]
    # pred_class = np.argmax(grasp_logits, axis=-1)
    # descenting_inds = np.where(pred_class == (score_classes - 1))[0]
    # np.random.shuffle(descenting_inds)
    #
    score = np.linspace(0, 1, score_classes + 1)[:-1][np.newaxis, :]
    pred_score = np.sum(grasp_logits * score, axis=1)
    descenting_inds = np.argsort(-pred_score)

    frame_R = preds["frame_R"][0].cpu().numpy().T
    frame_t = F.softmax(preds["frame_t"][0], dim=0).cpu().numpy().T
    T_STRIDE = 0.1

    topk = []
    antipodal_score = []
    collision = []
    multi_objects = []
    frames = []
    inds = []
    for i in descenting_inds:
        R = frame_R[i].reshape(3, 3)
        # schmidt orthogonalization
        x = R[:, 0]
        x = x / np.linalg.norm(x)
        y = R[:, 1]
        y = y - np.sum(x * y) * x
        y = y / np.linalg.norm(y)
        z = np.cross(x, y)
        R = np.stack([x, y, z], axis=1)
        t = frame_t[i]
        t_classes = t.shape[0]
        t_score = np.linspace(1, 0, t_classes + 1)[1:][np.newaxis, :]
        t = -(t * t_score).sum() * T_STRIDE * x + view_point_cloud[i]

        # if t[-1] < 0.80:
        #     continue

        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = t
        T_global_to_local = np.linalg.inv(H)
        T_global_to_local = torch.tensor(T_global_to_local, device=evaluation_point_cloud.device).float()

        # detect collision with view point cloud
        if evaluation_point_cloud.view_non_collision(T_global_to_local):
            eval_frame_results = evaluation_point_cloud.eval_frame(T_global_to_local, scene)
            antipodal_score.append(eval_frame_results["antipodal_score"])
            collision.append(eval_frame_results["collision"])
            multi_objects.append(eval_frame_results["multi_objects"])
            frames.append(H.reshape(-1))
            inds.append(i)

        if len(antipodal_score) == 10:
            break

    eval_results["antipodal_score"] = np.stack(antipodal_score)
    eval_results["collision"] = np.stack(collision)
    eval_results["multi_objects"] = np.stack(multi_objects)
    eval_results["frames"] = np.stack(frames)
    eval_results["inds"] = np.stack(inds)

    if output_dir is not None:
        if not osp.exists(output_dir):
            os.mkdir(output_dir)
        np.savetxt(osp.join(output_dir, "view_point_cloud.txt"), view_point_cloud, fmt="%.4f")
        np.savetxt(osp.join(output_dir, "scene_point_cloud.txt"), scene_point_cloud, fmt="%.4f")
        np.savetxt(osp.join(output_dir, "antipodal_score.txt"), eval_results["antipodal_score"], fmt="%.4f")
        np.savetxt(osp.join(output_dir, "collision.txt"), eval_results["collision"], fmt="%d")
        np.savetxt(osp.join(output_dir, "multi_objects.txt"), eval_results["multi_objects"], fmt="%d")
        np.savetxt(osp.join(output_dir, "frames.txt"), eval_results["frames"], fmt="%.4f")
        np.savetxt(osp.join(output_dir, "inds.txt"), eval_results["inds"], fmt="%.4f")

    return eval_results


def test_model(model,
               data_loader,
               log_period=10,
               output_dir="",
               TOPK=10,
               ):
    logger = logging.getLogger("tdgpd.test_simul_ours")
    meters = MetricLogger(delimiter="  ")
    model.eval()
    # model.train()
    end = time.time()

    multi_objects = []
    collision = []
    antipodal_score = []
    with torch.no_grad():
        for iteration, data_batch in enumerate(data_loader):
            data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items() if isinstance(v, torch.Tensor)}
            data_time = time.time() - end
            preds = model(data_batch)
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            if iteration % log_period == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "iter: {:05d}",
                            "{meters}",
                        ]
                    ).format(
                        iteration, meters=str(meters),
                    )
                )
            eval_results = eval_preds(preds, data_batch, k=TOPK,
                                      output_dir=osp.join(output_dir, "step_{:06d}".format(iteration))
                                      if iteration % 1 == 0 else None)
            multi_objects.append(eval_results["multi_objects"])
            collision.append(eval_results["collision"])
            antipodal_score.append(eval_results["antipodal_score"])

    multi_objects = np.concatenate(multi_objects)
    collision = np.concatenate(collision)
    antipodal_score = np.concatenate(antipodal_score)

    np.savetxt(osp.join(output_dir, "multi_objects.txt"), multi_objects, fmt="%d")
    np.savetxt(osp.join(output_dir, "collision.txt"), collision, fmt="%d")
    np.savetxt(osp.join(output_dir, "antipodal_score.txt"), antipodal_score, fmt="%.4f")

    logger.info("Total evaluated grasps: {}".format(multi_objects.shape[0]))
    logger.info("multi_objects: {:.2f}%, collision: {:.2f}%, antipodal_score: {:.4f}".format(
        (multi_objects.astype(np.float).sum() - collision.astype(np.float).sum())/np.logical_not(collision).astype(np.float).sum() * 100.0,
        collision.astype(np.float).mean() * 100.0,
        antipodal_score.sum() / (np.logical_not(np.logical_or(multi_objects, collision)).astype(np.float).sum())
    ))


def test(cfg, output_dir):
    logger = logging.getLogger("tdgpd.tester_simul_ours")
    # build model
    model, _, _ = build_model(cfg)
    model = nn.DataParallel(model).cuda()
    # build checkpointer
    checkpointer = CheckPointer(model, save_dir=output_dir)

    if cfg.TEST.WEIGHT:
        weight_path = cfg.TEST.WEIGHT.replace("@", output_dir)
        checkpointer.load(weight_path, resume=False)
    else:
        checkpointer.load(None, resume=True)

    # build data loader
    test_data_loader = build_data_loader(cfg, mode="test")
    start_time = time.time()
    test_model(model,
               test_data_loader,
               output_dir=output_dir,
               TOPK=cfg.TEST.TOPK)
    test_time = time.time() - start_time
    logger.info("Test forward time: {:.2f}s".format(test_time))


def main():
    args = parse_args()
    num_gpus = torch.cuda.device_count()

    cfg = load_cfg_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    # assert cfg.TEST.BATCH_SIZE == 1

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace("configs", "outputs")
        output_dir = output_dir.replace('@', config_path)
        mkdir(output_dir)

    logger = setup_logger("tdgpd", output_dir, prefix="test")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    test(cfg, output_dir)


if __name__ == "__main__":
    main()
