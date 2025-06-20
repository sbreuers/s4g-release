#!/usr/bin/env python
import argparse
import os
import os.path as osp
import logging
import time
import sys
import open3d
import numpy as np

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


def eval_preds(preds, data_batch, output_dir=None):
    eval_results = {}
    grasp_logits = F.softmax(preds["grasp_logits"], dim=-1).cpu().numpy()
    score_classes = grasp_logits.shape[-1]
    score = np.linspace(0, 1, score_classes + 1)[:-1][np.newaxis, :]
    pred_score = np.sum(grasp_logits * score, axis=1)
    # pred_class = np.argmax(grasp_logits, axis=-1)
    # topk = (pred_class == (score_classes - 1))
    topk = np.argsort(-pred_score)[:10]
    antipodal_score = data_batch["antipodal_score"][0].cpu().numpy()
    non_collision_bool = data_batch["non_collision_bool"][0].cpu().numpy()
    single_label_bool = data_batch["single_label_bool"][0].cpu().numpy()
    eval_results["antipodal_score"] = antipodal_score[topk]
    eval_results["collision"] = np.logical_not(non_collision_bool[topk])
    eval_results["multi_objects"] = np.logical_not(single_label_bool[topk])
    frames = data_batch["frame"][0].cpu().numpy()[topk]
    eval_results["frames"] = frames.reshape(frames.shape[0], -1)
    view_point_cloud = data_batch["scene_points"][0].cpu().numpy().T  # (N, 3)
    scene_point_cloud = data_batch["complete_cloud"][0].cpu().numpy().T  # (M, 3)

    if output_dir is not None:
        if not osp.exists(output_dir):
            os.mkdir(output_dir)
        np.savetxt(osp.join(output_dir, "view_point_cloud.txt"), view_point_cloud, fmt="%.4f")
        np.savetxt(osp.join(output_dir, "scene_point_cloud.txt"), scene_point_cloud, fmt="%.4f")
        np.savetxt(osp.join(output_dir, "antipodal_score.txt"), eval_results["antipodal_score"], fmt="%.4f")
        np.savetxt(osp.join(output_dir, "collision.txt"), eval_results["collision"], fmt="%d")
        np.savetxt(osp.join(output_dir, "multi_objects.txt"), eval_results["multi_objects"], fmt="%d")
        np.savetxt(osp.join(output_dir, "frames.txt"), eval_results["frames"], fmt="%.4f")

    return eval_results


def test_model(model,
               data_loader,
               log_period=10,
               output_dir="",
               TOPK=10,
               ):
    logger = logging.getLogger("tdgpd.test_simul_baseline")
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
                            "{:05d}",
                            "{meters}",
                        ]
                    ).format(
                        iteration,
                        meters=str(meters),
                    )
                )
            eval_results = eval_preds(preds, data_batch,
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
        # multi_objects.astype(np.float).sum() / np.logical_not(collision).astype(np.float).sum() * 100.0,
        (multi_objects.astype(np.float).sum() - collision.astype(np.float).sum())/np.logical_not(collision).astype(np.float).sum() * 100.0,
        collision.astype(np.float).mean() * 100.0,
        antipodal_score.sum()/(np.logical_not(np.logical_or(multi_objects, collision)).astype(np.float).sum())
    ))


def test(cfg, output_dir):
    logger = logging.getLogger("tdgpd.tester_simul_baseline")
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
    assert cfg.TEST.BATCH_SIZE == 1

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
