#!/usr/bin/env python
import os
import argparse
import os.path as osp
import logging
import time
import sys
import open3d

sys.path.insert(0, osp.dirname(__file__) + '/..')

import torch
import torch.nn as nn

from tdgpd.config import load_cfg_from_file
from tdgpd.utils.logger import setup_logger
from tdgpd.models.build_model import build_model
from tdgpd.utils.checkpoint import CheckPointer
from tdgpd.dataset import build_data_loader
from tdgpd.utils.metric_logger import MetricLogger
from tdgpd.utils.file_logger_cls import file_logger


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


def test_model(model,
               data_batch,
               output_dir=""
               ):
    logger = logging.getLogger("tdgpd.test")
    meters = MetricLogger(delimiter="  ")
    # model.eval()
    model.train()
    end = time.time()
    with torch.no_grad():
        data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items() if isinstance(v, torch.Tensor)}
        data_time = time.time() - end
        end = time.time()
        preds = model(data_batch)
        batch_time = time.time() - end
        with open("inference_time_{}.txt".format("ours"), "a+") as f:
            f.write("{:.4f}\n".format(batch_time * 1000.0))
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        logger.info(
            meters.delimiter.join(
                [
                    "{meters}",
                ]
            ).format(
                meters=str(meters),
            )
        )

        top_H, score = file_logger(data_batch, preds, 0, output_dir, prefix="test", with_label=False)

    return top_H, score


def test(cfg, data_batch, output_dir):
    logger = logging.getLogger("tdgpd.tester")
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
    # test_data_loader = build_data_loader(cfg, mode="test")
    start_time = time.time()
    test_model(model,
               data_batch,
               output_dir=output_dir,
               )
    test_time = time.time() - start_time
    logger.info("Test forward time: {:.2f}s".format(test_time))


def main(data_batch):
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
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("tdgpd", output_dir, prefix="test")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    test(cfg, data_batch, output_dir)


if __name__ == "__main__":
    import numpy as np

    point = open3d.io.read_point_cloud("/home/rayc/Projects/3DGPD/processed_pcd.ply")
    # point = open3d.io.read_point_cloud("/home/rayc/Projects/3DGPD/outputs/pn2_negweight1.0_deeper_val/"
    #                                    "valid_step00001/pred_pts.ply")
    point = np.asarray(point.points)

    # a = np.load("/home/rayc/Projects/3DGPD/data/ycb_data/training_data/87_view_0.p", allow_pickle=True)
    # point = a["point_cloud"].T
    rand_ind = np.random.choice(np.arange(point.shape[0]), 25600, replace=False)
    point = point[rand_ind, :]
    point = torch.tensor(point.T).float().unsqueeze(0)
    # point = np.load("/home/rayc/Projects/3DGPD/point.npy")
    # point = torch.tensor(point)

    data_batch = {"scene_points": point}
    main(data_batch)
