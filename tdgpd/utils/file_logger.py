import os
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d
import matplotlib.pyplot as plt
#from tdgpd.utils.io_utils import mkdir
import time

def file_logger(data_batch, preds, step, output_dir, prefix="", with_label=True):
    step_dir = osp.join(output_dir, "{}_step{:05d}".format(prefix, step))
    #mkdir(step_dir)
    os.makedirs(step_dir, exist_ok=True)
    print("start saving files in ", step_dir)

    if "grasp_logits" in preds.keys():
        # np.savetxt(osp.join(step_dir, "grasp_score.txt"), grasp_score, fmt="%.4f")
        tic = time.time()
        grasp_logits = preds["grasp_logits"]
        grasp_logits = F.softmax(grasp_logits, dim=1).detach().cpu().numpy()
        num_classes = grasp_logits.shape[1]
        score = np.linspace(0, 1, num_classes + 1)[:-1][np.newaxis, :]
        scene_pred = np.sum(score * grasp_logits, axis=1)
        top_ind = np.argsort(-scene_pred)[:100]
        frame = data_batch["frame"][0].cpu().numpy()
        with open("postprocess_time_{}.txt".format("gpd_12chann"), "a+") as f:
            f.write("{:.4f}\n".format((time.time() - tic) * 1000.0))

        np.save("/home/rayc/Projects/3DGPD/top_frames_baseline.npy", frame[top_ind])

        np.savetxt(osp.join(step_dir, "grasp_logits.txt"), grasp_logits, fmt="%.4f")

        return frame[top_ind], scene_pred[top_ind]

    if "scene_score_logits" in preds.keys():
        T_STRIDE = 0.1
        scene_points = data_batch["scene_points"][0].cpu().numpy().T
        np.savetxt(osp.join(step_dir, "scene_points.xyz"), scene_points, fmt="%.4f")
        if with_label:
            scene_score = data_batch["scene_score"][0].cpu().numpy()
            np.savetxt(osp.join(step_dir, "gt_scene_score.txt"), scene_score, fmt="%.4f")
            scene_score_labels = data_batch["scene_score_labels"][0].cpu().numpy()
            np.savetxt(osp.join(step_dir, "gt_scene_score_labels.txt"), scene_score_labels, fmt="%d")
        scene_score_logits = preds["scene_score_logits"][0]
        scene_score_logits = F.softmax(scene_score_logits, dim=0).detach().cpu().numpy().T
        np.savetxt(osp.join(step_dir, "scene_score_logits.txt"), scene_score_logits, fmt="%.4f")

        pred_frame_R = preds["frame_R"]
        pred_frame_R = pred_frame_R[0].transpose(0, 1).detach().cpu().numpy()
        np.savetxt(osp.join(step_dir, "pred_frame_R.txt"), pred_frame_R, fmt="%.4f")
        pred_frame_R = np.reshape(pred_frame_R, (-1, 3, 3))
        pred_frame_t = preds["frame_t"]
        # pred_frame_t = F.softmax(pred_frame_t[0], dim=0).transpose(0, 1).detach().cpu().numpy()
        # t_classes = pred_frame_t.shape[1]
        # t_score = np.linspace(1, 0, t_classes + 1)[1:][np.newaxis, :]
        # pred_frame_t = - (pred_frame_t * t_score).sum(1, keepdims=True) * T_STRIDE * pred_frame_R[:, :, 0] + scene_points
        # pred_frame_t = scene_points
        pred_frame_t = pred_frame_t[0].transpose(0, 1).detach().cpu().numpy()
        np.savetxt(osp.join(step_dir, "pred_frame_t.txt"), pred_frame_t, fmt="%.4f")

        if with_label:
            gt_frame_R = data_batch["best_frame_R"]
            batch_size, _, num_frame_points = gt_frame_R.shape
            gt_frame_R = gt_frame_R[0].transpose(0, 1).detach().cpu().numpy()
            np.savetxt(osp.join(step_dir, "gt_frame_R.txt"), gt_frame_R, fmt="%.4f")
            gt_frame_R = np.reshape(gt_frame_R, (num_frame_points, 3, 3))
            gt_frame_t = data_batch["best_frame_t"][0].transpose(0, 1).detach().cpu().numpy().astype(np.float)
            np.savetxt(osp.join(step_dir, "gt_frame_t.txt"), gt_frame_t, fmt="%.4f")

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(scene_points)

        score_classes = scene_score_logits.shape[1]
        score = np.linspace(0, 1, score_classes + 1)[:-1][np.newaxis, :]
        scene_pred = np.sum(score * scene_score_logits, axis=1)

        color_grad = 1024
        cmap = plt.get_cmap("jet", color_grad)
        if with_label:
            gt_color = np.zeros((scene_points.shape[0], 3))
            scene_score = scene_score_labels / score_classes
            for i in range(gt_color.shape[0]):
                gt_color[i, :] = cmap(scene_score[i])[:3]
            pcd.colors = open3d.utility.Vector3dVector(gt_color)

            open3d.io.write_point_cloud(osp.join(step_dir, "gt_pts.ply"), pcd)

            points = []
            color = []
            triangles = []
            for i, j in enumerate(np.arange(0, num_frame_points - 1, 2)):
                points.append(scene_points[j, :])
                points.append(gt_frame_t[j, :] * 0.5 + scene_points[j, :] * 0.5 + np.array([0.0001, 0.0001, 0.0001]))
                points.append(gt_frame_t[j, :])
                points.append(gt_frame_t[j, :])
                points.append(gt_frame_t[j, :] + gt_frame_R[j, :, 0] * 0.01 + gt_frame_R[j, :, 1] * 0.001)
                points.append(gt_frame_t[j, :] + gt_frame_R[j, :, 0] * 0.01)
                points.append(gt_frame_t[j, :])
                points.append(gt_frame_t[j, :] + gt_frame_R[j, :, 1] * 0.01 + gt_frame_R[j, :, 2] * 0.001)
                points.append(gt_frame_t[j, :] + gt_frame_R[j, :, 1] * 0.01)
                points.append(gt_frame_t[j, :])
                points.append(gt_frame_t[j, :] + gt_frame_R[j, :, 2] * 0.01 + gt_frame_R[j, :, 0] * 0.001)
                points.append(gt_frame_t[j, :] + gt_frame_R[j, :, 2] * 0.01)
                color.append([1, 0, 0])
                color.append([1, 0, 0])
                color.append([1, 0, 0])
                color.append([0, 1, 0])
                color.append([0, 1, 0])
                color.append([0, 1, 0])
                color.append([1, 1, 0])
                color.append([1, 1, 0])
                color.append([1, 1, 0])
                color.append([0, 0, 1])
                color.append([0, 0, 1])
                color.append([0, 0, 1])
                # triangles.append([12 * i, 12 * i + 1, 12 * i + 2])
                # triangles.append([12 * i + 3, 12 * i + 4, 12 * i + 5])
                triangles.append([12 * i + 6, 12 * i + 7, 12 * i + 8])
                # triangles.append([12 * i + 9, 12 * i + 10, 12 * i + 11])

            points = np.stack(points)
            color = np.stack(color)
            mesh = open3d.geometry.TriangleMesh()
            mesh.vertices = open3d.utility.Vector3dVector(points)
            mesh.vertex_colors = open3d.utility.Vector3dVector(color)
            mesh.triangles = open3d.utility.Vector3iVector(triangles)
            open3d.io.write_triangle_mesh(osp.join(step_dir, "gt_frame.ply"), mesh)

        points = []
        color = []
        triangles = []
        for i, j in enumerate(np.arange(0, scene_points.shape[0], 2)):
            points.append(scene_points[j, :])
            points.append(pred_frame_t[j, :] * 0.5 + scene_points[j, :] * 0.5 + np.array([0.0001, 0.0001, 0.0001]))
            points.append(pred_frame_t[j, :])
            points.append(pred_frame_t[j, :])
            points.append(pred_frame_t[j, :] + pred_frame_R[j, :, 0] * 0.01 + pred_frame_R[j, :, 1] * 0.001)
            points.append(pred_frame_t[j, :] + pred_frame_R[j, :, 0] * 0.01)
            points.append(pred_frame_t[j, :])
            points.append(pred_frame_t[j, :] + pred_frame_R[j, :, 1] * 0.01 + pred_frame_R[j, :, 2] * 0.001)
            points.append(pred_frame_t[j, :] + pred_frame_R[j, :, 1] * 0.01)
            points.append(pred_frame_t[j, :])
            points.append(pred_frame_t[j, :] + pred_frame_R[j, :, 2] * 0.01 + pred_frame_R[j, :, 0] * 0.001)
            points.append(pred_frame_t[j, :] + pred_frame_R[j, :, 2] * 0.01)
            color.append([1, 0, 0])
            color.append([1, 0, 0])
            color.append([1, 0, 0])
            color.append([0, 1, 0])
            color.append([0, 1, 0])
            color.append([0, 1, 0])
            color.append([1, 1, 0])
            color.append([1, 1, 0])
            color.append([1, 1, 0])
            color.append([0, 0, 1])
            color.append([0, 0, 1])
            color.append([0, 0, 1])
            # triangles.append([12 * i, 12 * i + 1, 12 * i + 2])
            # triangles.append([12 * i + 3, 12 * i + 4, 12 * i + 5])
            triangles.append([12 * i + 6, 12 * i + 7, 12 * i + 8])
            # triangles.append([12 * i + 9, 12 * i + 10, 12 * i + 11])

        points = np.stack(points)
        color = np.stack(color)
        mesh = open3d.geometry.TriangleMesh()
        mesh.vertices = open3d.utility.Vector3dVector(points)
        mesh.vertex_colors = open3d.utility.Vector3dVector(color)
        mesh.triangles = open3d.utility.Vector3iVector(triangles)
        open3d.io.write_triangle_mesh(osp.join(step_dir, "pred_frame.ply"), mesh)

        pred_color = np.zeros((scene_points.shape[0], 3))
        # scene_pred = scene_pred / np.max(scene_pred)
        for i in range(pred_color.shape[0]):
            pred_color[i, :] = cmap(scene_pred[i])[:3]
        pcd.colors = open3d.utility.Vector3dVector(pred_color)
        open3d.io.write_point_cloud(osp.join(step_dir, "pred_pts.ply"), pcd)
        np.savetxt(osp.join(step_dir, "pred_scene_score.txt"), scene_pred, fmt="%.4f")

        movable_logits = preds["movable_logits"][0]
        movable_logits = F.softmax(movable_logits, dim=0).detach().cpu().numpy()[1]
        if with_label:
            movable_labels = data_batch["scene_movable_labels"][0].cpu().numpy().astype(np.float)
            print("Movable points: ", np.sum(np.max(movable_labels, axis=0)))
            for direction in range(movable_labels.shape[0]):
                gt_color = np.zeros((scene_points.shape[0], 3))
                for i in range(gt_color.shape[0]):
                    gt_color[i, :] = cmap(movable_labels[direction, i])[:3]
                pcd.colors = open3d.utility.Vector3dVector(gt_color)
                open3d.io.write_point_cloud(osp.join(step_dir, "gt_movable_dir{}.ply".format(direction)), pcd)

        gt_color = np.zeros((scene_points.shape[0], 3))
        for i in range(gt_color.shape[0]):
            gt_color[i, :] = cmap(movable_logits[i])[:3]
        pcd.colors = open3d.utility.Vector3dVector(gt_color)
        open3d.io.write_point_cloud(osp.join(step_dir, "pred_movable.ply"), pcd)

        if not with_label:  # save top frames for real experiments
            from tdgpd.eval_experiment.eval_point_cloud import EvalExpCloud
            from tdgpd.eval_experiment.torch_scene_point_cloud import TorchScenePointCloud
            view_cloud = open3d.geometry.PointCloud()
            view_cloud.points = open3d.utility.Vector3dVector(scene_points)
            evaluation_point_cloud = EvalExpCloud(view_cloud)

            K = 50
            top_ind = np.argsort(-scene_pred)[:K]
            # movable = movable_logits[top_ind] > 0.5
            # top_ind = top_ind[movable]


            top_H = []
            score = []
            for ind in top_ind:
                R = pred_frame_R[ind]
                # schmidt orthogonalization
                x = R[:, 0]
                x = x / np.linalg.norm(x)
                y = R[:, 1]
                y = y - np.sum(x * y) * x
                y = y / np.linalg.norm(y)
                z = np.cross(x, y)
                R = np.stack([x, y, z], axis=1)
                t = pred_frame_t[ind]
                H = np.eye(4)
                H[:3, :3] = R
                H[:3, 3] = t

                if len(top_H) > 0:
                    tH = np.stack(top_H, axis=0)
                    He = H[np.newaxis, ...]
                    tH = np.sum(np.abs(tH - He), axis=(1, 2))
                    if np.min(tH) < 0.05:
                        continue

                T_global_to_local = np.linalg.inv(H)
                T_global_to_local = torch.tensor(T_global_to_local, device=evaluation_point_cloud.device).float()

                if evaluation_point_cloud.view_non_collision(T_global_to_local):
                    top_H.append(H)
                    score.append(scene_pred[ind])


            if len(top_H) > 0:
                top_H = np.stack(top_H, axis=0)
                np.save("top_frames.npy", top_H)
            else:
                print("### No viable frames in top 50. ###")
            return (top_H, score)

    print("saving finished")
