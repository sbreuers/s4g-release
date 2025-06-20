import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import pickle
import open3d
from time import time
from configs import config
import torch
from tqdm import trange
import sklearn.metrics

GASKET_RADIUS = 0.012
COS_THR = 0.99
THETA_SEARCH = list(range(0, 360, 30))
THETA_NUM = len(THETA_SEARCH)

COLLISION_POINT_THRESHOLD = 1 # 50

POINT_VOXEL_SIZE = 0.003 # 0.004


class GenerateContactObjectData:
    def __init__(self, model_dir, output_dir):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.zeros_x_direction = torch.tensor([[0, 1, 0]], device=self.device).float()

        # Common data pre-processing
        numpy_frame_move_back = np.eye(4)
        numpy_frame_move_back[0, 3] = -(config.FINGER_LENGTH - GASKET_RADIUS)
        numpy_local_search_to_local = np.tile(np.eye(4), (THETA_NUM, 1, 1))
        for i in range(THETA_NUM):
            numpy_local_search_to_local[i, 0, 0] = np.cos(THETA_SEARCH[i] / 180 * np.pi)
            numpy_local_search_to_local[i, 2, 2] = np.cos(THETA_SEARCH[i] / 180 * np.pi)
            numpy_local_search_to_local[i, 0, 2] = np.sin(THETA_SEARCH[i] / 180 * np.pi)
            numpy_local_search_to_local[i, 2, 0] = -np.sin(THETA_SEARCH[i] / 180 * np.pi)
        numpy_local_search_to_local = numpy_local_search_to_local @ numpy_frame_move_back
        self.torch_local_search_to_local = torch.tensor(numpy_local_search_to_local, dtype=torch.float,
                                                        device=self.device)
        self.torch_local_to_local_search = torch.inverse(self.torch_local_search_to_local)

    def dump(self, data, name):
        with open(os.path.join(self.output_dir, '{}.pkl'.format(name)), 'wb') as f:
            pickle.dump(data, f)
        print('Dump to file of {} with keys: {}'.format(name, data.keys()))

    def run_loop(self, object_names):
        for _, name in enumerate(object_names):
            print('Begin object {}'.format(name))
            tic = time()
            data_dict = {}
            ply_path = os.path.join(self.model_dir, "{}.obj".format(name))
            mesh = open3d.io.read_triangle_mesh(ply_path)
            mesh.scale(1/1000.0, center=mesh.get_center())
            mesh.compute_vertex_normals()
            pc = mesh.sample_points_uniformly(np.asarray(mesh.vertices).shape[0] * 10)
            print("after sampling", pc)
            pc = pc.voxel_down_sample(POINT_VOXEL_SIZE)
            print("after voxelizing", pc)

            points = np.asarray(pc.points)
            normals = np.asarray(pc.normals)
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            kd_tree = open3d.geometry.KDTreeFlann(pc)
            # normals = self.smooth_normal(normals, points, kd_tree)
            pc.normals = open3d.utility.Vector3dVector(normals)

            data_dict.update({'cloud': points, 'normal': normals})
            
            valid_row, valid_column, all_antipodal_score = self.cache_contact_pair(points, normals)
            print(f"valid_row.shape: {valid_row.shape}, valid_column.shape: {valid_column.shape}, antipodal_score.shape: {all_antipodal_score.shape}")

            frames, local_search_scores, antipodal_score = self._estimate_grasp_quality(points, valid_row, valid_column,
                                                                                        all_antipodal_score)
            frame_neighbor_index = self.get_frame_neighbor(frames, kd_tree)

            data_dict.update(
                {"global_to_local": frames, "search_score": local_search_scores, "antipodal_score": antipodal_score})
            data_dict.update({'frame_point_index': frame_neighbor_index})

            print(f"stored cloud and normal of shape {points.shape} points, {normals.shape} normals")
            print(f"stored global_to_local of shape {format(frames.shape)}")
            print(f"stored local_search_scores of shape {local_search_scores.shape}")
            print(f"stored antipodal_score of shape {antipodal_score.shape}")
            print(f"stored frame_point_index of shape {frame_neighbor_index.shape}")
            print("Finish {} with time: {}s".format(name, time() - tic))
            self.dump(data_dict, name)

    def get_frame_neighbor(self, frames, kd_tree):
        frame_neighbor_index = np.ones(frames.shape[0], dtype=int) * -1
        for j in range(frames.shape[0]):
            local_to_global = np.linalg.inv(frames[j])
            frame_center = local_to_global[0:3, 3]
            [k, idx, _] = kd_tree.search_knn_vector_3d(frame_center, 1)
            frame_neighbor_index[j] = idx[0]
        return frame_neighbor_index

    def smooth_normal(self, normals, points, kd_tree):
        smooth_normals = np.copy(normals)
        for i in range(points.shape[0]):
            single_normal = normals[i, :]
            [k, idx, _] = kd_tree.search_radius_vector_3d(points[i], radius=GASKET_RADIUS)
            if k < 3:
                continue
            neighbors_distance = points[idx, :] - points[i:i + 1, :]
            normal_distance = neighbors_distance @ single_normal.T
            plane_neighbor_bool = normal_distance < 0.001
            average_normal = np.mean(normals[idx, :][plane_neighbor_bool], axis=0)
            smooth_normals[i, :] = average_normal
        smooth_normals /= np.linalg.norm(smooth_normals, axis=1, keepdims=True)
        return smooth_normals

    def cache_contact_pair(self, points, normals):
        print(points.shape)
        #import sys
        #sys.exit()
        dist = sklearn.metrics.pairwise_distances(points)
        within_distance_bool = dist < config.HALF_BOTTOM_SPACE * 2
        distance_vector = -points[:, np.newaxis, :] + points[np.newaxis, :, :]
        distance_vector /= np.clip(np.linalg.norm(distance_vector, axis=2, keepdims=True), 0.0001, 1)
        print(distance_vector.shape)
        pairwise_cos = np.matmul(distance_vector, normals[:, :, np.newaxis]).squeeze(axis=2)
        cos_negative_bool = np.logical_and(pairwise_cos < 0, pairwise_cos.T < 0)
        average_cos = np.abs(pairwise_cos * pairwise_cos.T)
        # average_cos = np.abs(pairwise_cos)
        within_cos_bool = average_cos > COS_THR
        within_bool = np.logical_and(within_cos_bool, within_distance_bool)
        # valid_bool = np.logical_and(within_bool, cos_negative_bool)
        # print('After filtering the positive cos, {}/{} removed'.format(
        #     within_bool.sum() - valid_bool.sum(), within_bool.sum()))

        valid_bool = np.triu(within_bool)
        valid_row, valid_column = np.nonzero(valid_bool)
        antipodal_score = average_cos[valid_row, valid_column]

        return valid_row, valid_column, antipodal_score

    def _estimate_grasp_quality(self, points: np.ndarray, row_index, col_index, antipodal_score):
        torch_points = torch.tensor(points, device=self.device).float()
        torch_points_homo = torch.cat([torch_points.transpose(1, 0),
                                       torch.ones(1, torch_points.shape[0], dtype=torch.float, device=self.device)],
                                      dim=0)

        valid_frame_search_scores = []
        valid_frame = []
        valid_frame_antipodal_scores = []

        # homo_frames = torch.zeros(row_index.shape[0], THETA_NUM, 4, 4, dtype=torch.float, device=self.device)
        assert row_index.shape == col_index.shape
        distance_vector_all = -points[:, np.newaxis, :] + points[np.newaxis, :, :]
        distance_vector_torch = torch.tensor(distance_vector_all, device=self.device, dtype=torch.float)
        init_y_axis = distance_vector_torch[row_index, col_index] / distance_vector_torch[row_index, col_index].norm(
            dim=1, keepdim=True)

        projection_on_distance = (self.zeros_x_direction * init_y_axis).sum(1, keepdim=True)
        init_x_axis = self.zeros_x_direction - projection_on_distance * init_y_axis
        init_x_axis /= init_x_axis.norm(dim=1, keepdim=True)
        init_z_axis = torch.cross(init_x_axis, init_y_axis, dim=1)
        init_frames = torch.zeros(row_index.shape[0], 4, 4, dtype=torch.float, device=self.device)
        init_frames[:, 0:3, 0] = init_x_axis
        init_frames[:, 0:3, 1] = init_y_axis
        init_frames[:, 0:3, 2] = init_z_axis
        init_frames[:, 0:3, 3] = (torch_points[row_index, :] + torch_points[col_index, :]) / 2
        init_frames[:, 3, 3] = torch.tensor(1, dtype=torch.float, device=self.device)
        T_global_to_local = torch.inverse(init_frames)

        for i in trange(row_index.shape[0]):
            row = row_index[i]
            col = col_index[i]
            self.check_collision(torch_points_homo, T_global_to_local[i, :, :], valid_frame_search_scores,
                                 valid_frame, row, col, antipodal_score[i], valid_frame_antipodal_scores)
        assert len(valid_frame_search_scores) == len(valid_frame)
        print('After filtering the collision {} remain'.format(len(valid_frame_antipodal_scores)))

        valid_frame_search_scores = torch.tensor(valid_frame_search_scores)
        valid_frame = torch.stack(valid_frame, dim=0)

        return valid_frame.cpu().numpy(), valid_frame_search_scores.cpu().numpy(), np.array(
            valid_frame_antipodal_scores)

    def check_collision(self, torch_points_homo, T_global_to_local, frame_scores, frame, row, col, antipodal_score,
                        valid_frame_antipodal_scores):
        local_cloud = torch.matmul(T_global_to_local, torch_points_homo)

        close_plane_bool = (local_cloud[1, :] < config.HALF_BOTTOM_SPACE) & \
                           (local_cloud[1, :] > -config.HALF_BOTTOM_SPACE)
        if torch.sum(close_plane_bool) < COLLISION_POINT_THRESHOLD:
            #print('Too few points on the finger plane')
            return

        left_finger_plane_bool = (local_cloud[1, :] < config.HALF_BOTTOM_WIDTH) & \
                                 (local_cloud[1, :] > config.HALF_BOTTOM_SPACE)
        right_finger_plane_bool = (local_cloud[1, :] > -config.HALF_BOTTOM_WIDTH) & \
                                  (local_cloud[1, :] < -config.HALF_BOTTOM_SPACE)
        finger_plane_bool = left_finger_plane_bool | right_finger_plane_bool

        back_plane_bool = finger_plane_bool | close_plane_bool

        local_search_cloud = torch.matmul(self.torch_local_to_local_search, local_cloud.unsqueeze(0))

        for dtheta in range(THETA_NUM):
            darboux_local_search_cloud = local_search_cloud[dtheta, :, :]
            back_collision_bool_x = (darboux_local_search_cloud[0, :] < config.BACK_COLLISION_MARGIN) & \
                                    (darboux_local_search_cloud[0, :] > -config.BOTTOM_LENGTH)
            finger_collision_bool_x = (darboux_local_search_cloud[0, :] > config.BACK_COLLISION_MARGIN) & \
                                      (darboux_local_search_cloud[0, :] < config.FINGER_LENGTH)
            accumulate_search_scores = torch.zeros(1, dtype=torch.float, device=self.device)
            for dw in [-0.015, 0.015, 0]:
                close_region_num = torch.zeros(1, dtype=torch.float, device=self.device)
                z_collision_bool = (darboux_local_search_cloud[2, :] < config.HALF_HAND_THICKNESS + dw) & \
                                   (darboux_local_search_cloud[2, :] > -config.HALF_HAND_THICKNESS + dw)
                back_collision_bool = back_collision_bool_x & z_collision_bool & back_plane_bool

                if torch.sum(back_collision_bool) > config.BACK_COLLISION_THRESHOLD:
                    #print('Back collision')
                    continue

                finger_collision_bool = finger_collision_bool_x & z_collision_bool & finger_plane_bool
                if torch.sum(finger_collision_bool) > config.FINGER_COLLISION_THRESHOLD:
                    #print('Finger collision')
                    continue

                close_region_bool = finger_collision_bool_x & close_plane_bool & z_collision_bool
                close_region_num = torch.sum(close_region_bool, dtype=torch.float)

                if close_region_num < COLLISION_POINT_THRESHOLD:
                    #print('Too few points in the close region')
                    continue

                accumulate_search_scores += close_region_num / 3

            if accumulate_search_scores < COLLISION_POINT_THRESHOLD or close_region_num < COLLISION_POINT_THRESHOLD:
                #print('Too few points in the search region or close region')
                continue

            frame_scores.append(torch.min(accumulate_search_scores, close_region_num))
            final_frame = torch.matmul(self.torch_local_to_local_search[dtheta, :, :], T_global_to_local)
            frame.append(final_frame)
            valid_frame_antipodal_scores.append(antipodal_score)
        return


def main():
    model_dir = "../objects/mesh"
    output_dir = "../objects/single_object_grasp"
    os.makedirs(output_dir, exist_ok=True)
    gen = GenerateContactObjectData(model_dir=model_dir, output_dir=output_dir)
    object_names = ["Lager"]
    gen.run_loop(object_names)


if __name__ == '__main__':
    np.random.seed(1)
    main()
