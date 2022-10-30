from copy import deepcopy
import random
from time import sleep
import numpy as np

import torch
import trimesh

from criterion import Criterion
from loggers import BasicLogger
from utils.import_util import get_decoder, get_property
from variations.render_helpers import bundle_adjust_frames
from utils.mesh_util import MeshExtractor

torch.classes.load_library(
    "third_party/sparse_octree/build/lib.linux-x86_64-3.8/svo.cpython-38-x86_64-linux-gnu.so")


def get_network_size(net):
    size = 0
    for param in net.parameters():
        size += param.element_size() * param.numel()
    return size / 1024 / 1024


class Mapping:
    def __init__(self, args, logger: BasicLogger, vis=None, **kwargs):
        super().__init__()
        self.args = args
        self.logger = logger
        self.visualizer = vis
        self.decoder = get_decoder(args).cuda()

        self.loss_criteria = Criterion(args)
        self.keyframe_graph = []
        self.initialized = False

        mapper_specs = args.mapper_specs

        # optional args
        self.ckpt_freq = get_property(args, "ckpt_freq", -1)
        self.final_iter = get_property(mapper_specs, "final_iter", 0)
        self.mesh_res = get_property(mapper_specs, "mesh_res", 8)
        self.save_data_freq = get_property(
            args.debug_args, "save_data_freq", 0)

        # required args
        # self.overlap_th = mapper_specs["overlap_th"]
        self.voxel_size = mapper_specs["voxel_size"]
        self.window_size = mapper_specs["window_size"]
        self.num_iterations = mapper_specs["num_iterations"]
        self.n_rays = mapper_specs["N_rays_each"]
        self.sdf_truncation = args.criteria["sdf_truncation"]
        self.max_voxel_hit = mapper_specs["max_voxel_hit"]
        self.step_size = mapper_specs["step_size"]
        self.step_size = self.step_size * self.voxel_size
        self.max_distance = args.data_specs["max_depth"]

        embed_dim = args.decoder_specs["in_dim"]
        use_local_coord = mapper_specs["use_local_coord"]
        self.embed_dim = embed_dim - 3 if use_local_coord else embed_dim
        num_embeddings = mapper_specs["num_embeddings"]
        self.mesh_freq = args.debug_args["mesh_freq"]
        self.mesher = MeshExtractor(args)

        self.embeddings = torch.zeros(
            (num_embeddings, self.embed_dim),
            requires_grad=True, dtype=torch.float32,
            device=torch.device("cuda"))
        torch.nn.init.normal_(self.embeddings, std=0.01)

        self.svo = torch.classes.svo.Octree()
        self.svo.init(256, embed_dim, self.voxel_size)

        self.frame_poses = []
        self.depth_maps = []
        self.last_tracked_frame_id = 0

    def spin(self, share_data, kf_buffer):
        print("mapping process started!")
        while True:
            # torch.cuda.empty_cache()
            if not kf_buffer.empty():
                tracked_frame = kf_buffer.get()
                # self.create_voxels(tracked_frame)

                if not self.initialized:
                    if self.mesher is not None:
                        self.mesher.rays_d = tracked_frame.get_rays()
                    self.create_voxels(tracked_frame)
                    self.insert_keyframe(tracked_frame)
                    while kf_buffer.empty():
                        self.do_mapping(share_data)
                        # self.update_share_data(share_data, tracked_frame.stamp)
                    self.initialized = True
                else:
                    self.do_mapping(share_data, tracked_frame)
                    self.create_voxels(tracked_frame)
                    # if (tracked_frame.stamp - self.current_keyframe.stamp) > 50:
                    if (tracked_frame.stamp - self.current_keyframe.stamp) > 50:
                        self.insert_keyframe(tracked_frame)
                        print(
                            f"********** current num kfs: { len(self.keyframe_graph) } **********")

                # self.create_voxels(tracked_frame)
                tracked_pose = tracked_frame.get_pose().detach()
                ref_pose = self.current_keyframe.get_pose().detach()
                rel_pose = torch.linalg.inv(ref_pose) @ tracked_pose
                self.frame_poses += [(len(self.keyframe_graph) -
                                      1, rel_pose.cpu())]
                self.depth_maps += [tracked_frame.depth.clone().cpu()]

                if self.mesh_freq > 0 and (tracked_frame.stamp + 1) % self.mesh_freq == 0:
                    self.logger.log_mesh(self.extract_mesh(
                        res=self.mesh_res, clean_mesh=True), name=f"mesh_{tracked_frame.stamp:05d}.ply")

                if self.save_data_freq > 0 and (tracked_frame.stamp + 1) % self.save_data_freq == 0:
                    self.save_debug_data(tracked_frame)
            elif share_data.stop_mapping:
                break

        print(f"********** post-processing {self.final_iter} steps **********")
        self.num_iterations = 1
        for iter in range(self.final_iter):
            self.do_mapping(share_data, tracked_frame=None,
                            update_pose=False, update_decoder=False)

        print("******* extracting final mesh *******")
        pose = self.get_updated_poses()
        mesh = self.extract_mesh(res=self.mesh_res, clean_mesh=False)
        self.logger.log_ckpt(self)
        self.logger.log_numpy_data(np.asarray(pose), "frame_poses")
        self.logger.log_mesh(mesh)
        self.logger.log_numpy_data(self.extract_voxels(), "final_voxels")
        print("******* mapping process died *******")

    def do_mapping(self, share_data, tracked_frame=None,
                   update_pose=True, update_decoder=True):
        # self.map.create_voxels(self.keyframe_graph[0])
        self.decoder.train()
        optimize_targets = self.select_optimize_targets(tracked_frame)
        # optimize_targets = [f.cuda() for f in optimize_targets]

        bundle_adjust_frames(
            optimize_targets,
            self.embeddings,
            self.map_states,
            self.decoder,
            self.loss_criteria,
            self.voxel_size,
            self.step_size,
            self.n_rays,
            self.num_iterations,
            self.sdf_truncation,
            self.max_voxel_hit,
            self.max_distance,
            learning_rate=[1e-2, 1e-3],
            update_pose=update_pose,
            update_decoder=update_decoder
        )

        # optimize_targets = [f.cpu() for f in optimize_targets]
        self.update_share_data(share_data)
        # sleep(0.01)

    def select_optimize_targets(self, tracked_frame=None):
        # TODO: better ways
        targets = []
        selection_method = 'random'
        if len(self.keyframe_graph) <= self.window_size:
            targets = self.keyframe_graph[:]
        elif selection_method == 'random':
            targets = random.sample(self.keyframe_graph, self.window_size)
        elif selection_method == 'overlap':
            raise NotImplementedError(
                f"seletion method {selection_method} unknown")

        if tracked_frame is not None and tracked_frame != self.current_keyframe:
            targets += [tracked_frame]
        return targets

    def update_share_data(self, share_data, frameid=None):
        share_data.decoder = deepcopy(self.decoder).cpu()
        tmp_states = {}
        for k, v in self.map_states.items():
            tmp_states[k] = v.detach().cpu()
        share_data.states = tmp_states
        # self.last_tracked_frame_id = frameid

    def insert_keyframe(self, frame):
        # kf check
        print("insert keyframe")
        self.current_keyframe = frame
        self.keyframe_graph += [frame]
        # self.update_grid_features()

    def create_voxels(self, frame):
        points = frame.get_points().cuda()
        pose = frame.get_pose().cuda()
        points = points@pose[:3, :3].transpose(-1, -2) + pose[:3, 3]
        voxels = torch.div(points, self.voxel_size, rounding_mode='floor')

        self.svo.insert(voxels.cpu().int())
        self.update_grid_features()

    @torch.enable_grad()
    def update_grid_features(self):
        voxels, children, features = self.svo.get_centres_and_children()
        centres = (voxels[:, :3] + voxels[:, -1:] / 2) * self.voxel_size
        children = torch.cat([children, voxels[:, -1:]], -1)

        centres = centres.cuda().float()
        children = children.cuda().int()

        map_states = {}
        map_states["voxel_vertex_idx"] = features.cuda()
        map_states["voxel_center_xyz"] = centres
        map_states["voxel_structure"] = children
        map_states["voxel_vertex_emb"] = self.embeddings
        self.map_states = map_states

    @torch.no_grad()
    def get_updated_poses(self):
        frame_poses = []
        for i in range(len(self.frame_poses)):
            ref_frame_ind, rel_pose = self.frame_poses[i]
            ref_frame = self.keyframe_graph[ref_frame_ind]
            ref_pose = ref_frame.get_pose().detach().cpu()
            pose = ref_pose @ rel_pose
            frame_poses += [pose.detach().cpu().numpy()]
        return frame_poses

    @torch.no_grad()
    def extract_mesh(self, res=8, clean_mesh=False):
        sdf_network = self.decoder
        sdf_network.eval()

        voxels, _, features = self.svo.get_centres_and_children()
        index = features.eq(-1).any(-1)
        voxels = voxels[~index, :]
        features = features[~index, :]
        centres = (voxels[:, :3] + voxels[:, -1:] / 2) * self.voxel_size

        encoder_states = {}
        encoder_states["voxel_vertex_idx"] = features.cuda()
        encoder_states["voxel_center_xyz"] = centres.cuda()
        encoder_states["voxel_vertex_emb"] = self.embeddings

        frame_poses = self.get_updated_poses()
        mesh = self.mesher.create_mesh(
            self.decoder, encoder_states, self.voxel_size, voxels,
            frame_poses=frame_poses[-1], depth_maps=self.depth_maps[-1],
            clean_mseh=clean_mesh, require_color=True, offset=-10, res=res)
        return mesh

    @torch.no_grad()
    def extract_voxels(self, offset=-10):
        voxels, _, features = self.svo.get_centres_and_children()
        index = features.eq(-1).any(-1)
        voxels = voxels[~index, :]
        features = features[~index, :]
        voxels = (voxels[:, :3] + voxels[:, -1:] / 2) * \
            self.voxel_size + offset
        print(torch.max(features)-torch.count_nonzero(index))
        return voxels

    @torch.no_grad()
    def save_debug_data(self, tracked_frame, offset=-10):
        """
        save per-frame voxel, mesh and pose 
        """
        pose = tracked_frame.get_pose().detach().cpu().numpy()
        pose[:3, 3] += offset
        frame_poses = self.get_updated_poses()
        mesh = self.extract_mesh(res=8, clean_mesh=True)
        voxels = self.extract_voxels().detach().cpu().numpy()
        keyframe_poses = [p.get_pose().detach().cpu().numpy()
                          for p in self.keyframe_graph]

        for f in frame_poses:
            f[:3, 3] += offset
        for kf in keyframe_poses:
            kf[:3, 3] += offset

        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        color = np.asarray(mesh.vertex_colors)

        self.logger.log_debug_data({
            "pose": pose,
            "updated_poses": frame_poses,
            "mesh": {"verts": verts, "faces": faces, "color": color},
            "voxels": voxels,
            "voxel_size": self.voxel_size,
            "keyframes": keyframe_poses,
            "is_keyframe": (tracked_frame == self.current_keyframe)
        }, tracked_frame.stamp)