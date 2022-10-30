from multiprocessing.managers import BaseManager
from time import sleep

import torch
import torch.multiprocessing as mp

from loggers import BasicLogger
from mapping import Mapping
from share import ShareData, ShareDataProxy
from tracking import Tracking
from utils.import_util import get_dataset
from visualization import Visualizer


class VoxSLAM:
    def __init__(self, args):
        self.args = args
        
        # logger (optional)
        self.logger = BasicLogger(args)
        # visualizer (optional)
        self.visualizer = Visualizer(args, self)

        # shared data 
        mp.set_start_method('spawn', force=True)
        BaseManager.register('ShareData', ShareData, ShareDataProxy)
        manager = BaseManager()
        manager.start()
        self.share_data = manager.ShareData()
        # keyframe buffer 
        self.kf_buffer = mp.Queue(maxsize=1)
        # data stream
        self.data_stream = get_dataset(args)
        # tracker 
        self.tracker = Tracking(args, self.data_stream, self.logger, self.visualizer)
        # mapper
        self.mapper = Mapping(args, self.logger, self.visualizer)
        # initialize map with first frame
        self.tracker.process_first_frame(self.kf_buffer)
        self.processes = []

    def start(self):
        mapping_process = mp.Process(
            target=self.mapper.spin, args=(self.share_data, self.kf_buffer))
        mapping_process.start()
        print("initializing the first frame ...")
        sleep(5)
        tracking_process = mp.Process(
            target=self.tracker.spin, args=(self.share_data, self.kf_buffer))
        tracking_process.start()

        vis_process = mp.Process(
            target=self.visualizer.spin, args=(self.share_data,))
        self.processes = [tracking_process, mapping_process]

        if self.args.enable_vis:
            vis_process.start()
            self.processes += [vis_process]

    def wait_child_processes(self):
        for p in self.processes:
            p.join()

    @torch.no_grad()
    def get_raw_trajectory(self):
        return self.share_data.tracking_trajectory

    @torch.no_grad()
    def get_keyframe_poses(self):
        keyframe_graph = self.mapper.keyframe_graph
        poses = []
        for keyframe in keyframe_graph:
            poses.append(keyframe.get_pose().detach().cpu().numpy())
        return poses
