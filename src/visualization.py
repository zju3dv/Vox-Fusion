import open3d as o3d


class Visualizer:
    def __init__(self, args, voxslam):
        self.args = args
        self.voxslam = voxslam
        self.pose_history = []

    def spin(self, share_data):
        print("visualization process start")
        try:
            engine = o3d.visualization.VisualizerWithKeyCallback()
            engine.create_window(
                window_name="VoxSLAM", width=1280, height=720, visible=True)
            engine.run()
            engine.destroy_window()
        except Exception as e:
            print(f"error creating window {e}")
        finally:
            share_data.stop_mapping = True
            share_data.stop_tracking = True
            print("visualization process died")

    def insert_frame_pose(self, pose):
        self.pose_history += [pose]
