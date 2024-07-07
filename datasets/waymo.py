import os.path as osp
import numpy as np
from pyquaternion import Quaternion
import bisect
import json
from .data_classes import PointCloud, Box
from mmengine.registry import DATASETS


@DATASETS.register_module()
class WaymoDataset:

    def __init__(self, path, category_name="Vehicle", mode='all', **kwargs):
        self.path = path
        self.category_name = category_name
        assert category_name in ['Vehicle', 'Pedestrian']
        bench_dir = osp.join(path, 'benchmark', category_name.lower())
        self.bench = json.load(open(osp.join(bench_dir, 'bench_list.json')))

        def extract_ids_from_bench(bench_name):
            b = json.load(open(osp.join(bench_dir, bench_name)))
            ids = set()
            for tracklet_info in b:
                ids.add(tracklet_info['id'])
            return ids

        self.easy_ids = extract_ids_from_bench('easy.json')
        self.medium_ids = extract_ids_from_bench('medium.json')
        self.hard_ids = extract_ids_from_bench('hard.json')

        assert mode.lower() in ['all', 'easy', 'medium', 'hard']
        if mode.lower() != 'all':
            bench = []
            for i in range(len(self.bench)):
                tracklet_info = self.bench[i]
                t_id = tracklet_info['id']
                if mode.lower() == 'easy' and t_id in self.easy_ids:
                    bench.append(self.bench[i])
                elif mode.lower() == 'medium' and t_id in self.medium_ids:
                    bench.append(self.bench[i])
                elif mode.lower() == 'hard' and t_id in self.hard_ids:
                    bench.append(self.bench[i])
            self.bench = bench

        self.tracklet_num_frames = []
        for tracklet_index, tracklet_info in enumerate(self.bench):
            frame_range = tracklet_info['frame_range']
            self.tracklet_num_frames.append(frame_range[1] - frame_range[0] + 1)

        last_ed_frame_id = 0
        self.tracklet_anno_list = []
        for num_frames in self.tracklet_num_frames:
            self.tracklet_anno_list.append(range(last_ed_frame_id, last_ed_frame_id + num_frames))
            last_ed_frame_id += num_frames

        self.pcds = None
        self.gt_infos = None
        self.cache_tracklet_id = None
        self.mode = None

    def get_num_tracklets(self):
        return len(self.bench)

    def get_num_tracklet_frames(self, tracklet_id):
        return self.tracklet_num_frames[tracklet_id]

    def get_frames(self, tracklet_id, frame_id):
        frames = []
        for i in frame_id:
            frames.append(self.get_frame(tracklet_id, i))
        return frames

    def get_frame(self, tracklet_id, frame_id):
        tracklet_info = self.bench[tracklet_id]
        t_id = tracklet_info['id']
        if t_id in self.easy_ids:
            self.mode = 'easy'
        elif t_id in self.medium_ids:
            self.mode = 'medium'
        elif t_id in self.hard_ids:
            self.mode = 'hard'
        segment_name = tracklet_info['segment_name']
        frame_range = tracklet_info['frame_range']
        if tracklet_id != self.cache_tracklet_id:
            self.cache_tracklet_id = tracklet_id
            if self.gt_infos:
                del self.gt_infos
            if self.pcds:
                del self.pcds
            self.gt_infos = np.load(osp.join(self.path, 'gt_info', '{:}.npz'.format(
                segment_name)), allow_pickle=True)
            self.pcds = np.load(osp.join(self.path, 'pc', 'raw_pc', '{:}.npz'.format(
                segment_name)), allow_pickle=True)
        return self._build_frame(frame_range, t_id, frame_id)

    def get_comp_template_pcd(self, tracklet_id):
        raise NotImplementedError()

    def get_tracklet_frame_id(self, idx):
        tracklet_id = bisect.bisect_right(
            self.tracklet_ed_frame_id, idx)
        assert self.tracklet_st_frame_id[
            tracklet_id] <= idx and idx < self.tracklet_ed_frame_id[tracklet_id]
        frame_id = idx - \
            self.tracklet_st_frame_id[tracklet_id]
        return tracklet_id, frame_id

    def _build_frame(self, frame_range, t_id, frame_id):
        idx = frame_range[0] + frame_id
        pcd = PointCloud(self.pcds[str(idx)].T)

        frame_bboxes = self.gt_infos['bboxes'][idx]
        frame_ids = self.gt_infos['ids'][idx]
        index = frame_ids.index(t_id)
        bbox = frame_bboxes[index]

        center = [bbox[0], bbox[1], bbox[2]]
        size = [bbox[5], bbox[4], bbox[6]]
        orientation = Quaternion(axis=[0, 0, 1], angle=bbox[3])

        bbox = Box(center, size, orientation)

        return {'pc': pcd, '3d_bbox': bbox, 'mode': self.mode}


def print_np(**kwargs):
    for k, v in kwargs.items():
        print(k, np.concatenate((v[:5], v[-5:]), axis=0))
