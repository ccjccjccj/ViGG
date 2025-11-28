import numpy as np
import torch

class VideoDataset(torch.utils.data.Dataset):
    """
        Dataset for video frames. It samples tuples of consecutive frames
    """

    def __init__(self, cfg, data_dict, split):
        self.cfg = cfg
        self.split = split
        self.num_views = cfg.num_views
        self.view_spacing = cfg.view_spacing
        self.data_dict = data_dict

        # The option to do strided frame pairs is to under sample Validation and Test
        # sets since there's a huge number of frames to start with.
        #  An example of strided vs non strided for a view spacing of 10:
        # strided:      (0, 10), (10, 20), (20, 30), etc
        # non-strided:  (0, 10), ( 1, 11), ( 2, 12), etc
        strided = split in ["valid", "test"]
        self.instances = self.dict_to_instances(self.data_dict, strided)

        # Print out dataset stats
        print("================================")
        print(f"Stats for {cfg.name} - {split}")
        print(f"Numer of instances {len(self.instances)}")
        print("Configs:")
        print(cfg)
        print("================================")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        cls_id, s_id, f_ids = self.instances[index]
        s_instance = self.data_dict[cls_id][s_id]["instances"]
        output = {"uid": index, "class_id": cls_id, "sequence_id": s_id}

        # Read in separate instances
        for i, id_i in enumerate(f_ids):
            # transform and save rgb
            output["path_" + str(i)] = s_instance[id_i]["rgb_path"]

        return output

    def dict_to_instances(self, data_dict, strided):
        """
        converts the data dictionary into a list of instances
        Input: data_dict -- sturcture  <classes>/<models>/<instances>

        Output: all dataset instances
        """
        instances = []

        # populate dictionary
        for cls_id in data_dict:
            for s_id in data_dict[cls_id]:
                frames = list(data_dict[cls_id][s_id]["instances"].keys())
                frames.sort()

                if strided:
                    frames = frames[:: self.view_spacing]
                    stride = 1
                else:
                    stride = self.view_spacing

                num_frames = len(frames)

                for i in range(num_frames - self.num_views * stride):
                    f_ids = []
                    for v in range(self.num_views):
                        f_ids.append(frames[i + v * stride])

                    # Hacky way of getting source to be in the middle for triplets
                    mid = self.num_views // 2
                    f_ids = f_ids[mid:] + f_ids[:mid]
                    instances.append([cls_id, s_id, tuple(f_ids)])

        return instances
