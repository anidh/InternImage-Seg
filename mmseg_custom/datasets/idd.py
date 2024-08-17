# import os.path as osp
# import mmengine.fileio as fileio

# from mmseg.registry import DATASETS
# from .basesegdataset import BaseSegDataset
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset



@DATASETS.register_module()
class IDDDataset(CustomDataset):

    CLASSES = ("road","drivable fallback","sidewalk","non drivable fallback", "person", "rider",
    "motorcycle","bicycle","autorickshaw", "car", "truck", "bus", "vehicle fallback", "curb",
     "wall", "fence", "guard rail","billboard", "traffic sign", "traffic light", "pole",
      "obs-str-bar-fallback", "building", "bridge", "vegetation","sky"),

    PALETTE = [[128, 64, 128], [250, 170, 160], [81, 0, 81], [244, 35, 232], [230, 150, 140], [152, 251, 152], [220, 20, 60],
            [246, 198, 145], [255, 0, 0], [0, 0, 230], [119, 11, 32], [255, 204, 54], [0, 0, 142], [0, 0, 70], [0, 60, 100],
            [0, 0, 90], [0, 0, 110], [0, 80, 100], [136, 143, 153], [220, 190, 40], [102, 102, 156], [190, 153, 153], [180, 165, 180],
            [174, 64, 67], [220, 220, 0], [250, 170, 30]]

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)