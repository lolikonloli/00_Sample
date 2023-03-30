# ---------------------------------------------------------------------------- #
#                                      导入库                                     #
# ---------------------------------------------------------------------------- #
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VisionDataset
from PIL import Image
import numpy as np
from pathlib2 import Path
import pytest
from loguru import logger
from torchvision import transforms


# ---------------------------------------------------------------------------- #
#                                    自定义数据集                                    #
# ---------------------------------------------------------------------------- #
class CVC_ClinicDB(VisionDataset):

    def __init__(self, root: str, transforms: callable) -> None:
        super().__init__(root, transforms)
        
        
        
        
        self.root_path = Path(root)

    def __getitem__(self, index):
        img_path: str = f'{self.root_path.absolute()}/img/{index+1}.png'
        gt_path: str = f'{self.root_path.absolute()}/gt/{index+1}.png'
        logger.info(f'\nimg path:{img_path}\n, gt path: {gt_path}')

        img = Image.open(img_path)
        mask = Image.open(gt_path)
        if self.transform is not None:
            image = self.transform(image)
        return img, mask

    def __len__(self):
        return 2


# ---------------------------------------------------------------------------- #
#                                      测试                                      #
# ---------------------------------------------------------------------------- #


class TestDataset():

    def test_read_dataset(self):

        root_path = Path.cwd()
        logger.info(f'root_path: {root_path}')

        dataset = CVC_ClinicDB(
            f'/home/lolikonloli/00_AI/00_Sample/96_data/CVC_ClinicDB/train',
            None)
        logger.info(type(dataset[0]))
        logger.info(len(dataset))
    
    def test_transfrom(self):
        ...

if __name__ == '__main__':
    pytest.main(['-s', '-v', '-x', __file__])