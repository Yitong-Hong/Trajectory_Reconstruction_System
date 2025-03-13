import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .model import Net

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        


    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            if im is None:
                print("resize 遇到空图像！")
            # print("before resize, im.shape:",im.shape)
            imm=im
            # print("after resize, im.shape:",cv2.resize(imm.astype(np.float32)/255., size).shape)
            return cv2.resize(im.astype(np.float32)/255., size)

        # 调试输出 im_crops 的长度和内容，查看是否为空或包含无效数据
        # print(f"裁剪图像数量: {len(im_crops)}")
        for idx, im in enumerate(im_crops):
            if im is None or im.size == 0:

                print(f"警告：图像 {idx} 是空的或无效的！")
                # 打印图像的相关信息：shape 和 dtype
                if im is None:
                    print(f"图像 {idx} 的类型是 None！")
                else:
                    print(f"图像 {idx} 的尺寸: {im.shape}, 数据类型: {im.dtype}")
                    #打印图像信息


        try:
            im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        except Exception as e:
            print(f"处理图像裁剪时出现错误：{e}")
            return torch.empty(0)  # 返回空张量，避免后续错误
    # im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("train.jpg")[:,:,(2,1,0)]
    extr = Extractor("checkpoint/ckpt.t7")
    print(img.shape)
    feature = extr(img)
    print(feature.shape)

