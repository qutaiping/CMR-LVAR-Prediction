# type: ignore[no-any-return]
import sys
from os.path import abspath, dirname
from typing import IO, Dict
import os
import numpy as np
import SimpleITK as sitk
import torch
import yaml
import tarfile
import matplotlib.pyplot as plt
import functools

def save_nii(arr, output_path_file):
    im_sitk = sitk.GetImageFromArray(arr)
    sitk.WriteImage(im_sitk, output_path_file)
    return

class ResampledClassificationConfig:
    def __init__(self, network_f, config: Dict):
        self.network_f = network_f
        if self.network_f is not None:
            from mmcv import Config

            if isinstance(self.network_f, str):
                self.network_cfg = Config.fromfile(self.network_f)
            else:
                import tempfile

                with tempfile.TemporaryDirectory() as temp_config_dir:
                    with tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix=".py") as temp_config_file:
                        with open(temp_config_file.name, "wb") as f:
                            f.write(self.network_f.read())

                        self.network_cfg = Config.fromfile(temp_config_file.name)

    def __repr__(self) -> str:
        return str(self.__dict__)


class ResampledClassificationModel:
    def __init__(
        self, model_f: IO, network_f, config_f,
    ):
        self.model_f = model_f
        self.network_f = network_f
        self.config_f = config_f


class ResampledClassificationPredictor:
    def __init__(self, gpu: int, model: ResampledClassificationModel):
        self.gpu = gpu
        self.model = model
        if self.model.config_f is not None:
            if isinstance(self.model.config_f, str):
                with open(self.model.config_f, "r") as config_f:
                    self.config = ResampledClassificationConfig(self.model.network_f, yaml.safe_load(config_f),)
            else:
                self.config = ResampledClassificationConfig(
                    self.model.network_f, yaml.safe_load(self.model.config_f),
                )
        else:
            self.config = None
        self.load_model()

    @classmethod
    def build_predictor_from_tar(cls, tar: tarfile.TarFile, gpu: int):
        files = tar.getnames()

        model_segUrinary_vessel = ResampledClassificationModel(
            model_f=tar.extractfile(tar.getmember("Infar_phase.pt")) 
            if "Infar_phase.pt" in files
            else tar.extractfile(tar.getmember("Infar_phase.pth")),
            network_f=tar.extractfile(tar.getmember("phase_cls_config.py")),
            config_f=tar.extractfile(tar.getmember("cls_phase.yaml")),
        )

        return ResampledClassificationPredictor(gpu=gpu, model=model_segUrinary_vessel)

    def load_model(self) -> None:
        if isinstance(self.model.model_f, str):
            if self.model.model_f.endswith(".pth"):
                self.load_model_pth()
            else:
                self.load_model_jit()
        else:
            try:
                self.load_model_jit()
            except Exception:
                self.load_model_pth()

    def load_model_jit(self) -> None:
        from torch import jit

        if not isinstance(self.model.model_f, str):
            self.model.model_f.seek(0)
        self.net = jit.load(self.model.model_f, map_location=f"cuda:{self.gpu}")
        self.net.cuda(self.gpu)
    
    def load_model_pth(self) -> None:
        from train.custom.model.utils import build_network

        import importlib.util
        import os
        custom_path = os.path.join(dirname(dirname(abspath(__file__))),"train","custom","__init__.py") 
        spec = importlib.util.spec_from_file_location("custom", custom_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        #sys.modules["custom"] = module
        
        config = self.config.network_cfg

        self.net = build_network(config.model, test_cfg=config.test_cfg)

        if not isinstance(self.model.model_f, str):
            self.model.model_f.seek(0)
        checkpoint = torch.load(self.model.model_f, map_location=f"cuda:{self.gpu}")
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.eval()
        self.net.cuda(self.gpu)

        sys.path.pop()
        remove_names = []
        for k in sys.modules.keys():
            if "custom." in k or "custom" == k or "starship.umtf" in k:
                remove_names.append(k)
        for k in remove_names:
            del sys.modules[k]

    def _normalization(self, vol):
        hu_max = torch.max(vol)
        hu_min = torch.min(vol)
        vol_normalized = (vol - hu_min) / (hu_max - hu_min + 1e-8)
        return vol_normalized

    def _crop_data(self, vol, flow, mask, c_t, target_shape, config):
        device = vol.device if hasattr(vol, 'device') else 'cpu'
        vol = torch.as_tensor(vol, device=device)
        mask = torch.as_tensor(mask, device=device)
        c_t = torch.as_tensor(c_t, device=device)
        target_shape = torch.as_tensor(target_shape, device=device)
        
        input_shape = torch.tensor(vol.shape[-3:], device=device)
        
        start = (c_t - target_shape // 2).floor().long()
        end = start + target_shape
        
        output_shape = (*vol.shape[:-3], *target_shape)
        output_shape_flow = (*flow.shape[:-3], *target_shape)
        cropped_vol = torch.zeros(output_shape, dtype=vol.dtype, device=device)
        cropped_mask = torch.zeros(output_shape, dtype=mask.dtype, device=device)
        cropped_flow = torch.zeros(output_shape_flow, dtype=vol.dtype, device=device)
        
        src_start = torch.maximum(start, torch.tensor([0,0,0], device=device))
        src_end = torch.minimum(end, input_shape)
        
        dst_start = (src_start - start).long()
        dst_end = dst_start + (src_end - src_start).long()
        
        cropped_vol[
            ...,
            dst_start[0]:dst_end[0],
            dst_start[1]:dst_end[1],
            dst_start[2]:dst_end[2]
        ] = vol[
            ...,
            src_start[0]:src_end[0],
            src_start[1]:src_end[1],
            src_start[2]:src_end[2]
        ]
        
        cropped_mask[
            ...,
            dst_start[0]:dst_end[0],
            dst_start[1]:dst_end[1],
            dst_start[2]:dst_end[2]
        ] = mask[
            ...,
            src_start[0]:src_end[0],
            src_start[1]:src_end[1],
            src_start[2]:src_end[2]
        ]

        cropped_flow[
            ...,
            dst_start[0]:dst_end[0],
            dst_start[1]:dst_end[1],
            dst_start[2]:dst_end[2]
        ] = flow[
            ...,
            src_start[0]:src_end[0],
            src_start[1]:src_end[1],
            src_start[2]:src_end[2]
        ]
        cropped_flow = cropped_flow.squeeze(0)
        cropped_vol = self._normalization(cropped_vol)
        img = torch.cat([cropped_vol, cropped_flow, cropped_mask], dim=1)
        return img, cropped_flow

    
    def predict(self, hu_volume: np.ndarray, flow: np.ndarray, Infar_mask: np.ndarray):
        flow = abs(flow)
        hu_volume = torch.from_numpy(hu_volume.astype(np.float32))[None, None]
        flow = torch.from_numpy(flow.astype(np.float32))[None, None]

        Infar_pos = np.argwhere(Infar_mask)
        Infar_pos_min = np.min(Infar_pos, axis=0)
        Infar_pos_max = np.max(Infar_pos, axis=0)
        Infar_center = (Infar_pos_min + Infar_pos_max) / 2

        Infar_phy_center = Infar_center

        Infar_mask = torch.from_numpy(Infar_mask.astype(np.float32))[None, None]
        res = self._get_cls_result(hu_volume, flow, Infar_mask, Infar_phy_center)
        return res

    def _get_cls_result(self, hu_volume, flow, Infar_mask, Infar_phy_center: np.ndarray):
        config = self.config.network_cfg

        with torch.no_grad(), torch.cuda.device(self.gpu):
            data = self._get_cls_input(hu_volume, flow, Infar_mask, Infar_phy_center, config)
            data = data.cuda().detach()
            pred = self.net.forward_test(data)
            pred = pred.cpu().detach().numpy()[0][0]

        return pred
    
    

    def _get_cls_input(self, hu_volume, flow, Infar_mask, c_t, config):
        tgt_shape = config.patch_size
        data, _ = self._crop_data(hu_volume, flow, Infar_mask, c_t, tgt_shape, config)
        return data

    def free(self):
        # TODO: add free logic
        if self.net is not None:
            del self.net
        torch.cuda.empty_cache()
