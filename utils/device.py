import os
import pdb
import torch
import torch.nn as nn


class GpuDataParallel(object):
    def __init__(self):
        self.gpu_list = []
        self.output_device = None

    def set_device(self, device):
        device = str(device)
        # Check if MPS is available (Apple Silicon)
        if torch.backends.mps.is_available():
            self.gpu_list = []
            output_device = "mps"
        elif device != 'None':
            self.gpu_list = [i for i in range(len(device.split(',')))]
            os.environ["CUDA_VISIBLE_DEVICES"] = device
            output_device = self.gpu_list[0]
            self.occupy_gpu(self.gpu_list)
        else:
            self.gpu_list = []
            output_device = "cpu"

        self.output_device = output_device

    def model_to_device(self, model):
        # model = convert_model(model)
        model = model.to(self.output_device)
        # Only use DataParallel for CUDA devices with multiple GPUs
        if len(self.gpu_list) > 1 and self.output_device != "mps":
            model = nn.DataParallel(
                model,
                device_ids=self.gpu_list,
                output_device=self.output_device)
        return model

    def data_to_device(self, data):
        if isinstance(data, torch.FloatTensor):
            return data.to(self.output_device)
        elif isinstance(data, torch.DoubleTensor):
            return data.float().to(self.output_device)
        elif isinstance(data, torch.ByteTensor):
            return data.long().to(self.output_device)
        elif isinstance(data, torch.LongTensor):
            return data.to(self.output_device)
        elif isinstance(data, list) or isinstance(data, tuple):
            return [self.data_to_device(d) for d in data]
        else:
            raise ValueError(data.shape, "Unknown Dtype: {}".format(data.dtype))

    def criterion_to_device(self, loss):
        return loss.to(self.output_device)

    def occupy_gpu(self, gpus=None):
        """
            make program appear on nvidia-smi or allocate MPS device.
        """
        # Check if MPS is available (Apple Silicon)
        if torch.backends.mps.is_available():
            # For MPS, we don't need to do anything special to occupy the device
            return

        # For CUDA devices
        if len(gpus) == 0:
            torch.zeros(1).cuda()
        else:
            gpus = [gpus] if isinstance(gpus, int) else list(gpus)
            for g in gpus:
                torch.zeros(1).cuda(g)
