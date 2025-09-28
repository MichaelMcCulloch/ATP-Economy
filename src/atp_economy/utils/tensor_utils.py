# src/atp_economy/utils/tensor_utils.py
import torch

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
