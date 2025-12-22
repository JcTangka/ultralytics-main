# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

print("cuda_available:", torch.cuda.is_available())
print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
