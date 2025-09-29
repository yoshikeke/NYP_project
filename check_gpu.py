import torch

# GPUが利用可能かチェック
is_available = torch.cuda.is_available()
print(f"GPU is available: {is_available}")

if is_available:
    # 利用可能なGPUの数を表示
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")

    # 0番目のGPUの名前を表示
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Name: {gpu_name}")
else:
    print("CUDA is not available. Please check your NVIDIA driver and PyTorch installation.")