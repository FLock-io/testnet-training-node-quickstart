from torch.cuda import get_device_name


def get_gpu_type():
    try:
        gpu_name = get_device_name(0)
        return gpu_name
    except Exception as e:
        return f"Error retrieving GPU type: {e}"
