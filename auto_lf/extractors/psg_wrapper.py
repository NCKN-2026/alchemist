# Cần cài: openmim, mmdet, mmcv-full
try:
    from mmdet.apis import init_detector, inference_detector
except ImportError:
    init_detector = None

class PSGExtractor:
    def __init__(self, config_path, checkpoint_path, device='cuda:0'):
        if init_detector is None:
            raise ImportError("Please install MMDetection for PSG mode")
        print("Loading PSG model...")
        self.model = init_detector(config_path, checkpoint_path, device=device)

    def extract(self, image_path):
        result = inference_detector(self.model, image_path)
        # Parse result của OpenPSG ra danh sách triplet string
        # Code parse chi tiết bạn xem lại câu trả lời trước
        # Return ví dụ: ["worker-wear-helmet", "worker-on-ladder"]
        return []