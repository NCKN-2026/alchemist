# Cần cài: pip install transformers torch pillow
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

class VLMExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        print("Loading BLIP model...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

    def extract(self, image_path):
        raw_image = Image.open(image_path).convert('RGB')
        # Upscale nếu ảnh quá nhỏ (quan trọng cho CIFAR)
        if raw_image.size[0] < 224:
            raw_image = raw_image.resize((224, 224))
            
        inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        # Biến caption thành dạng giả triplet để Miner dễ học
        # VD: "a frog sitting on a leaf" -> ["object_frog", "context_leaf", "action_sitting"]
        # Ở đây làm đơn giản bằng cách tách từ
        tokens = set(caption.lower().split())
        return list(tokens)