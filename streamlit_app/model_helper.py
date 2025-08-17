import torch
from torch import nn
from transformers import AutoImageProcessor, Swinv2ForImageClassification
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class SwinV2Classifier:
    def __init__(self, model_path, class_names):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        
        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-small-patch4-window16-256")
        self.model = Swinv2ForImageClassification.from_pretrained(
            "microsoft/swinv2-small-patch4-window16-256",
            num_labels=len(class_names),
            ignore_mismatched_sizes=True
        )
        
        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        
        # Basic transform for inference
        self.transform = transforms.Compose([
            transforms.Resize(self.processor.size["height"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.processor.image_mean,
                std=self.processor.image_std
            )
        ])
    
    def predict(self, image_path):
        """Make prediction on single image"""
        image = Image.open(image_path).convert("RGB")
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            conf, pred = torch.max(probs, 1)
        
        return {
            "class": self.class_names[pred.item()],
            "confidence": conf.item(),
            "probabilities": {name: float(prob) for name, prob in zip(self.class_names, probs[0])}
        }
    def debug_prediction(self, image_path):
        """Debug prediction with visualization"""
        result = self.predict(image_path)
        image = Image.open(image_path)
        
        # Plotting
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Input Image")
        
        plt.subplot(1, 2, 2)
        plt.barh(self.class_names, list(result["probabilities"].values()))
        plt.title(f"Prediction: {result['class']} ({result['confidence']:.2%})")
        plt.tight_layout()
        
        return plt

