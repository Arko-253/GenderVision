import torch
import cv2
from torchvision import models, transforms
from PIL import Image

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Model setup
model = models.resnet34(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("gender_classifier_best.pt", map_location=device))
model.eval().to(device)

# ✅ Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ✅ Mapping
idx_to_label = {0: 'Female', 1: 'Male'}

# ✅ Classification function
def classify_gender_resnet(frame, track_id=None, threshold=0.6):
    try:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
        gender = idx_to_label[pred.item()]
        if conf.item() < threshold:
            return "Unknown", conf.item()
        return gender, conf.item()
    except Exception as e:
        print(f"[!] classify_gender_resnet Error: {e}")
        return "Unknown", 0.0
