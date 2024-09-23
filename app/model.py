import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# 모델 정의 및 클래스 수 맞추기 (10개의 클래스로 설정)
def load_model(model_path: str, device: torch.device):
    model = models.mobilenet_v3_large(weights=None)  # Pretrained 가중치를 사용하지 않음
    model.classifier[3] = torch.nn.Linear(in_features=1280, out_features=10)  # 분류기의 출력 크기 조정
    model = model.to(device)

    # 저장된 가중치 로드
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 평가 모드로 전환
    model.eval()

    return model

# 이미지 전처리 정의
def get_preprocessing():
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess

def predict(model, img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()
