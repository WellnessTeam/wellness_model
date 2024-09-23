# model_api.py
from fastapi import FastAPI, HTTPException
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import torch
import boto3
from io import BytesIO
import os
import botocore
from urllib.parse import urlparse
import logging  
import torchvision.models as models


os.environ['KMP_DUPLICATE_LIB_OK']='True'

app = FastAPI()

# 로깅 설정
logger = logging.getLogger(__name__)

# s3 클라이언트 설정
s3_client = boto3.client('s3')

# 모델 정의 및 클래스 수 맞추기 (8개의 클래스로 설정)
model = models.mobilenet_v3_large(weights=None)  # Pretrained 가중치를 사용하지 않음
model.classifier[3] = torch.nn.Linear(in_features=1280, out_features=10)  # 분류기의 출력 크기 조정

# device 설정 (CUDA 사용 가능 여부 확인)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델을 장치에 할당
model = model.to(device)

# 저장된 가중치 로드
best_model_path = 'KJSmodelTest_0921.pth'
model.load_state_dict(torch.load(best_model_path, map_location=device))

# 평가 모드로 전환
model.eval()

# 이미지 전처리 정의
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# S3 URI를 읽고 모델 분류

# def get_image_from_s3(s3_url: str):
#     try:
#         # s3://bucket_name/key 형식의 URL을 분해
#         bucket_name, key = s3_url.replace("s3://", "").split("/", 1)
        
#         # S3에서 객체 다운로드
#         response = s3_client.get_object(Bucket=bucket_name, Key=key)
#         image_data = response['Body'].read()
        
#         # 이미지 로드
#         img = Image.open(BytesIO(image_data))
#         return img
    
#     except botocore.exceptions.ClientError as e:
#         error_code = e.response['Error']['Code']
#         if error_code == "NoSuchKey":
#             raise HTTPException(status_code=404, detail="Image not found in S3 bucket.")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error fetching image from S3: {str(e)}")

def get_image_from_s3(s3_url: str):
    try:
        # URL 파싱 (s3:// 또는 https:// 모두 처리)
        parsed_url = urlparse(s3_url)
        
        # s3:// 형식일 경우
        if parsed_url.scheme == "s3":
            bucket_name = parsed_url.netloc
            key = parsed_url.path.lstrip('/')
        
        # https:// 형식일 경우
        elif parsed_url.scheme == "https":
            # 버킷 이름과 키 추출
            bucket_name = parsed_url.netloc.split('.')[0]  # 버킷 이름 추출
            key = parsed_url.path.lstrip('/')
        
        else:
            raise ValueError("Invalid URL scheme. Only 's3://' and 'https://' are supported.")
        
        # S3에서 객체 다운로드
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        image_data = response['Body'].read()
        
        # 이미지 로드
        try:
            img = Image.open(BytesIO(image_data))
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image format.")
        
        return img
    
    except botocore.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == "NoSuchKey":
            raise HTTPException(status_code=404, detail="Image not found in S3 bucket.")
        else:
            raise HTTPException(status_code=500, detail=f"Error fetching image from S3: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
@app.post("/predict_url/")
async def predict_url(image_url: str):
    try:
        # S3에서 이미지 가져오기
        img = get_image_from_s3(image_url)

        # 이미지를 RGB 형식으로 변환
        img = img.convert('RGB')

        # 전처리 적용
        img_tensor = preprocess(img).unsqueeze(0)

        # 모델 예측
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return {"category_id": predicted.item()}
    
    except Exception as e:
        # 다른 모든 예외 처리
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
