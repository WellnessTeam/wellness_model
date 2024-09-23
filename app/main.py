from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import boto3
from urllib.parse import urlparse
import botocore
import torch
import logging  
from model import load_model, get_preprocessing, predict

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = FastAPI()

# 로깅 설정
logger = logging.getLogger(__name__)

# s3 클라이언트 설정
s3_client = boto3.client('s3')

# device 설정 (CUDA 사용 가능 여부 확인)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 및 전처리 함수 로드
model = load_model('KJSmodelTest_0921.pth', device)
preprocess = get_preprocessing()

# S3에서 이미지를 가져오는 함수
def get_image_from_s3(s3_url: str):
    try:
        parsed_url = urlparse(s3_url)

        # s3:// 형식일 경우
        if parsed_url.scheme == "s3":
            bucket_name = parsed_url.netloc
            key = parsed_url.path.lstrip('/')

        # https:// 형식일 경우
        elif parsed_url.scheme == "https":
            bucket_name = parsed_url.netloc.split('.')[0]
            key = parsed_url.path.lstrip('/')

        else:
            raise ValueError("Invalid URL scheme. Only 's3://' and 'https://' are supported.")

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
        predicted = predict(model, img_tensor)

        return {"category_id": predicted}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
