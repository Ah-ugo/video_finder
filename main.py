from fastapi import FastAPI, UploadFile, HTTPException
from pymongo import MongoClient
import cloudinary
import cloudinary.uploader
import cv2
import numpy as np
import torch
import faiss
from torchvision import models, transforms
from io import BytesIO
import requests
from PIL import Image
from dotenv import load_dotenv
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

app = FastAPI()

url = os.getenv("MONGO_URL")
client = MongoClient(url)
db = client["videos_db"]
videos_collection = db["videos"]

# Configure  Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("API_KEY"),
    api_secret=os.getenv("API_SECRET")
)

# Load pre-trained ResNet model for feature extraction
model = models.resnet50(pretrained=True)
model.eval()

# Define image transformations for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Faiss index for similarity search
d = 1000  # Feature vector dimension
faiss_index = faiss.IndexFlatL2(d)  # L2 similarity

def upload_to_cloudinary(file: UploadFile):
    """Upload file to Cloudinary and return the URL."""
    result = cloudinary.uploader.upload(file.file, resource_type="video")
    return result['url']

def extract_frames(video_path, interval=30):
    """Extract frames at the given interval (in seconds) from video."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        for _ in range(frame_interval - 1):
            cap.grab()
    cap.release()
    return frames

def image_to_feature_vector(image):
    """Convert an image to a feature vector using ResNet."""
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image).squeeze().numpy()  # Convert to numpy array
    return features


@app.post("/upload-video/")
async def upload_video(file: UploadFile):

    try:
        video_url = upload_to_cloudinary(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cloudinary error: {str(e)}")

    # Extract frames and features
    frames = extract_frames(video_url)

    video_id = videos_collection.insert_one({"video_url": video_url, "features": []}).inserted_id
    video_features = []

    for frame in frames:
        features = image_to_feature_vector(frame)
        video_features.append(features)
        faiss_index.add(np.array([features]))  # Add to Faiss index


    video_features_list = [features.tolist() for features in video_features]


    videos_collection.update_one({"_id": video_id}, {"$set": {"features": video_features_list}})

    return {"status": "Video uploaded and processed", "video_url": video_url}



@app.post("/match-image/")
async def match_image(image_url: str):
    # Download image and convert to feature vector
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image = np.array(image)
    image_feature = image_to_feature_vector(image).reshape(1, -1)

    # Search for the closest match in Faiss
    _, indices = faiss_index.search(image_feature, k=1)

    # Convert the NumPy array to a list before querying MongoDB
    matched_video = videos_collection.find_one({"features": indices[0].tolist()})

    if matched_video:
        return {"video_url": matched_video["video_url"]}
    else:
        return {"status": "No match found"}
