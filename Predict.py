from Model import BaseModel
import json
import numpy as np
from PIL import Image
from torchvision import transforms as T
import torch 


checkpoint = torch.load('Checkpoint/checkpoint.pt')
with open('Dataset/answer.json', 'r', encoding = 'utf8') as f:
    answer_space = json.load(f)
swap_space = {v : k for k, v in answer_space.items()}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BaseModel().to(device)
model.load_state_dict(checkpoint['model_state_dict'])

def generate_caption(image, question):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, str):
        image = Image.open(image).convert("RGB")
    transform = T.Compose([T.Resize((224, 224)),T.ToTensor()])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(image, question)
        idx = torch.argmax(logits)
        return swap_space[idx.item()]

if __name__ == "__main__":
    image = 'Dataset/train/68857.jpg'
    question = 'màu của chiếc bình là gì'
    pred = generate_caption(image, question)
    print(pred)