import pandas as pd
import json
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

test_image_dir = 'Dataset/test'
train_image_dir = 'Dataset/train'
answer_path = 'Dataset/answer.json'
train_path = 'Dataset/train.csv'
test_path = 'Dataset/test.csv'

with open(answer_path, 'r', encoding = 'utf8') as f:
    answer_space = json.load(f)

class VQADataset(Dataset):
    def __init__(self, image_dir, annot_path, answer_space, transform):
        super().__init__()
        self.image_dir = image_dir
        self.annot_path = annot_path
        self.df = pd.read_csv(self.annot_path)
        self.trans = transform
        self.answer_space = answer_space
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        question = self.df.iloc[idx]['question']
        answer = self.df.iloc[idx]['answer']
        label = self.answer_space[answer]
        img_id = self.df.iloc[idx]['img_id']
        img_path = self.image_dir + '/' + str(img_id) + '.jpg'
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            return None, None, None
        if self.trans:
            image = self.trans(image)
        
        return image, question, label
    
if __name__ == '__main__':
    transform = T.Compose([T.Resize((224, 224)),T.ToTensor()])
    train_dataset = VQADataset(train_image_dir, train_path, answer_space, transform)
    test_dataset = VQADataset(test_image_dir, test_path, answer_space, transform)
    train_loader = DataLoader(dataset = train_dataset, batch_size = 32, shuffle = True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = 32, shuffle = False)
    for idx, data in enumerate(train_loader):
        image, question, label = data
        break
    print(image.size())