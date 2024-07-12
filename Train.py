import os
import torch
import torch.nn as nn
import torch.nn as nn
from tqdm import tqdm
from Make_dataset import train_loader
from Model import BaseModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BaseModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)
scaler = torch.cuda.amp.GradScaler()
criterion = nn.CrossEntropyLoss()

def training(model = model, criterion = criterion, epochs = 30, 
             optimizer = optimizer, scaler = scaler, train_loader = train_loader,
            checkpoint_path = 'Checkpoint/checkpoint.pt', save_path = 'Checkpoint/last_checkpoint.pt'):
    patience = 3
    training_losses = []
    best_loss = float('inf')
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint.get('best_loss', best_loss)
    
    for epoch in range(epochs):
        model.train()
        batch_train_losses = []
        train_loader_iter = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
        for _, data in enumerate(train_loader_iter):
            image, question, label = data
            out = model(image, question)
            loss = criterion(out, label.to(device))
            train_loader_iter.set_postfix()
            batch_train_losses.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
            train_loader_iter.set_postfix(train_loss=loss.item())
        train_loss = sum(batch_train_losses) / len(batch_train_losses)
        training_losses.append(train_loss)

        
        if train_loss > best_loss:
            patience -= 1
        if patience == 0:
            print(f"Early stopping at epoch {epoch + 1}")
            return 
        
        if train_loss < best_loss:
            best_loss = train_loss
            patience = 3
            torch.save({
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'best_loss' : best_loss
            }, save_path)
            print(f"Model is saved at epoch {epoch + 1} with loss : {best_loss}")
        else:
            print(f"Loss at epoch {epoch + 1} is {train_loss}")


if __name__ == "__main__":
    training()