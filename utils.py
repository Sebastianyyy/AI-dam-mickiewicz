import torch
import numpy as np
def train_one_epoch(model,criterion,train_loader,optimizer,device,epoch):
    running_loss = 0.
    model.train(True)
    
    for i, data in enumerate(train_loader):
        inputs,labels=data
        
        inputs=inputs.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)[:, -1, :]
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        running_loss += loss.item()
        
        if i % 300 == 299:
            last_loss = running_loss/300  # loss per batch
            print(f'Epoch {epoch} batch {i+1} loss: {last_loss}')
            running_loss = 0.


@torch.no_grad()
def evaluate(model,data_loader,device):
    model.eval(True)
    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0.
    cumulative_loss=[]
    for i, data in enumerate(data_loader):
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)[:, -1, :]

        loss = criterion(outputs, labels)

        running_loss += loss.item()
        cumulative_loss.append(loss.item())
        if i % 300 == 299:
            last_loss = running_loss/300
            print(f'Validation loss in batch {i+1} loss: {last_loss}')
            running_loss = 0.
    print(f'Validation loss total: {np.mean(cumulative_loss)}')
