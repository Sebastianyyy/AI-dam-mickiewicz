import argparse 
import data
from torch.utils.data import DataLoader
from utils import train_one_epoch, evaluate
import torch


parser=argparse.ArgumentParser()
parser.add_argument("--train",action="store_true")
parser.add_argument("--sample",action="store_true")
parser.add_argument("--text",action="store",type=str,default=" ")
parser.add_argument("--model", choices=['bigram', 'ngram', 'mamba','transformer','lstm','gru','rnn'],default="bigram")
parser.add_argument("--batch-size", action="store",default=64)
parser.add_argument("--device",action="store",default="cuda")
parser.add_argument("--data-path",action="store",default="pan_tadeusz.txt")
parser.add_argument("--save-path", action="store", default="model.pt")
parser.add_argument("--learning-rate",action="store",default=0.001)
parser.add_argument("--embedding-dim", action="store", default="36")
parser.add_argument("--seq-len", action="store", default="8")
parser.add_argument("--dropout",action="store",default="0.2")
parser.add_argument("--num-blocks",action="store",default=4)
parser.add_argument("--epoch",action="store",default=10)


args = parser.parse_args()


data_obj=data.DataPreparation("input.txt")

train, val, test = data_obj.build_datasets()

val=data.MickiewiczDataSet(val,4)

train_dataloader = DataLoader(train, batch_size=args.batch_size)

val_dataloader = DataLoader(val, batch_size=args.batch_size)

test_dataloader = DataLoader(test, batch_size=args.batch_size)
model=None
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(args.epoch):
    model.train(True)
    
    train_one_epoch(model=model,criterion=criterion,train_loader=train_dataloader,optimizer=optimizer,device=args.device,epoch=epoch)
    
    model.eval()
    
    evaluate(model=model,data_loader=val_dataloader,device=args.device)
    