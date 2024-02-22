import argparse 
import data
from torch.utils.data import DataLoader
from utils import train_one_epoch, evaluate, load_model_from_path, save_model_to_path
import torch
from models import BiGram,NGram,RNN,Config, TransformerModel
from timm.optim.optim_factory import create_optimizer


def get_args_parser():
    parser = argparse.ArgumentParser(
        "AdamMickiewicz training and evaluation", add_help=False)
    
    parser.add_argument("--batch-size", action="store",default=64)
    parser.add_argument("--epochs", action="store", default=10)

    parser.add_argument(
        "--model", choices=['bigram', 'ngram', 'transformer', 'rnn'], default="ngram")
    parser.add_argument("--dropout", action="store", default="0.2")


    parser.add_argument("--opt",default="adam")
    parser.add_argument("--lr", action="store", default=0.001)
    parser.add_argument("--weight-decay",action="store",default=0)
    parser.add_argument("--momentum",action="store",default=0)

    parser.add_argument("--train",action="store_true")
    parser.add_argument("--sample",action="store_true")
    parser.add_argument("--eval",action="store_true")
    
    parser.add_argument("--text",action="store",type=str,default=" ")
    parser.add_argument("--data-path",action="store",default="input.txt")


    parser.add_argument("--device",action="store",default="cpu")
    parser.add_argument("--save-path", action="store", default='')
    
    parser.add_argument("--resume", action="store", default='')
    
    
    parser.add_argument("--embedding-dim", action="store", default=36)
    parser.add_argument("--num-blocks",action="store",default=4)
    parser.add_argument("--seq-len", action="store", default=8)
    parser.add_argument("--n-head", action="store", default=8)
    parser.add_argument("--n-layers", action="store", default=2)
    parser.add_argument("--hidden", action="store", default=32)


    return parser


def main(args):
    
    print(args)
    
    device=torch.device(args.device)
    torch.manual_seed(3213)

    data_obj=data.DataPreparationCharacterLevel(args.data_path)

    train, val, test = data_obj.build_datasets()
    
    train_dataloader = DataLoader(
        data.MickiewiczDataSetChar(train,args.seq_len), batch_size=args.batch_size, drop_last=True)
    val_dataloader = DataLoader(
        data.MickiewiczDataSetChar(val, args.seq_len), batch_size=args.batch_size, drop_last=True)

    test_dataloader = DataLoader(
        data.MickiewiczDataSetChar(test, args.seq_len), batch_size=args.batch_size, drop_last=True)
    
    conf=Config.Config(seq_len=args.seq_len,vocab_size=data_obj.get_vocab_size(),embedding_dim=args.embedding_dim,dropout=args.dropout, batch_size=args.batch_size, n_head=args.n_head,hidden=args.hidden,n_layers=args.n_layers)
    models = {'bigram':BiGram.Bigram, 'ngram':NGram.NGram, 'transformer':TransformerModel.TransformerModel, 'rnn':RNN.RNN}
    model=models[args.model](conf)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = create_optimizer(args, model)

    if args.resume:
        load_model_from_path(model, args.resume)
        
    model=model.to(args.device)
    
    
    if True:
        for epoch in range(args.epochs):
            model.train(True) 
            train_one_epoch(model=model,criterion=criterion,train_loader=train_dataloader,optimizer=optimizer,device=args.device,epoch=epoch)
            model.eval(True)
            evaluate(model=model,data_loader=val_dataloader,device=args.device)
            
    if args.eval:
        model.eval(True)
        evaluate(model=model,data_loader=test_dataloader,device=args.device)
    
    if args.sample:
        model.sample()    
    
    if args.save_path:
        save_model_to_path(model=model,name=args.save_path)
        
        
if __name__=='__main__':
    parser = argparse.ArgumentParser("AdamMickiewicz training and evaluation", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
