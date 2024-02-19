import argparse 
import data
from torch.utils.data import DataLoader

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

args = parser.parse_args()


data_obj=data.DataPreparation("input.txt")

train, val, test = data_obj.build_datasets()

val=data.MickiewiczDataSet(val,4)

train_dataloader = DataLoader(val, batch_size=4)




