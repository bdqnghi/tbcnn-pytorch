import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim

import pickle
from utils.data.dataset import MonoLanguageProgramData
from utils.data.dataloader import ProgramDataloader
from utils.data.dataset import my_collate
# from tensorboardX import SummaryWriter
from utils.model import TBCNN
from utils.run_tbcnn import train
from utils.run_tbcnn import validate
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--train_batch_size', type=int, default=7, help='train batch size')
parser.add_argument('--test_batch_size', type=int, default=32, help='test batch size')
parser.add_argument('--val_batch_size', type=int, default=32, help='val batch size')
parser.add_argument('--state_dim', type=int, default=5, help='GGNN hidden state size')
parser.add_argument('--n_steps', type=int, default=10, help='propogation steps number of GGNN')
parser.add_argument('--niter', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--verbal', type=bool, default=True, help='print training info or not')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--n_classes', type=int, default=104, help='manual seed')
parser.add_argument('--train_directory', default="ProgramData_pkl_train_test_val/train", help='train program data')
parser.add_argument('--test_directory', default="ProgramData_pkl_train_test_val/test", help='test program data')
parser.add_argument('--val_directory', default="ProgramData_pkl_train_test_val/val", help='validation program data')
parser.add_argument('--model_path', default="model/model.ckpt", help='path to save the model')
parser.add_argument('--n_hidden', type=int, default=50, help='number of hidden layers')
parser.add_argument('--size_vocabulary', type=int, default=59, help='maximum number of node types')
parser.add_argument('--is_training_ggnn', type=bool, default=True, help='Training GGNN or BiGGNN')
parser.add_argument('--training', action="store_true",help='is training')
parser.add_argument('--testing', action="store_true",help='is testing')
parser.add_argument('--training_percentage', type=float, default=1.0 ,help='percentage of data use for training')
parser.add_argument('--log_path', default="" ,help='log path for tensorboard')
parser.add_argument('--epoch', type=int, default=0, help='epoch to test')
parser.add_argument('--embeddings', default="embedding/fast_pretrained_vectors.pkl", help='pretrained embeddings url, there are 2 objects in this file, the first object is the embedding matrix, the other is the lookup dictionary')

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if not os.path.exists("model"):
    os.makedirs("model")

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

# This part is the implementation to illustrate Graph-Level output from program data
def main(opt):
    with open(opt.embeddings, 'rb') as fh:
        embeddings, embed_lookup = pickle.load(fh,encoding='latin1')
        num_feats = len(embeddings[0])

    opt.num_features = num_feats
    opt.output_size = 100

    train_dataset = MonoLanguageProgramData(opt.train_directory, embeddings, embed_lookup, num_feats, opt.n_classes)
    train_dataloader = ProgramDataloader(train_dataset, batch_size=opt.train_batch_size, collate_fn= my_collate, shuffle=True, num_workers=0)

    # test_dataset = MonoLanguageProgramData(opt.test_directory, embeddings, embed_lookup, num_feats, opt.n_classes)
    # test_dataloader = ProgramDataloader(test_dataset, batch_size=opt.test_batch_size, collate_fn= my_collate, shuffle=True, num_workers=0)
    
    val_dataset = MonoLanguageProgramData(opt.val_directory, embeddings, embed_lookup, num_feats, opt.n_classes)
    val_dataloader = ProgramDataloader(val_dataset, batch_size=opt.val_batch_size, collate_fn= my_collate, shuffle=True, num_workers=0)
    

    epoch = opt.epoch
    if os.path.exists(opt.model_path):
        print("Using saved model....")
        net = torch.load(opt.model_path)
    else:
        net = TBCNN(opt)
    net.double()

    criterion = nn.CrossEntropyLoss()

    if opt.cuda:
        net.cuda()
        criterion.cuda()

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    for epoch in range(epoch+1, epoch + opt.niter):
        train(epoch, train_dataloader, net, criterion, optimizer, opt)
        validate(epoch, val_dataloader, net, criterion, optimizer, opt)
        # train(epoch, train_dataloader, net, criterion, optimizer, opt, writer)
   

if __name__ == "__main__":
    main(opt)

