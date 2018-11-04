import torch
from torch.autograd import Variable
from shutil import copyfile
import torch.nn.functional as F

def train(epoch, dataloader, net, criterion, optimizer, opt):
    
    for i, (nodes, children, target) in enumerate(dataloader, 0):
    
        # net.zero_grad()

        # print(nodes)
        # print(nodes.shape)

        if opt.cuda:
            nodes = nodes.cuda()
            children = children.cuda()
            target = target.cuda()

        nodes = Variable(nodes)
        children = Variable(children)
        target = Variable(target)
        output = net(nodes, children)

        if opt.cuda:
            output = output.cuda()
    
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % int(len(dataloader) / 10 + 1) == 0 and opt.verbal:
            print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, opt.niter, i, len(dataloader), loss.item()))

    torch.save(net, opt.model_path)

def validate(epoch, dataloader, net, criterion, optimizer, opt):
    test_loss = 0
    correct = 0
    net.eval()

    for i, (nodes, children, target) in enumerate(dataloader, 0):
    
        net.zero_grad()

        if opt.cuda:
            nodes = nodes.cuda()
            children = children.cuda()
            target = target.cuda()

        nodes = Variable(nodes)
        children = Variable(children)
        target = Variable(target)
        output = net(nodes, children)

        if opt.cuda:
            output = output.cuda()
        
        output = F.softmax(output,1)
        # print(output)
        # print(target)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
  
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()


    test_loss /= len(dataloader.dataset)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))