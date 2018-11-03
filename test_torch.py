import torch


def gather_nd(params, indices, name=None):
    '''
    the input indices must be a 2d tensor in the form of [[a,b,..,c],...], 
    which represents the location of the elements.
    '''
    
    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1
    
    for i in range(ndim)[::-1]:
        idx += indices[i] * m 
        m *= params.size(i)
    
    return torch.take(params, idx)

# (15, 720, 30)
# (15, 720, 19, 2)
# (15, 720, 19, 30)

x = torch.randn(15, 720,30)
# print(x)
# print(x.shape)
# indices = torch.ByteTensor([[0, 0],[1,1]])

# indices2 = torch.LongTensor([15], [720])
# print(x.index_select(0, indices2))

# print(x.masked_select(indices))


# x = torch.randn(2, 2)
# print(x.shape)
# print(x)
# indices = torch.ByteTensor([0,1])
# indices = torch.ByteTensor([[0],[1]])
# print(x.masked_select(indices))


params = torch.randn(15, 200, 30)
# indices  = torch.randn(15, 200, 19, 2)
print(params.shape)
indices = torch.LongTensor([155,200, 15])
# print(x.index_select(0, indices))
# print(x.index_select(0, indices).shape)

x = torch.randn(1, 4, 2)
print(x[:,:,1].shape)

# x = torch.randn(2, 2)
# print(x)
# indices = torch.LongTensor([[0, 0],[1,1]])
# print([x[i] for i in indices])