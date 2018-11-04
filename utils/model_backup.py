import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

from .function_util import tensordot
torch.set_default_tensor_type('torch.cuda.DoubleTensor')

def tile(a, dim, n_tile, is_cuda):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))

    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    if is_cuda:
        order_index = order_index.cuda()
    return torch.index_select(a, dim, order_index)


class ETA_T(nn.Module):
    def __init__(self, opt):
        super(ETA_T, self).__init__()
        self.opt = opt
        self.cuda = False
        if self.opt.cuda:
            self.cuda = True
      
    
    def forward(self, children):
        """
        Compute weight matrix for how much each vector belongs to the 'top'
    
        This part is tricky, this implementation only slide over a window of depth `, which means a child node in a window
        always has depth = 1, according to the formula in the original paper, top-coefficient in this case is alwasy 0/1 = 1
        """

        batch_size = children.shape[0]
        max_tree_size = children.shape[1]
        max_children = children.shape[2]


        # eta_t is shape (batch_size x max_tree_size x max_children + 1)
        eta = torch.cat((torch.ones((max_tree_size, 1)),torch.zeros((max_tree_size, max_children))),1)
        eta = eta.unsqueeze(0)
        eta = tile(eta, 0, batch_size, self.cuda)
        return eta

class ETA_R(nn.Module):
    def __init__(self, opt):
        super(ETA_R, self).__init__()
        self.opt = opt
        self.cuda = False
        if self.opt.cuda:
            self.cuda = True
  

    def forward(self, children, coef_t):
        """Compute weight matrix for how much each vector belogs to the 'right'"""
        # children is batch_size x max_tree_size x max_children
        batch_size = children.shape[0]
        max_tree_size = children.shape[1]
        max_children = children.shape[2]

        # num_siblings is shape (batch_size x max_tree_size x 1)
        num_siblings = max_children - (children == 0).sum(dim=2,keepdim=True)
      
        # num_siblings is shape (batch_size x max_tree_size x max_children + 1)
        num_siblings = tile(num_siblings, 2, max_children + 1, self.cuda).double()

        # creates a mask of 1's and 0's where 1 means there is a child there
        # has shape (batch_size x max_tree_size x max_children + 1)
        mask = torch.cat((torch.zeros((batch_size, max_tree_size, 1)).double(),torch.min(children,torch.ones((batch_size, max_tree_size, max_children)).double())),2)
        
        # child indices for every tree (batch_size x max_tree_size x max_children + 1)
        child_indices = torch.arange(-1.0, float(max_children) , 1.0).double()
        child_indices = child_indices.unsqueeze(0)
        child_indices = child_indices.unsqueeze(0)
        child_indices = tile(child_indices, 0 , batch_size, self.cuda)
        child_indices = tile(child_indices, 1, max_tree_size, self.cuda)
        child_indices = torch.mul(child_indices,mask)

        # weights for every tree node in the case that num_siblings = 0
        # shape is (batch_size x max_tree_size x max_children + 1)
        singles = torch.cat((torch.zeros((batch_size, max_tree_size,1)).double(), torch.tensor((),dtype=torch.double).new_full((batch_size, max_tree_size, 1),0.5), torch.zeros((batch_size, max_tree_size, max_children - 1)).double()),2)

        # eta_r is shape (batch_size x max_tree_size x max_children + 1)
        result = torch.where(
            torch.eq(num_siblings,torch.ones((batch_size, max_tree_size, max_children + 1)).double()),
            singles,
            torch.mul((1.0 - coef_t).double(),torch.div(child_indices,num_siblings-1.0).double())
        )
        return result

class ETA_L(nn.Module):
    def __init__(self, opt):
        super(ETA_L, self).__init__()
        self.opt = opt
        self.cuda = False
        if self.opt.cuda:
            self.cuda = True
      
     
    def forward(self, children, coef_t, coef_r):
        """Compute weight matrix for how much each vector belongs to the 'left'"""
        # creates a mask of 1's and 0's where 1 means there is a child there
        # has shape (batch_size x max_tree_size x max_children + 1)
        batch_size = children.shape[0]
        max_tree_size = children.shape[1]
        max_children = children.shape[2]

        mask = torch.cat((
            torch.zeros((batch_size, max_tree_size, 1)).double(),torch.min(children, torch.ones((batch_size, max_tree_size, max_children)).double()))
        ,2)
        
        # eta_l is shape (batch_size x max_tree_size x max_children + 1)
        result = torch.mul(torch.mul((1.0 - coef_t).double(),(1.0 - coef_r).double()).double(),mask)
        return result

class CHILDREN_TENSOR(nn.Module):
    def __init__(self, opt):
        super(CHILDREN_TENSOR, self).__init__()
        self.opt = opt
        self.cuda = False
        if self.opt.cuda:
            self.cuda = True

        self.num_features = opt.num_features
        self.zero_vecs = torch.zeros((batch_size, 1, self.num_features)).double()
     
    def forward(self, nodes, children, feature_size):
        # children is batch x num_nodes
        batch_size = children.shape[0]
        num_nodes = children.shape[1]
        max_children = children.shape[2]

        # replace the root node with the zero vector so lookups for the 0th
        # vector return 0 instead of the root vector
        # zero_vecs is (batch_size, num_nodes, 1)
        
        # vector_lookup is (batch_size x num_nodes x feature_size)
        # print("Shape zero vec : " + str(zero_vecs.shape))
        vector_lookup = torch.cat((zero_vecs, nodes[:, 1:, :]), 1)
        # print("Vector look up : " + str(vector_lookup.shape))
        # children is (batch_size x num_nodes x num_children x 1)
        children = children.unsqueeze(3)
        # prepend the batch indices to the 4th dimension of children
        # batch_indices is (batch_size x 1 x 1 x 1)
        batch_indices = torch.arange(0, batch_size)
        batch_indices = batch_indices.view(batch_size, 1, 1, 1).double()
        # batch_indices is (batch_size x num_nodes x num_children x 1)
        batch_indices = tile(batch_indices, 1, num_nodes, self.cuda)
        batch_indices = tile(batch_indices, 2, max_children, self.cuda)
        # children is (batch_size x num_nodes x num_children x 2)
        children = torch.cat((batch_indices, children), 3).long()
        # output will have shape (batch_size x num_nodes x num_children x feature_size)
        # NOTE: tf < 1.1 contains a bug that makes backprop not work for this!
      
        result = vector_lookup[children[:,:,:,0],children[:,:,:,1],:]

        return result


# class CONVOLUTIONAL_LAYER(nn.Module):
#     def __init__(self, opt):
#         super(TBCNN, self).__init__()

#         self.opt = opt
#         self.num_features = opt.num_features
#         self.output_size = opt.output_size
#         self.label_size = opt.n_classes

#         self.hidden_layer = nn.Sequential(
#             nn.Linear(self.output_size, self.output_size),
#             nn.LeakyReLU(),
#             nn.Linear(self.output_size, self.label_size),
           
#         )

class CONV_NODE(nn.Module):
    def __init__(self, opt):
        super(CONV_NODE, self).__init__()
        self.opt = opt
        self.num_features = opt.num_features
        self.output_size = opt.output_size

        self.w_t = torch.randn(self.num_features, self.output_size).double()
        self.w_l = torch.randn(self.num_features, self.output_size).double()
        self.w_r = torch.randn(self.num_features, self.output_size).double()
        self.b_conv = torch.randn(self.output_size).double()

        self.conv_step = CONV_STEP(self.opt)

     
    def forward(self, nodes, children, feature_size):
        """Perform convolutions over every batch sample."""
        batch_size = children.shape[0]
        max_tree_size = children.shape[1]
        max_children = children.shape[2]

        conv_result = self.conv_step(nodes, children, feature_size, self.w_t, self.w_r, self.w_l, self.b_conv)
        return conv_result

class CONV_LAYER(nn.Module):
    def __init__(self, opt):
        super(CONV_LAYER, self).__init__()
        self.opt = opt
        self.num_features = opt.num_features
        self.output_size = opt.output_size
        self.conv_node = CONV_NODE(self.opt)

    def forward(self, num_conv, nodes, children, feature_size):
        nodes = [
            self.conv_node(nodes, children, feature_size)
            for _ in range(num_conv)
        ]
        return torch.cat(nodes, 2)

class CONV_STEP(nn.Module):
    def __init__(self, opt):
        super(CONV_STEP, self).__init__()
        self.opt = opt
        self.cuda = False
        if self.opt.cuda:
            self.cuda = True
    
        self.num_features = opt.num_features
        self.children_tensor = CHILDREN_TENSOR(self.opt)

        self.eta_t =  ETA_T(self.opt)
        self.eta_r =  ETA_R(self.opt)
        self.eta_l =  ETA_L(self.opt)
    
    def forward(self, nodes, children, feature_size, w_t, w_r, w_l, b_conv):
        """Convolve a batch of nodes and children.

        Lots of high dimensional tensors in this function. Intuitively it makes
        more sense if we did this work with while loops, but computationally this
        is more efficient. Don't try to wrap your head around all the tensor dot
        products, just follow the trail of dimensions.
        """

        # nodes is shape (batch_size x max_tree_size x feature_size)
        batch_size = children.shape[0]
        max_tree_size = children.shape[1]
        max_children = children.shape[2]

        # children is shape (batch_size x max_tree_size x max_children)
        # children_tensor = CHILDREN_TENSOR(self.max_children, self.batch_size, self.max_tree_size, feature_size, self.opt)
        children_vectors = self.children_tensor(nodes, children, feature_size)
        # add a 4th dimension to the nodes tensor
        nodes = nodes.unsqueeze(2)
        # tree_tensor is shape (batch_size x max_tree_size x max_children + 1 x feature_size)

        tree_tensor = torch.cat((nodes, children_vectors), 2)


        c_t = self.eta_t(children).double()
        c_r = self.eta_r(children, c_t).double()
        c_l = self.eta_l(children, c_t, c_r).double()

        coef = torch.stack((c_t, c_r, c_l),3).double()

        weights = torch.stack([w_t, w_r, w_l], 0).double()

    
        # reshape for matrix multiplication
        x = batch_size * max_tree_size
        y = max_children + 1

        result = tree_tensor.view(x,y,feature_size)

        coef = coef.view(x,y,3)
      
        result = torch.matmul(torch.transpose(result, 2, 1), coef)
        result = result.view(batch_size, max_tree_size, 3, feature_size)

        # # output is (batch_size, max_tree_size, output_size)
        result = torch.tensordot(result, weights, [[2, 3], [0, 1]])
        # # output is (batch_size, max_tree_size, output_size)
    
        return torch.tanh(result + b_conv)

class TBCNN(nn.Module):
    """
    Tree-Based Convolutional Neural Network (TBCNN)
  
    """
    def __init__(self, opt):
        super(TBCNN, self).__init__()

        self.opt = opt
        self.num_features = opt.num_features
        self.output_size = opt.output_size
        self.label_size = opt.n_classes

        self.hidden_layer = nn.Sequential(
            nn.Linear(self.output_size, self.output_size),
            nn.LeakyReLU(),
            nn.Linear(self.output_size, self.label_size),
           
        )

        self.conv_layer = CONV_LAYER(self.opt)
      
        self._initialization()

    # def tile(self, a, dim, n_tile):
    #     init_dim = a.size(dim)
    #     repeat_idx = [1] * a.dim()
    #     repeat_idx[dim] = n_tile
    #     a = a.repeat(*(repeat_idx))

    #     order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    #     if self.opt.cuda:
    #         order_index = order_index.cuda()
    #     return torch.index_select(a, dim, order_index)
   

    def pooling_layer(self, nodes):
        return torch.max(nodes, 1)[0]

   
    # def conv_layer(self, num_conv, nodes, children, feature_size):
    #     nodes = [
    #         self.conv_node(nodes, children, feature_size)
    #         for _ in range(num_conv)
    #     ]
    #     return torch.cat(nodes, 2)

    # def conv_node(self, nodes, children, feature_size):
    #     conv_result = self.conv_step(nodes, children, feature_size)
    #     return conv_result

    # def conv_node(self, nodes, children, feature_size, output_size):
        
    #     w_t = torch.randn(feature_size, output_size).double()
    #     w_l = torch.randn(feature_size, output_size).double()
    #     w_r = torch.randn(feature_size, output_size).double()

    #     b_conv = torch.randn(output_size).double()

    #     conv_result = self.conv_step(nodes, children, feature_size, w_t, w_r, w_l, b_conv)
    #     return conv_result

    # def conv_step(self, nodes, children, feature_size):
    #     """Convolve a batch of nodes and children.

    #     Lots of high dimensional tensors in this function. Intuitively it makes
    #     more sense if we did this work with while loops, but computationally this
    #     is more efficient. Don't try to wrap your head around all the tensor dot
    #     products, just follow the trail of dimensions.
    #     """

    #     # nodes is shape (batch_size x max_tree_size x feature_size)
    #     batch_size = children.shape[0]
    #     max_tree_size = children.shape[1]
    #     max_children = children.shape[2]

    #     # children is shape (batch_size x max_tree_size x max_children)
    #     children_tensor = CHILDREN_TENSOR(max_children, batch_size, max_tree_size, feature_size, self.opt)
    #     children_vectors = children_tensor(nodes, children, feature_size)
    #     # add a 4th dimension to the nodes tensor
    #     nodes = nodes.unsqueeze(2)
    #     # tree_tensor is shape (batch_size x max_tree_size x max_children + 1 x feature_size)

    #     tree_tensor = torch.cat((nodes, children_vectors), 2)
    #     eta_t =  ETA_T(batch_size, max_tree_size, max_children, self.opt)
    #     eta_r =  ETA_R(batch_size, max_tree_size, max_children, self.opt)
    #     eta_l =  ETA_L(batch_size, max_tree_size, max_children, self.opt)

    #     c_t = eta_t(children).double()
    #     c_r = eta_r(children, c_t).double()
    #     c_l = eta_l(children, c_t, c_r).double()

    #     coef = torch.stack((c_t, c_r, c_l),3).double()

    #     weights = torch.stack([self.w_t, self.w_r, self.w_l], 0).double()

    
    #     # reshape for matrix multiplication
    #     x = batch_size * max_tree_size
    #     y = max_children + 1

    #     result = tree_tensor.view(x,y,feature_size)

    #     coef = coef.view(x,y,3)
      
    #     result = torch.matmul(torch.transpose(result, 2, 1), coef)
    #     result = result.view(batch_size, max_tree_size, 3, feature_size)

    #     # # output is (batch_size, max_tree_size, output_size)
    #     result = tensordot(result, weights, [[2, 3], [0, 1]])
    #     # # output is (batch_size, max_tree_size, output_size)
    
    #     return torch.tanh(result + self.b_conv)
       

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)

    def forward(self, nodes, children):
       
        conv = self.conv_layer(1, nodes, children,  self.num_features)
        # print(conv)
        # print(conv.shape)
        pooling = self.pooling_layer(conv)
        # print(pooling)
        # print(pooling.shape)
        features = self.hidden_layer(pooling)
        # print(features)
        return features
