import torch

class LBSign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        output = torch.clamp((input*255),0.,255.).round()
        return output/255.

    @staticmethod
    def backward(ctx, grad_output):
        #return grad_output.clamp_(-1, 1)
        return grad_output

