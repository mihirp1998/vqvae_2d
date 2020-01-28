import torch
from torch.autograd import Function
#<<<<<<< HEAD
import ipdb
st = ipdb.set_trace
#=======
import cross_corr

mbr = cross_corr.meshgrid_based_rotation(16,16,16) # TODO: change '16' to actual size.
#>>>>>>> dc4fab3cb9f342258faa7c388190a35e4a29eccd

class VectorQuantization(Function):
    
    @staticmethod
    def forward(ctx, inputs, codebook, object_level):
        '''
        TODO: Making following assumptions for rotation
        inputs is of size C x H x W
        '''
        
        with torch.no_grad():
            # assuming C,H,W. So probably will need to fix that after shape validation
            # This flatenning only makes sense when embedding_size is C*H*W. 
            # Otherwise we'll have to permute the rotated_inputs probably before flatenning.
            embedding_size = codebook.size(1)
            dictionary_size = codebook.size(0)
            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            if object_level:
                B,_,_,_ = list(inputs.shape)
                if True:
                    rotated_inputs = mbr.rotate2D(inputs).permute(0,2,1,3,4)
                    B,angles,C,H,W = list(rotated_inputs.shape)
                    C = C*H*W
                    assert(C==embedding_size)
                    rot_input = rotated_inputs.reshape(B,angles,-1)
                    rot_inputs_sqr = torch.sum(rot_input ** 2, dim=2, keepdim=True)
                    
                    rot_distances = (rot_inputs_sqr + codebook_sqr - 2 * torch.matmul(rot_input, codebook.t()))                    
                    
                    dB, dA, dF = rot_distances.shape
                    rot_distances = rot_distances.view(B, -1)
                    rotIdxMin = torch.argmin(rot_distances, dim=1).unsqueeze(1)
                    best_rotations = rotIdxMin//dF # Find the rotation for min distance
                    best_rotations = best_rotations.squeeze(1)
                    encoding_indices = rotIdxMin%dF # Find the best index (which will be column in rotAngle-index grid)
                    encoding_indices = encoding_indices.squeeze(1)
                    ctx.mark_non_differentiable(encoding_indices)
                    return encoding_indices,best_rotations
                else:
                    inputs_flatten = inputs.reshape(B,-1)
                    inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
                    distances = torch.addmm(codebook_sqr + inputs_sqr,inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)
                    _, indices_flatten = torch.min(distances, dim=1)
                    ctx.mark_non_differentiable(indices_flatten)
                    return indices_flatten,torch.zeros(1)
            else:
                    inputs_size = inputs.size()
                    inputs_flatten = inputs.view(-1, embedding_size) # Seems like input has shape H,W,C. I am 
                    inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
                    distances = torch.addmm(codebook_sqr + inputs_sqr,inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)
                    _, indices_flatten = torch.min(distances, dim=1)
                    indices = indices_flatten.view(*inputs_size[:-1])
                    ctx.mark_non_differentiable(indices)
                    return indices,torch.zeros(1)

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook,object_level):
        indices,rotations_select = vq(inputs, codebook,object_level)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)
        codes_flatten = torch.index_select(codebook, dim=0,index=indices_flatten)
        codes = codes_flatten.view_as(inputs)
        if object_level:
            codes = mbr.rotate2D_pose(codes,rotations_select).squeeze(2)
        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
__all__ = [vq, vq_st]
