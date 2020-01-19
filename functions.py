import torch
from torch.autograd import Function
import cross_corr

mbr = cross_corr.meshgrid_based_rotation(16,16,16) # TODO: change '16' to actual size.

class VectorQuantization(Function):
    
    @staticmethod
    def forward(ctx, inputs, codebook):
        '''
        TODO: Making following assumptions for rotation
        inputs is of size C x H x W
        '''
        
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size) # Seems like input has shape H,W,C. I am 
            # assuming C,H,W. So probably will need to fix that after shape validation

            rotated_inputs = mbr.rotate2D(inputs.unsqueeze(0)).permute(0, 2, 1, 3, 4) #B,angles,C,H,W

            # This flatenning only makes sense when embedding_size is C*H*W. 
            # Otherwise we'll have to permute the rotated_inputs probably before flatenning.
            rotated_inputs_flatten = rotated_inputs.view(-1, rotated_inputs.shape[1], embedding_size)
            
            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            rot_inputs_sqr = torch.sum(rotated_inputs_flatten ** 2, dim=2, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)
            
            # Compute the distances to the codebook
            rot_distances = torch.addmm(codebook_sqr + rot_inputs_sqr,
                rotated_inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)
            
            dB, dA, dF = rot_distances.shape
            _, rot_indices_flatten = torch.min(rot_distances.view(dB, -1))
            rot_indices = rot_indices_flatten%dF
            rot_indices = rot_indices.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(rot_indices)

            # return indices
            return rot_indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

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
