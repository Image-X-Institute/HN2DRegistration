import math
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
#from torchmetrics.clustering import MutualInfoScore
# from jaxtyping import Float


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None,device='cpu'):
        self.win = win
        self.device = device

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(self.device)
        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        #return -torch.mean(cc)
        return 1.0-torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

class L1:
    """
    L1 Loss
    """
    
    def loss(self, y_true, y_pred):
        return torch.mean((y_true-y_pred))

class flow_mask:
    """
    Computes the MSE between a predicted and ground-truth DVF inside a binary mask
    """

    def loss(self, target_flow, predict_flow, mask):
        mask = torch.cat((mask, mask, mask), 1)
        error = target_flow[mask == 1] - predict_flow[mask == 1]
        return torch.sum(error ** 2) / torch.count_nonzero(error)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, y_pred):
        dif = [f * f for f in self._diffs(y_pred)]
        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        return grad.mean()

class Affine:
    """
    Affine registrations have 2nd order derivatives = 0. 
    A rigidity penalty term for nonrigid registration. Staring et al. (2007)
    Attempt to use a finite difference Second-order central approximation.
    """
    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)
        #print(ndims)

        df = [None] * ndims
        for i in range(0,ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[2:, ...] + y[:-2, ...] - 2*y[1:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, y_pred):
        dif = [f * f for f in self._diffs(y_pred)]
        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        return grad.mean()
    

class winPearson:
    """
    Local (over window) Pearson correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")
        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute windows
        Ii = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        Ji = conv_fn(Ji, sum_filt, stride=stride, padding=padding)

        print(Ii.shape)

        #cc = cross * cross / (Ii * Ji + 1e-5)
        cc = 0

        return -torch.mean(cc)

class slicePearson:

    def loss(self, y_true, y_pred, mask):

        corr_all = torch.zeros(y_true.shape[0], y_true.shape[4])

        # iterate over batches
        for batch in range(y_true.shape[0]):

            # iterate over slices
            for n in range(64, y_true.shape[4]):

                # extract slice
                slice_mask = y_pred[batch, :, :, :, n]

                if torch.all(slice_mask == 0):
                    corr_all[batch, n] = 0

                else:
                    # extract slice
                    slice_true = y_true[batch, :, :, :, n]
                    slice_pred = y_pred[batch, :, :, :, n]

                    # delete outside mask
                    slice_true = slice_true[slice_mask == 0]
                    slice_pred = slice_pred[slice_mask == 0]

                    # flatten
                    x = torch.zeros((2, slice_true.shape[0]))
                    x[0, :] = slice_true
                    x[1, :] = slice_pred
                    print(x.shape)

                    corr = torch.corrcoef(x)
                    print(corr)
                    #corr_all[n] = torch.corrcoef()
                    break
                
class JacDeterminant:
    """
    Using code from https://github.com/eigenvivek/polypose/blob/f37c28472894b926422ac5322e532fec09b4082a/src/polypose/loss.py
    """
    def __init__(self,J):#Float[torch.Tensor, "B D H W 3"]):
        self.J = J
    def calculate_jacobian(self): #-> Float[torch.Tensor, "B D H W 3 3"]:
        """
        Compute the Jacobian of the flow field with 1st order finite differences.
        """
        # note can also use torch.gradient which would be more
        # accurate, using second-order central differences
        dy = self.J[:, 1:, :-1, :-1] - self.J[:, :-1, :-1, :-1]
        dx = self.J[:, :-1, 1:, :-1] - self.J[:, :-1, :-1, :-1]
        dz = self.J[:, :-1, :-1, 1:] - self.J[:, :-1, :-1, :-1]
        return dx, dy, dz


    def calculate_jacdet(self): #-> Float[torch.Tensor, "B D H W"]:
        """
        Compute the Jacobian determinant of the flow field.
    
        The flow field (input points + displacement field) should be in units of voxels (i.e., already normalized to the volume size).
    
        Using code from https://github.com/Kidrauh/neural-atlasing/blob/216f624aea3708589e60beee5285eb6781acb981/sinf/utils/util.py#L172-L184
        """
        dx, dy, dz = self.jacobian()
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])
        Jdet = Jdet0 - Jdet1 + Jdet2
        return Jdet
    
    
    def calculate_divergence(self):# -> Float[torch.Tensor, "B D H W 3"]:
        """
        Compute the divergence of the flow field.
        """
        dx, dy, dz = self.jacobian()
        return (dx + dy + dz).abs().square()
    
    
class MutualInformation(torch.nn.Module):
    """
    Mutual Information
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        #print(sigma)

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    def forward(self, y_true, y_pred):
        return -self.mi(y_true, y_pred)

    def loss(self, y_true, y_pred):
        return self.mi(y_true, y_pred)
    
# class MutualInformation:
#     """
#     Using mutual information code from:
#         https://lightning.ai/docs/torchmetrics/stable/clustering/mutual_info_score.html
#     """
#     def __init__(self):
#         self.mi_score = MutualInfoScore()
    
#     def loss(self, y_true, y_pred):
#         return self.mi_score(y_pred,y_true)
    