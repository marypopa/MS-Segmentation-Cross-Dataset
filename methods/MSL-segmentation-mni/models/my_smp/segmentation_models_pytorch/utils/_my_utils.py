import torch
import torch.nn as nn
from .base import Loss
import numpy as np
from skimage import measure, morphology

class WeightedBCELoss(Loss):
    def __init__(self, weight=None):
        super(WeightedBCELoss, self).__init__()
        self.weight = weight

    def forward(self, output, target):
        # Apply sigmoid activation to the model output
        # output = torch.sigmoid(output)

        # Compute the binary cross-entropy loss
        eps_clip = 1e-7
        output = torch.clamp(output, min=eps_clip, max=1 - eps_clip)
        loss = - (self.weight * target * torch.log(output) + (1 - self.weight) * (1 - target) * torch.log(1 - output))
        # if torch.isinf(loss).any().item():
        #     print("infinity...")
        #     ind_inf = torch.isinf(loss)
        #     print("output: ", output[ind_inf])
        #     print("target: ", target[ind_inf])
        #     print("loss: ", loss[ind_inf])

        # Take the mean over the batch
        return torch.mean(loss)

# works on array
# def smax(x, factor=21.65):
#     # The factor was chosen given that images are of size 224x224 and making sure that in the worst case of all pixels
#     # being "undecided" (p=0.5), their contribution exp(factor*pi) is not larger than a single pi=1, i.e. exp(factor*1)
#     # 224^2*exp(factor*0.5) = exp(factor*1). Therefore factor = 2*log(224^2) = 4*log(224) = 21.65
#     n1 = x.size - 1
#     return np.log(np.sum(np.exp(factor*x)) - n1)/factor
# def smin(x, factor=-21.65):
#     return np.log(np.sum(np.exp(factor*x)))/factor
def smax_component(p, component, factor=21.65):
    # The factor was chosen given that images are of size 224x224 and making sure that in the worst case of all pixels
    # being "undecided" (p=0.5), their contribution exp(factor*pi) is not larger than a single pi=1, i.e. exp(factor*1)
    # 224^2*exp(factor*0.5) = exp(factor*1). Therefore factor = 2*log(224^2) = 4*log(224) = 21.65
    pc = p[np.where(component == 1)]
    n1 = pc.size
    # print(">>> n1=", n1)
    return np.log(np.sum(np.exp(factor*pc)) / n1)/factor
def smin_component(p, component, factor=-21.65):
    pc = p[np.where(component == 1)]
    n1 = pc.size
    return np.log(np.sum(np.exp(factor*pc)) / n1)/factor
class CCELoss(Loss):
    def __init__(self, dilation_radius=2, max_function=smax_component, min_function=smin_component):
        # max_function = smax or np.max
        super(CCELoss, self).__init__()
        self.dilation_radius = dilation_radius
        self.max_function = max_function
        self.min_function = min_function

    def forward(self, output, target):

        # compute 2D connected component
        cceps = np.zeros(target.size(0))
        ccens = np.zeros(target.size(0))
        num_components = np.zeros(target.size(0))
        for i in range(target.size(0)):
            target_i = target[i,0].detach().cpu().numpy()
            labels_target = measure.label(target_i, background=0)
            # num_labels, labels = cv2.connectedComponents(image)
            ccep = 0.0
            num_components_target = np.max(labels_target)
            num_components[i] = num_components_target
            output_i = output[i,0].detach().cpu().numpy()

            for label in range(1, num_components_target):
                component = (labels_target == label) + 0.0
                # ccep += -np.log(self.max_function(component * output_i))
                ccep += -np.log(self.max_function(output_i, component))
            cceps[i] = ccep

            dilated_target = morphology.dilation(target_i, morphology.disk(self.dilation_radius))
            complement_target = 1-(dilated_target >= 0.5)
            # ccen = -np.log(self.min_function(complement_target * (1-output_i)))
            ccen = -np.log(self.min_function(1-output_i, complement_target))
            ccens[i] = ccen
        ccep_mean = np.mean(cceps)
        ccen_mean = np.mean(ccens)
        cce_mean = ccep_mean + ccen_mean
        return torch.tensor(cce_mean).to('cuda'), torch.tensor(ccep_mean).to('cuda'), torch.tensor(ccen_mean).to('cuda')

class wBCE_CCELoss(WeightedBCELoss,CCELoss):
    def __init__(self, weightBCE, weight, dilation_radius=2):
        super(wBCE_CCELoss, self).__init__()
        self.weight = weight
        self.dilation_radius = dilation_radius
        self.wBCE = WeightedBCELoss(weightBCE)
        self.CCELoss = CCELoss()
        # precomputed values are stored for later access

    def compute(self, output, target):
        wbceloss = self.wBCE(output, target)
        # wcceloss = self.weight*self.CCELoss(output, target)
        cce_mean, ccep_mean, ccen_mean = self.CCELoss(output, target)
        wcceloss = self.weight*cce_mean
        wcceploss = self.weight*ccep_mean
        wccenloss = self.weight*ccen_mean

        wbce_cce = wbceloss + wcceloss
        return wbce_cce, wbceloss, wcceloss, wcceploss, wccenloss

    def forward(self, output, target):
        wbce_cce, wbce, wcce, wccep, wccen = self.compute(output, target)
        return wbce_cce
