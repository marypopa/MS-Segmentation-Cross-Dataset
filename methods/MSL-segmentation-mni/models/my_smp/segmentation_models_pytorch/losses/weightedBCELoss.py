# from torch import nn
# import torch
# import numpy as np
# from skimage import measure, morphology
#
#
# class WeightedBCELoss(nn.Module):
#     def __init__(self, weight=None):
#         super(WeightedBCELoss, self).__init__()
#         self.weight = weight
#
#     def forward(self, output, target):
#         # Apply sigmoid activation to the model output
#         # output = torch.sigmoid(output)
#
#         # Compute the binary cross-entropy loss
#         eps_clip = 1e-7
#         output = torch.clamp(output, min=eps_clip, max=1 - eps_clip)
#         loss = - (self.weight * target * torch.log(output) + (1 - self.weight) * (1 - target) * torch.log(1 - output))
#         # if torch.isinf(loss).any().item():
#         #     print("infinity...")
#         #     ind_inf = torch.isinf(loss)
#         #     print("output: ", output[ind_inf])
#         #     print("target: ", target[ind_inf])
#         #     print("loss: ", loss[ind_inf])
#
#         # Take the mean over the batch
#         return torch.mean(loss)
#
#
# ## afisat acest CCE cand folosim WCE, sa apara si el afisat
# ## si asta afisat si cu ponderea
# ## lasam sa mearga, - raportul dintre ele si incercam sa le echilibram
# ## si punem ponderea ip cu asta
# class CCELoss(nn.Module):
#     def __init__(self, weight=None, dilation_radius=2):
#         super(CCELoss, self).__init__()
#         self.weight = weight
#         self.dilation_radius = dilation_radius
#
#     def forward(self, output, target):
#
#         # compute 2D connected component
#         for i in range(target.size[0]):
#             labels_target = measure.label(target, background=0)
#             # num_labels, labels = cv2.connectedComponents(image)
#             cce = 0.0
#             num_components_target = np.max(labels_target)
#
#             for label in range(1, num_components_target):
#                 component = (labels_target == label) + 0.0
#                 cce += -np.log(np.max(component * output))
#
#             dilated_target = morphology.dilation(target, morphology.disk(self.dilation_radius))
#             complement_target = 1-(dilated_target >= 0.5)
#
#             cce += -np.log(1 - np.max(complement_target * output))
#
#
#
#         return torch.mean(loss)