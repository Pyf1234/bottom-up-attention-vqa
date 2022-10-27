import torch
import torch.nn as nn
from torch.nn import functional as F
from vqa_debias_loss_functions import Plain


class EstimatorCV():
    def __init__(self, feature_num, class_num):
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.zeros(class_num).view(class_num,1).cuda()

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        onehot_NxC = labels
       

        expe_temp_CxA=torch.matmul(onehot_NxC.permute(1,0), features)
        

        batch_class_num_Cx1=onehot_NxC.sum(0).view(C,1)
        batch_class_num_Cx1[batch_class_num_Cx1==0]=1
        

        expe_CxA=expe_temp_CxA.div(batch_class_num_Cx1)
       

        pow2_expe_temp_CxA=torch.matmul(torch.transpose(onehot_NxC,0,1),(features.pow(2)))
        pow2_expe_CxA=pow2_expe_temp_CxA.div(batch_class_num_Cx1)
       

        batch_var_CxA=pow2_expe_CxA-expe_CxA.pow(2)
    

        batch_weight_Cx1=batch_class_num_Cx1.div(batch_class_num_Cx1+self.Amount)
        batch_weight_Cx1[batch_weight_Cx1!=batch_weight_Cx1]=0
       


        addition_var_CxA=torch.matmul(torch.matmul((batch_weight_Cx1),torch.transpose((1 - batch_weight_Cx1),0,1)),((self.Ave - expe_CxA).pow(2)))
       

        self.CoVariance = (self.CoVariance.mul(1 - batch_weight_Cx1) + batch_var_CxA
                           .mul(batch_weight_Cx1)).detach() + addition_var_CxA.detach()

        self.Ave = (self.Ave.mul(1 - batch_weight_Cx1) + expe_CxA.mul(batch_weight_Cx1)).detach()

        self.Amount+=batch_class_num_Cx1


class ISDALoss(nn.Module):
    def __init__(self, feature_num, class_num):
        super(ISDALoss, self).__init__()

        self.estimator = EstimatorCV(feature_num, class_num)

        self.class_num = class_num

        self.BCE_with_logits=Plain()

    def isda_aug(self, fc, features, y, one_hot, cv_matrix, ratio):

        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        labels=torch.argmax(one_hot,-1).view(1,-1)

        weight_m = list(fc.parameters())[2]

        CV_temp = cv_matrix[labels].squeeze()

        aug=torch.matmul(CV_temp,torch.transpose(torch.pow(weight_m,2),0,1))

        aug_result = y-(one_hot-0.5) * aug*ratio

        return aug_result

    def forward(self, model, fc, v,q,a,b, ratio):

        features = model(v, None, q, a, b)

        y = fc(features)

        self.estimator.update_CV(features.detach(), a.detach())

        isda_aug_y = self.isda_aug(fc, features, y, a, self.estimator.CoVariance.detach(), ratio)

        loss = F.binary_cross_entropy_with_logits(isda_aug_y, a)*a.size(1)

        return loss, y