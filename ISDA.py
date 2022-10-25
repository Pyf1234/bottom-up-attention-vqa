import torch
import torch.nn as nn
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
        print(onehot_NxC.shape)

        expe_temp_CxA=torch.transpose(onehot_NxC,0,1).mul(features)
        print(expe_temp_CxA.shape)

        batch_class_num_Cx1=onehot_NxC.sum(0).view(C,1)
        batch_class_num_Cx1[batch_class_num_Cx1==0]=1
        print(batch_class_num_Cx1.shape)

        expe_CxA=expe_temp_CxA.div(batch_class_num_Cx1)
        print(expe_CxA.shape)

        pow2_expe_temp_CxA=torch.transpose(onehot_NxC,0,1).mul(features.pow(2))
        pow2_expe_CxA=pow2_expe_temp_CxA.div(batch_class_num_Cx1)
        print(pow2_expe_CxA.shape)

        batch_var_CxA=pow2_expe_CxA-expe_CxA.pow(2)
        print(batch_var_CxA.shape)

        batch_weight_Cx1=batch_class_num_Cx1.div(batch_class_num_Cx1+self.Amount)
        batch_weight_Cx1[batch_weight_Cx1!=batch_weight_Cx1]=0
        print(batch_weight_Cx1.shape)


        addition_var_CxA=batch_weight_Cx1.mul(1 - batch_weight_Cx1).mul((self.Ave - expe_CxA).pow(2))
        print(addition_var_CxA.shape)

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