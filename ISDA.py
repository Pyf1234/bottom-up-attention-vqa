import torch
import torch.nn as nn
from vqa_debias_loss_functions import Plain


class EstimatorCV():
    def __init__(self, feature_num, class_num):
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.zeros(class_num).cuda()

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot=labels

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(
            var_temp.permute(1, 2, 0),
            var_temp.permute(1, 0, 2)
        ).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A)
        )
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.Amount.view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.Ave - ave_CxA).view(C, A, 1),
                (self.Ave - ave_CxA).view(C, 1, A)
            )
        )

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                      .mul(weight_CV)).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()

        self.Amount += onehot.sum(0)


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

        #little bug
        labels=torch.argmax(one_hot,-1).view(1,-1)

        weight_m = list(fc.parameters())[2]

        CV_temp = cv_matrix[labels].squeeze()

        #little bug
        a=torch.matmul(weight_m,CV_temp)
        b=torch.matmul(a,weight_m.T)
        aug=torch.diagonal(b,dim1=-2,dim2=-1)

        aug_result = y-(one_hot-0.5) * aug*ratio

        return aug_result

    def forward(self, model, fc, v,q,a,b, ratio):

        features = model(v, None, q, a, b)

        y = fc(features)

        self.estimator.update_CV(features.detach(), a)

        isda_aug_y = self.isda_aug(fc, features, y, a, self.estimator.CoVariance.detach(), ratio)

        loss = self.BCE_with_logits(isda_aug_y, a)

        return loss, y