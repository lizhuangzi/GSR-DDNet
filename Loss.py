
import torch
from torch import nn
from torchvision.models.vgg import vgg19



class GeneratorLoss(nn.Module):
    def __init__(self,lambda1,lambda2,lambda3):
        super(GeneratorLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        loss_network = nn.Sequential(*list(vgg.features)[:37]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = -torch.mean(out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss

        #return image_loss + 0.0015 * adversarial_loss + 0.008 * perception_loss + 2e-8 * tv_loss
        return self.lambda1*image_loss + self.lambda2 * perception_loss + self.lambda3 * adversarial_loss
