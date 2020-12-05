import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.utils as vutils
from vgg import VGG
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator,self).__init__()
        self.main = nn.Sequential(
            #Output size: (64,16,16)
            nn.ConvTranspose2d(nz,64,16,1,bias=False),
            nn.ReLU(inplace=False),

            #Output size: (3,32,32)
            nn.ConvTranspose2d(64,3,17,1,bias=False),
            # nn.ReLU(inplace=False),

            #Output size: (64,25,25)
            # nn.ConvTranspose2d(128,64,9,1,bias=False),
            # nn.ReLU(inplace=False),

            #Output size: (3,32,32)
            # nn.ConvTranspose2d(64,3,8,1,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            
            #Output size: (64,11,11)
            nn.Conv2d(3,64,11,2),
            nn.LeakyReLU(0.2, inplace=False),

            #Output size: (128,1,1)
            nn.Conv2d(64,128,11,2),
            nn.LeakyReLU(0.2, inplace=False),

            #Output size: (256,1,1)
            # nn.Conv2d(128,256,5,2),
            # nn.LeakyReLU(0.2, inplace=False),

            nn.Flatten(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.main(x)
        return x


class counterGAN():
    def __init__(self,device):
        
        self.nz = 100
        self.beta1 = 0.5
        self.real_label = 1
        self.fake_label = 0
        self.L = 100
        self.device = device
        self.iters = 0

        self.netG = Generator(self.nz).to(self.device)
        self.netD = Discriminator().to(self.device)
        self.netTarget = VGG('VGG16').to(self.device)
        self.netTarget.load_state_dict(torch.load('BestClassifierModel.pth',map_location=self.device))

        # fixed_noise -> stores fixed generator seed for inference
        self.fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)

        self.netG.apply(self.weights_init)
        self.netD.apply(self.weights_init)

        self.optimizerG = optim.Adam(self.netG.parameters(), lr=2e-4, betas=(self.beta1,0.999))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=2e-4, betas=(self.beta1,0.999))
        self.criterion = nn.BCELoss()
        self.criterionTarget = nn.CrossEntropyLoss()

        # criterionPerturbation -> norm of the generated noise
        self.criterionPerturbation = nn.MSELoss()

    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    def save_state_dicts(self,path):
        
        torch.save({
            'netG_state_dict':self.netG.state_dict(),
            'netD_state_dict':self.netD.state_dict(),
            'optimizerG_state_dict':self.optimizerG.state_dict(),
            'optimizerD_state_dict':self.optimizerD.state_dict()
        },path)
        

    def load_state_dicts(self,path):
        
        self.netG.load_state_dict(torch.load(path)['netG_state_dict'])
        self.netD.load_state_dict(torch.load(path)['netD_state_dict'])
        self.optimizerG.load_state_dict(torch.load(path)['optimizerG_state_dict'])
        self.optimizerD.load_state_dict(torch.load(path)['optimizerD_state_dict'])

        print('Loaded state dicts')

    def train(self,init_epoch, num_epochs,dataloaders,dataloaders_adv):

        G_losses = []
        D_losses = []
        T_losses = []
        P_losses = []
        Losses = []
        img_list_adv = []
        img_list_real = []
        self.iters = 0

        
        for epoch in range(init_epoch,num_epochs):
            # pbar = tqdm(total=len(dataloaders_adv['train']))
            for idx, (D, D_adv) in enumerate(zip(dataloaders['train'], dataloaders_adv['train'])):
                # pbar.update(1)
                self.netG.train()
                self.netD.train()
                
                #Train Discriminator on adversarial image
                self.netD.zero_grad()
                
                
                
                #adversarial images definition              
                image_adv = D_adv[0].to(self.device)
                target_labels_adv = D_adv[1].to(self.device)
                batch_size_adv = image_adv.size()[0]
                # label_adv -> target label to use for adversarial images while training discriminator
                label_adv = torch.full((batch_size_adv,),self.fake_label,dtype=torch.float,device=self.device)
                
                #real images definition
                image = D[0][:batch_size_adv,...].to(self.device)
                target_labels = D[1].to(self.device)
                batch_size = image.size()[0]
                # label -> target label to use for real images while training discriminator
                label = torch.full((batch_size,),self.real_label,dtype=torch.float,device=self.device)
                

                out_real= self.netD(image).view(-1)
                loss_d_real = self.criterion(out_real,label)
                # loss_d_real.backward()

                # D_x -> output of the discriminator for real images. Between (0,1)
                D_x = out_real.mean().item()

                #Train Discriminator on generated images
                noise = torch.randn(batch_size,self.nz,1,1,device=self.device)
                generated = self.netG(noise)+image_adv
                label_adv = torch.full((batch_size,),self.fake_label,dtype=torch.float,device=self.device)

                out_generated = self.netD(generated.clone().detach()).view(-1)
                loss_d_generated = self.criterion(out_generated,label_adv.detach())
                # loss_d_generated.backward()

                # D_G_z1 -> output of the discriminator for generated images. Between (0,1)
                D_G_z1 = out_generated.mean().item()

                # loss_d -> total loss of discriminator
                loss_d = loss_d_generated+loss_d_real

                # self.optimizerD.step()

                #Train Generator
                self.netG.zero_grad()
                self.netTarget.zero_grad()

                label_adv = torch.full((batch_size,),self.real_label,dtype=torch.float,device=self.device)

                out_generated_2 = self.netD(generated.clone()).view(-1)
                loss_g = self.criterion(out_generated_2,label_adv)
                # loss_g.backward(retain_graph=True)
                
                # Calculate loss on classifier
                out_classifier = self.netTarget(generated.clone())
                loss_c = self.criterionTarget(out_classifier,target_labels_adv)
                # loss_c.backward()

                # D_G_z2 -> output of the discriminator for generated images. Between (0,1)
                D_G_z2 = out_generated_2.mean().item()

                # Calculate norm of noise generated
                loss_p = self.criterionPerturbation(image,generated)

                # loss -> final loss
                loss = loss_d+(loss_c*loss_c)+loss_g-loss_p
                loss.backward()
                self.optimizerD.step()
                self.optimizerG.step()

                # self.optimizerG.step()

                if idx%50==0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_C: %.4f\tLoss_P: %.4f\tOverall loss: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, idx, len(dataloaders['train']),
                     loss_d.item(), loss_g.item(), loss_c.item(), loss_p.item(), loss.item(), D_x, D_G_z1, D_G_z2))
                
                G_losses.append(loss_g.item())
                D_losses.append(loss_d.item())
                T_losses.append(loss_c.item())
                P_losses.append(loss_p.item())
                Losses.append(loss.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (self.iters % 50 == 0) or ((epoch == num_epochs-1) and (idx == len(dataloaders['train'])-1)):
                    with torch.no_grad():
                        generated = (self.netG(self.fixed_noise)+image_adv[:64]).detach().cpu()
                        img_list_adv.append(vutils.make_grid(generated, padding=2, normalize=True))
                    # TODO Visualize
                    with torch.no_grad():
                        generated = (self.netG(self.fixed_noise)+image[:64]).detach().cpu()
                        img_list_real.append(vutils.make_grid(generated, padding=2, normalize=True))
                self.iters += 1

                
            # pbar.close()
            if epoch%5==0 or epoch==num_epochs-1:
                self.visualize_images(img_list_adv,epoch,img_type='Adversarial')
                self.visualize_images(img_list_real,epoch,img_type='Real')
            self.save_state_dicts(f'BestcounterGAN_{epoch}.pth')
        return D_losses,G_losses, img_list_adv, img_list_real


    def visualize_images(self, img_list, epoch=None, img_type='Real'):

        if epoch==None:
            
            epoch = ''
        
        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Generated Images {}".format(img_type))
        plt.imshow(np.transpose(img_list[-1],(1,2,0)))
        plt.savefig('./images/Image_{}_{}.png'.format(epoch,img_type))

        return

    def inference(self, images, fixed_noise):
        '''
        Parameters:
        images -> input images Tensor(64,3,32,32)
        fixed_noise -> fixed noise (64,100,1,1)

        Returns:
        generated -> generated images Tensor(64,3,32,32)
        '''

        generated = (self.netG(self.fixed_noise)+images).detach().cpu()

        return generated


        



