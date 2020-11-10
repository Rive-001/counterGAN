import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data as data


class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator,self).__init__()
        self.main = nn.Sequential(
            #Output size: (256,9,9)
            nn.ConvTranspose2d(nz,256,9,1,bias=False),
            nn.ReLU(),

            #Output size: (128,17,17)
            nn.ConvTranspose2d(256,128,9,1,bias=False),
            nn.ReLU(),

            #Output size: (64,25,25)
            nn.ConvTranspose2d(128,64,9,1,bias=False),
            nn.ReLU(),

            #Output size: (1,32,32)
            nn.ConvTranspose2d(64,3,8,1,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            
            #Output size: (64,14,14)
            nn.Conv2d(3,64,5,2),
            nn.LeakyReLU(0.2),

            #Output size: (128,5,5)
            nn.Conv2d(64,128,5,2),
            nn.LeakyReLU(0.2),

            #Output size: (256,1,1)
            nn.Conv2d(128,256,5,2),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.main(x)
        return x


class counterGAN():
    def __init__(device):
        
        self.nz = 100
        self.beta1 = 0.5
        self.real_label = 1
        self.fake_label = 0

        self.netG = Generator(self.nz).to(device)
        self.netD = Discriminator().to(device)

        self.optimizerG = optim.Adam(self.netG.parameters(), lr=2e-4, betas=(beta1,0.999))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=2e-4, betas=(beta1,0.999))
        self.criterion = nn.BCELoss()


    def train(init_epoch, num_epochs,dataloader):

        G_losses = []
        D_losses = []

        for epoch in range(init_epoch,num_epochs):
            for idx,D in enumerate(dataloader['train']):
                
                #Train Discriminator on real image
                self.netD.zero_grad()

                image = D[0].to(device)
                batch_size = image.size()[0]
                label = torch.full((batch_size,),self.real_label,dtype=torch.float,device=device)

                out_real = self.netD(image).view(-1)
                loss_d_real = self.criterion(out_real,label)
                loss_d_real.backward()

                D_x = out_real.mean().item()

                #Train Discriminator on generated images
                noise = torch.randn(batch_size,self.nz,1,1,device=device)

                generated = self.netG(noise)
                label.fill_(self.fake_label)

                out_generated = self.netD(generated.detach()).view(-1)
                loss_d_generated = self.criterion(out_generated,label)
                loss_d_generated.backward()

                D_G_z1 = out_generated.mean().item()

                loss_d = loss_d_generated+loss_d_real
                self.optimizerD.step()

                #Train Generator
                self.netG.zero_grad()

                label.fill_(self.real_label)

                out_generated_2 = self.netD(generated).view(-1)
                loss_g = self.criterion(out_generated_2,label)
                loss_g.backward()

                D_G_z2 = out_generated_2.mean().item()
                self.optimizerG.step()

                if idx%50==0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, idx, len(dataloader['train']),
                     loss_d.item(), loss_g.item(), D_x, D_G_z1, D_G_z2))
                
                G_losses.append(loss_g.item())
                D_losses.append(loss_d.item())

        return D_losses,G_losses

