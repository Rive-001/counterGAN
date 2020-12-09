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


def get_seed(image=None, z_size=100, device="cuda"):
    # return torch.randn(image.shape[0], z_size, 1, 1, device=device)
    """Return seed that is non-differentiable representation of image"""
    if image is None:
        return torch.randn(z_size, requires_grad=False, device=device)
    else:
        vector = (1.0 * (image > 0.5)).reshape(image.shape[0], 1, -1)
        vector = torch.nn.functional.interpolate(vector, size=z_size)
        return vector.squeeze(1)


class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Output size: (64,16,16)
            nn.ConvTranspose2d(nz, 64, 16, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            # Output size: (3,32,32)
            nn.ConvTranspose2d(64, 3, 17, 1, bias=False),
            nn.BatchNorm2d(3),
            # nn.ReLU(inplace=False),
            # Output size: (64,25,25)
            # nn.ConvTranspose2d(128,64,9,1,bias=False),
            # nn.ReLU(inplace=False),
            # Output size: (3,32,32)
            # nn.ConvTranspose2d(64,3,8,1,bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.main(x)
        return x


class GeneratorLinear(nn.Module):
    def __init__(self, nz):
        super(GeneratorLinear, self).__init__()
        self.main = nn.Sequential(
            nn.ReLU(),
            nn.Linear(100, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 32 * 32 * 3),
            # Output size: (64,16,16)
            # nn.ConvTranspose2d(nz,64,16,1,bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=False),
            # #Output size: (3,32,32)
            # nn.ConvTranspose2d(64,3,17,1,bias=False),
            # nn.BatchNorm2d(3),
            # nn.ReLU(inplace=False),
            # Output size: (64,25,25)
            # nn.ConvTranspose2d(128,64,9,1,bias=False),
            # nn.ReLU(inplace=False),
            # Output size: (3,32,32)
            # nn.ConvTranspose2d(64,3,8,1,bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.reshape(-1, 100)
        x = self.main(x)
        x = x.reshape(-1, 3, 32, 32)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Output size: (64,11,11)
            nn.Conv2d(3, 64, 11, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=False),
            # Output size: (128,1,1)
            nn.Conv2d(64, 128, 11, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False),
            # Output size: (256,1,1)
            # nn.Conv2d(128,256,5,2),
            # nn.LeakyReLU(0.2, inplace=False),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.main(x)
        return x


class DiscriminatorLinear(nn.Module):
    def __init__(self):
        super(DiscriminatorLinear, self).__init__()
        self.main = nn.Sequential(
            # in (N,3,32,32)
            nn.Linear(32 * 32 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(0.2),
            # nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            # Output size: (64,11,11)
            # nn.Conv2d(3,64,11,2),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(0.2, inplace=False),
            # #Output size: (128,1,1)
            # nn.Conv2d(64,128,11,2),
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU(0.2, inplace=False),
            # Output size: (256,1,1)
            # nn.Conv2d(128,256,5,2),
            # nn.LeakyReLU(0.2, inplace=False),
            # nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.main(x.reshape(x.shape[0], -1))
        return x


class counterGAN:
    def __init__(self, device, lr=None, linear=False, wandb=None):

        self.nz = 100
        self.beta1 = 0.5
        self.real_label = 1
        self.fake_label = 0
        self.L = 100
        self.device = device
        self.iters = 0
        if linear == True:
            self.netG = GeneratorLinear(self.nz).to(self.device)
            self.netD = DiscriminatorLinear().to(self.device)
        else:
            self.netG = Generator(self.nz).to(self.device)
            self.netD = Discriminator().to(self.device)
        self.netTarget = VGG("VGG16").to(self.device)
        self.netTarget.load_state_dict(
            torch.load(
                "/home/stars/Code/tarang/idl_proj/counterGAN/BestClassifierModel.pth",
                map_location=self.device,
            )
        )

        # fixed_noise -> stores fixed generator seed for inference
        self.fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)

        self.netG.apply(self.weights_init)
        self.netD.apply(self.weights_init)

        if lr is None:
            lr = 2e-3
        self.lr = lr
        self.wandb = wandb
        if self.wandb:
            self.wandb.watch(self.netG, log="all")
            self.wandb.watch(self.netD, log="all")
        self.optimizerG = optim.AdamW(
            self.netG.parameters(),
            lr=self.lr,
            betas=(self.beta1, 0.999),
            weight_decay=4e-3,
        )
        self.optimizerD = optim.AdamW(
            self.netD.parameters(),
            lr=self.lr,
            betas=(self.beta1, 0.999),
            weight_decay=4e-3,
        )
        self.schedulerG = optim.lr_scheduler.StepLR(
            self.optimizerG, step_size=3, gamma=0.93
        )
        self.schedulerD = optim.lr_scheduler.StepLR(
            self.optimizerD, step_size=3, gamma=0.93
        )
        self.criterion = nn.BCELoss()
        # self.criterion = nn.MSELoss()
        self.criterionTarget = nn.CrossEntropyLoss()

        # criterionPerturbation -> norm of the generated noise
        self.criterionPerturbation = nn.MSELoss()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    def save_state_dicts(self, path, wandb=None):

        torch.save(
            {
                "netG_state_dict": self.netG.state_dict(),
                "netD_state_dict": self.netD.state_dict(),
                "optimizerG_state_dict": self.optimizerG.state_dict(),
                "optimizerD_state_dict": self.optimizerD.state_dict(),
            },
            path,
        )
        if wandb:
            wandb.save(path)

    def load_state_dicts(self, path):

        self.netG.load_state_dict(torch.load(path)["netG_state_dict"])
        self.netD.load_state_dict(torch.load(path)["netD_state_dict"])
        self.optimizerG.load_state_dict(torch.load(path)["optimizerG_state_dict"])
        self.optimizerD.load_state_dict(torch.load(path)["optimizerD_state_dict"])

        print("Loaded state dicts")

    def train(
        self,
        init_epoch,
        num_epochs,
        dataloaders,
        dataloaders_adv,
        gsm_enabled=False,
        verbose=True,
    ):

        G_losses = []
        D_losses = []
        T_losses = []
        P_losses = []
        Losses = []
        img_list_adv = []
        img_list_real = []
        self.iters = 0
        epsilon = 0.4

        for epoch in range(init_epoch, num_epochs):
            for idx, (D, D_adv) in enumerate(
                zip(dataloaders["train"], dataloaders_adv["train"])
            ):
                self.netG.train()
                self.netD.train()

                # Train Discriminator on adversarial image
                self.netD.zero_grad()

                # adversarial images definition
                image_adv = D_adv[0].to(self.device)
                target_labels_adv = D_adv[1].to(self.device)
                batch_size_adv = image_adv.size()[0]
                # label_adv -> target label to use for adversarial images while training discriminator
                label_adv = torch.full(
                    (batch_size_adv,),
                    self.fake_label,
                    dtype=torch.float,
                    device=self.device,
                )

                # real images definition
                image = D[0][:batch_size_adv, ...].to(self.device)
                target_labels = D[1].to(self.device)
                batch_size = image.size()[0]
                # label -> target label to use for real images while training discriminator
                label = torch.full(
                    (batch_size,),
                    self.real_label,
                    dtype=torch.float,
                    device=self.device,
                )

                gaus_noise = torch.randn(*image.size(), device=self.device)
                out_real = self.netD(image + gaus_noise).view(-1)
                loss_d_real = self.criterion(out_real, label)

                loss_d_real.backward()

                # D_x -> output of the discriminator for real images. Between (0,1)
                # D_x = out_real.mean()
                # D_x.backward()
                D_x = out_real.mean().item()

                # Train Discriminator on generated images
                # noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
                noise = get_seed(image, z_size=self.nz)
                if gsm_enabled:
                    generated = torch.sign(self.netG(noise)) * epsilon + image_adv
                else:
                    generated = self.netG(noise) + image_adv
                label_adv = torch.full(
                    (batch_size,),
                    self.fake_label,
                    dtype=torch.float,
                    device=self.device,
                )

                out_generated = self.netD(generated.clone().detach()).view(-1)
                loss_d_generated = self.criterion(out_generated, label_adv.detach())

                loss_d_generated.backward()

                # D_G_z1 -> output of the discriminator for generated images. Between (0,1)
                # D_G_z1 = out_generated.mean()
                # (-1*D_G_z1).backward()
                D_G_z1 = out_generated.mean().item()

                # loss_d -> total loss of discriminator
                loss_d = loss_d_generated + loss_d_real

                # nn.utils.clip_grad_norm_(self.netD.parameters(), 2)

                if idx % 10 == 0:
                    self.optimizerD.step()

                # Train Generator
                self.netG.zero_grad()
                self.netTarget.zero_grad()

                label_adv = torch.full(
                    (batch_size,),
                    self.real_label,
                    dtype=torch.float,
                    device=self.device,
                )

                out_generated_2 = self.netD(generated.clone()).view(-1)
                loss_g = self.criterion(out_generated_2, label_adv)
                loss_g.backward(retain_graph=True)

                # D_G_z2 = out_generated_2.mean()
                # D_G_z2.backward(retain_graph=True)

                # Calculate loss on classifier
                out_classifier = self.netTarget(generated.clone())
                loss_c = self.criterionTarget(out_classifier, target_labels_adv)
                (10 * loss_c).backward(retain_graph=True)

                # Calculate norm of noise generated
                loss_p = self.criterionPerturbation(image, generated)
                loss_p.backward()

                # loss -> final loss
                loss = loss_d + loss_c + loss_g + loss_p
                # loss.backward()
                self.optimizerG.step()
                # if idx%5==0 and idx!=0:
                # update the Discriminator every 5th step
                # self.optimizerD.step()

                # D_G_z2 -> output of the discriminator for generated images. Between (0,1)
                D_G_z2 = out_generated_2.mean().item()
                if self.wandb:
                    self.wandb.log(
                        {
                            "Loss_C": loss_c.item(),
                            "Loss_D": loss_d.item(),
                            "Loss_G": loss_g.item(),
                            "Noise_norm": loss_p.item(),
                            "Real_img_Disc_Dx": D_x,
                            "Generated_img_Disc_DG_z1": D_G_z1,
                            "Generated_img_Disc_after_DG_z2": D_G_z2,
                            "epoch": epoch,
                        }
                    )
                if idx % 50 == 0 and verbose == True:
                    print(
                        "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_C: %.4f\tLoss_P: %.4f\tOverall loss: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                        % (
                            epoch,
                            num_epochs,
                            idx,
                            len(dataloaders["train"]),
                            loss_d.item(),
                            loss_g.item(),
                            loss_c.item(),
                            loss_p.item(),
                            loss.item(),
                            D_x,
                            D_G_z1,
                            D_G_z2,
                        )
                    )

                G_losses.append(loss_g.item())
                D_losses.append(loss_d.item())
                T_losses.append(loss_c.item())
                P_losses.append(loss_p.item())
                Losses.append(loss.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (self.iters % 50 == 0) or (
                    (epoch == num_epochs - 1) and (idx == len(dataloaders["train"]) - 1)
                ):
                    with torch.no_grad():
                        if gsm_enabled:
                            generated = (
                                (
                                    torch.sign(self.netG(self.fixed_noise)) * epsilon
                                    + image_adv[:64]
                                )
                                .detach()
                                .cpu()
                            )
                        else:
                            generated = (
                                (self.netG(get_seed(image_adv[:64])) + image_adv[:64])
                                .detach()
                                .cpu()
                            )
                        img_list_adv.append(
                            vutils.make_grid(generated, padding=2, normalize=True)
                        )
                    # TODO Visualize
                    with torch.no_grad():
                        if gsm_enabled:
                            generated = (
                                (
                                    torch.sign(self.netG(self.fixed_noise)) * epsilon
                                    + image[:64]
                                )
                                .detach()
                                .cpu()
                            )
                        else:
                            generated = (
                                (self.netG(get_seed(image[:64])) + image[:64])
                                .detach()
                                .cpu()
                            )
                        img_list_real.append(
                            vutils.make_grid(generated, padding=2, normalize=True)
                        )
                self.iters += 1

            if epoch % 5 == 0 or epoch == num_epochs - 1 and verbose == True:
                self.visualize_images(
                    img_list_adv, epoch, img_type="Adversarial", wandb=self.wandb
                )
                self.visualize_images(
                    img_list_real, epoch, img_type="Real", wandb=self.wandb
                )

            self.schedulerG.step()
            self.schedulerD.step()

            if epoch % 10 == 0 and verbose == True:
                self.save_state_dicts(
                    f"Linear_BestcounterGAN_{epoch}.pth", wandb=self.wandb
                )
            self.save_state_dicts(f"Linear_last_counterGAN.pth", wandb=self.wandb)
        return D_losses, G_losses, T_losses, img_list_adv, img_list_real

    def visualize_images(self, img_list, epoch=None, img_type="Real", wandb=None):

        if epoch == None:
            epoch = ""

        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Generated Images {}".format(img_type))
        plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
        plt.savefig("./linear_images/Image_{}_{}.png".format(epoch, img_type))
        if wandb:
            wandb.log({img_type: wandb.Image(img_list[-1]), "epoch": epoch})

        return

    def inference(self, images, num_z=5, batch_size=64):
        """
        Parameters:
        images -> input images Tensor(64,3,32,32)
        num_z -> ensemble size

        Returns:
        classifications -> list of classifications of size ensemble size [list of Tensors(64)]
        """
        classifications = []
        orig_classification = None
        self.netG.eval()
        self.netTarget.eval()
        with torch.no_grad():
            orig_classification = (
                torch.argmax(torch.softmax(self.netTarget(images), dim=-1), dim=-1)
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            for i in range(num_z):
                # z = torch.randn(batch_size, 100, 1, 1, device=self.device)
                z = get_seed(images)
                generated = self.netG(z) + images
                classification = torch.argmax(
                    torch.softmax(self.netTarget(generated), dim=-1), dim=-1
                )
                classifications.append(classification.detach().squeeze().cpu().numpy())

        return np.array(classifications), orig_classification


# On adv and real dataset:

# Image, Groundtruth, Prediction, CounterGanPrediction
# img, class, forward_pass_on_image, forward_pass_on_image_with_noise

