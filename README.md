# Adversarial Defense Using Generative Adversarial Networks

Adversarial attacks are imperceptible perturbations, carefully chosen by an attacker, that make neural networks fail. They are a significant risk for deep learning models deployed in production. We developed counterGAN, a Generative Adversarial Network (GAN) to defend against such an attack by countering the attacker. counterGAN learns the distribution of the underlying noise added by the attacker, allowing us to generate a cancelling noise that can neutralise the effects of the attack. The detailed description of our approaches, inspirations and results can be found in our [project report](https://drive.google.com/file/d/1eCfUROOFICeE3ULkTs8NUIx5HpHWqceQ/view?usp=sharing).

<div align="center"><img src="https://miro.medium.com/max/4800/1*PmCgcjO3sr3CPPaCpy5Fgw.png" width="600" height="300"></div>
<div align="center">Source: Explaining and Harnessing Adversarial Examples, Goodfellow et al, ICLR 2015.</div>

## Technology Stack

* **Language and Frameworks**: Python, Pytorch
* **Visualization**: [Weights & Biases](https://wandb.ai/site)
* **Computing Platform**: AWS EC2

<div align="center"><img src="https://github.com/Rive-001/counterGAN/blob/main/counterGAN.png" width="800" height="400"></div>
<div align="center">counterGAN architecture</div>

## Experiments

* Convolutional layers in Generator and Discriminator
* Fully connected layers in Generator and Discriminator
* Losses: Binary Cross-entropy loss, Mean squared-error loss, Wasserstein loss

## Results

| | Linear+BCE | Linear+MSE | Conv+MSE | Conv+BCE | Linear+WGAN-GP |
| --- | --- | --- | --- | --- | --- |
| Adv. classifier accuracy | 1.03% | 1.03% | 1.03% | 1.03% | 1.03% |
| Adv. defense accuracy | 14.88% | 11.87% | 13.03% | 12.47% | 10.70% |

## Conclusion

We achieved the best results by using fully connected layers with binary cross-entropy loss. There are also quite a few other experiments performed to solve the issue of mode collapse, details of which could be found in our [project report](https://drive.google.com/file/d/1eCfUROOFICeE3ULkTs8NUIx5HpHWqceQ/view?usp=sharing). Our initial success goes to show that the concept is viable and that better results could be obtained by further experimentation. We were also able to achieve a better inference runtime as compared to a popular solution in this field, [DefenseGAN](https://arxiv.org/abs/1805.06605). This was due to the design of the inference step in DefenseGAN, that performs multiple iterations of gradient descent during inference. Our model had a better inference time as we required only a single pass through our model.

## Team

* [Rishi Verma](https://github.com/Rive-001)
* [Sanil Pande](https://github.com/sanilpande)
* [Vidhi Upadhyay](https://github.com/VidhiUpadhyay)
* [Tarang Shah](https://github.com/t27)
