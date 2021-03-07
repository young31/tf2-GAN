# GAN code repo

#### 다양한 활용을 위한 레퍼런스 코드 저장소

- GAN과 관련한 논문들의 주요 사용 기술을 구현해본 저장소
- 완전한 모형을 만들기보다 핵심 기술을 구현하여 저장해두는 것을 목표로함
  - ~~노트북으로 작업하기 때문에 딥한 것은 실제로 어렵다..~~
- tf2를 사용하며 model class overriding한 스타일
- 구체적인 내용은 다른 블로그 등에 자세히 설명이 돼 있다.

## Contents

- BiGAN
  - z -> X와 X -> z를 동시에 실행
- BigGAN
  - hierachical latent codes + conditional embedding
- CGAN
  - conditional generation
- D2GAN
  - reverse KLD(p|q and q|p)
- DCGAN
  - use convolutional layers
- DFM
  - add denoiser(denoising AE) in generator
- LOGAN
  - Apply latent optimization on DCGAN
- LSGAN
  - use l2-loss instead of crossentropy loss
- RaGAN
  - use relative loss(prior information that half of data is fake)
- SAGAN
  - self-attention if GAN
- WGAN
  - wasserstein-1 loss instead of crossentropy(dual form and GP)

- ebGAN
  - add energy-based loss