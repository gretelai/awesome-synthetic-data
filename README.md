# awesome-synthetic-data

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of resources dedicated to Synthetic Data

_If you want to contribute to this list, read the [contribution guidelines](contributing.md) first. Please add your favourite synthetic data resource by raising a [pull request](https://github.com/gretelai/awesome-synthetic-data/pulls)_

Also, a listed repository should be deprecated if:
* Repository's owner explicitly says that "this library is not maintained".
* Not committed for a long time (2~3 years).


## Contents

* [Research Summaries and Trends](#research-summaries-and-trends)
* [Tutorials](#tutorials)
  * [Reading Content](#reading-content)
  * [Videos and Online Courses](#videos-and-online-courses)
* [Libraries](#libraries)
  * [Text](#text-tabular-and-time-series)
  * [Image](#image)
  * [Audio](#audio)
  * [Simulation](#simulation)
* [Academic Papers](#academic-papers)
  * [Language Models](#language-models)
  * [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
  * [Diffusion Models](#diffusion-models)
  * [Fair AI](#fair-ai)
  * [Algorithmic Privacy](#algorithmic-privacy)
* [Services](#services)
* [Prominent Synthetic Data Research Labs](#prominent-synthetic-data-research-labs)
* [Datasets](#datasets)

## Research Summaries and Trends
[Back to Top](#contents)

## Tutorials
[Back to Top](#contents)

### Reading Content
[Back to Top](#contents)

Introductions and Guides to Synthetic Data

Blogs and Newsletters
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) - Andrej Karpathy's intro to RNNs.
* [Annotated Diffusion](https://huggingface.co/blog/annotated-diffusion) - Tutorial on original diffusion model paper with code

Videos and Online Courses

### Videos and Online Courses
[Back to Top](#contents)

### Diffusion Models
* [Learning to Generate Data by Estimating Gradients of the Data Distribution](https://youtu.be/nv-WTeKRLl0) - Video by Yang Song from Stanford. Excellent theory and interesting applications.

## Libraries
Open Source Generative Synthetic Data Models, Libraries and Frameworks | [Back to Top](#contents)

### Text, Tabular and Time-Series

* [gretel-synthetics](https://github.com/gretelai/gretel-synthetics) - Generative models for structured and unstructured text, tabular, and multi-variate time-series data featuring differentially private learning.
* [SDV](https://github.com/sdv-dev/SDV) - Synthetic Data Generator for tabular, relational, and time series data.
* [Synthea](https://github.sre.pub/synthetichealth/synthea) - Synthetic Patient Population Simulator.
* [ydata-synthetic](https://github.com/ydataai/ydata-synthetic) - Synthetic structured data generators.
* [synthpop](https://cran.r-project.org/web/packages/synthpop/index.html) - A tool for producing synthetic versions of microdata.


### Image
* [Contrastive Unpaired Translation](https://github.com/taesungp/contrastive-unpaired-translation) - Contrastive unpaired image-to-image translation, faster and lighter training than cyclegan.
* [StyleGAN 3](https://github.com/NVlabs/stylegan3) - Official PyTorch implementation of StyleGAN3 from NeurIPS 2021.
* [Denoising Diffusion Pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) - Implementation of DDPM

### Audio
* [Jukebox](https://github.com/openai/jukebox/) - OpenAI's Jukebox- A Generative Model for Music.

### Simulation
* [AirSim](https://microsoft.github.io/AirSim/) - AirSim is a simulator for drones, cars and more, built on Unreal and Unity engines.
* [Nvidia Dataset Synthesizer](https://github.com/NVIDIA/Dataset_Synthesizer) - NDDS is a UE4 plugin from NVIDIA to empower computer vision researchers to export high-quality synthetic images with metadata.
* [OpenAI Gym](https://github.com/openai/gym) - A toolkit for developing and comparing reinforcement learning algorithms.
* [Unity Perception](https://github.com/Unity-Technologies/com.unity.perception) Perception toolkit for sim2real training and validation in Unity.

### Video
* [Video Diffusion Pytorch](https://github.com/lucidrains/video-diffusion-pytorch) - Implementation of video diffusion models in pytorch

## Academic Papers
[Back to Top](#contents)

### Language Models
* **Evaluating Large Language Models Trained on Code** (2021) March Chen et al. [[pdf]](https://arxiv.org/pdf/2107.03374.pdf)

### Generative Adversarial Networks (GANs)
* **Modeling Tabular Data using Conditional GAN** (2019) Xu et al. [[pdf]](https://arxiv.org/pdf/1907.00503.pdf)
* **Generating Long Videos of Dynamic Scenes** (2022) Tim Brooks [[pdf]](https://arxiv.org/pdf/2206.03429.pdf)
* **Generative Adversarial Networks** (2014) Ian J. Goodfellow et al. [[pdf]](https://arxiv.org/abs/1406.2661)
* **Conditional Generative Adversarial Nets** (2014) Mehdi Mirza et al. [[pdf]](https://arxiv.org/abs/1411.1784)
* **Modeling Tabular Data using Conditional GAN** (2019) Xu et al. [[pdf]](https://arxiv.org/pdf/1907.00503.pdf)
* **Wasserstein GAN** (2017) Martin Arjovsky, et al.[[pdf]](https://arxiv.org/abs/1701.07875)
* **Improved Training of Wasserstein GANs** (2017) Ishaan Gulrajani, et al. [[pdf]](https://arxiv.org/abs/1704.00028)
* **Time-series Generative Adversarial Networks** (2019) Jinsung Yoon, et all [[pdf]](https://proceedings.neurips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf)

### Diffusion Models
* **Generative Modeling by Estimating Gradients of the Data Distribution** (2021) Yang Song [[pdf]](https://yang-song.github.io/blog/2021/score/)
* **Diffusion Models are Autoencoders** S. Dielman (2021) [[pdf]](https://benanne.github.io/2022/01/31/diffusion.html)
* **Deep Unsupervised Learning using Nonequilibrium Thermodynamics** (2015) J Sohl-Dickstein et al. [[pdf]](https://arxiv.org/pdf/1503.03585.pdf)
* **KNN-Diffusion: Image Generation via Large-Scale Retrieval** (2022) Oron Ashual [[pdf]](https://arxiv.org/pdf/2204.02849.pdf)

### Fair AI

* **A Framework for Understanding Sources of Harm throughout the Machine Learning Life Cycle** (2021) Harini Suresh, John Guttag [[pdf]](https://arxiv.org/pdf/1901.10002.pdf)
* **DECAF: Generating Fair Synthetic Data Using Causally-Aware Generative Networks** (2021) Boris van Breugel et al [[pdf]](https://openreview.net/forum?id=XN1M27T6uux)
* **On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?** (2021) Emily M. Bender, et al. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)
* **A Survey on Bias and Fairness in Machine Learning** (2022) Ninareh Mehrabi [[pdf]](https://arxiv.org/pdf/1908.09635.pdf)
* **AI Fairness (Approaches & Mathematical Definitions)** (2022) Jonathan Hui [[blog]](https://jonathan-hui.medium.com/ai-fairness-approaches-mathematical-definitions-49cc418feebd)
* **AI Fairness 360: An Extensible Toolkit for Detecting, Understanding, and Mitigating Unwanted Algorithmic Bias** (2018) Rachel K. E. Bellamy et al [[pdf]](https://arxiv.org/pdf/1810.01943.pdf)

### Algorithmic Privacy

* **Deep Learning with Differential Privacy** (2016) Abadi et al. [[pdf]](https://arxiv.org/pdf/1607.00133.pdf)
* **An Efficient DP-SGD Mechanism for Large Scale NLP Models** (2021) Dupuy et al. [[pdf]](https://arxiv.org/pdf/2107.14586.pdf)
* **PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees** (2018) Jordon et al. [[pdf]](https://openreview.net/pdf?id=S1zk9iRqF7)
* **Don't Generate Me: Training Differentially Private Generative Models with Sinkhorn Divergence** (2021) Cao et al. [[pdf]](https://arxiv.org/pdf/2111.01177.pdf)
* **Differentially Private Fine-tuning of Language Models** (2022) Yu et al. [[pdf]](https://openreview.net/pdf?id=Q42f0dfjECO)

## Services
Synthetic Data as API with higher level functionality such model training, fine-tuning, and generation | [Back to Top](#contents)
* [List of Synthetic Data Startups in 2021](https://elise-deux.medium.com/the-list-of-synthetic-data-companies-2021-5aa246265b42) - Not all of these necessarily have APIs. 

## Prominent Synthetic Data Research Labs
[Back to Top](#contents)

## Datasets
[Back to Top](#contents)
* [HuggingFace Datasets](https://huggingface.co/docs/datasets/index) - Library for easily accessing and sharing datasets, and evaluation metrics for Natural Language Processing (NLP), computer vision, and audio tasks.
* [Google Cloud Public Datasets](https://cloud.google.com/datasets) - Publicly available and free machine learning and analytics datasets.
* [Kaggle Datasets](https://www.kaggle.com/datasets) - Data science and machine learning datasets.
* [/r/datasets](https://www.reddit.com/r/datasets/) - A place to share, find, and discuss Datasets.
* [Papers with Code - Datasets](https://paperswithcode.com/datasets) - The mission of Papers with Code is to create a free and open resource with Machine Learning papers, code, datasets, methods and evaluation tables.
* [Awesome Public Datasets](https://github.com/awesomedata/awesome-public-datasets) - Topic centric, high quality, public data sources
* [Data.gov](http://data.gov) - U.S. Government's open data
* [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) - Popular datasets used by the machine learning community
* [Google Research Dataset Search](https://datasetsearch.research.google.com/) - Discover datasets hosted in thousands of repositories across the web

## License
[License](./LICENSE) - CC0
