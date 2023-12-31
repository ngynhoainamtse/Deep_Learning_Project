# Deep_Learning_Project
Fine-tuning Diffusion Model with LoRA on MNIST Dataset

In this notebook, I implement Fine-tuning of the Diffusion Model with Low-Rank Adaptation(LoRA) from scratch on the MNIST dataset using Pytorch. 

1. I first build a CNN with linear layers and train this model to perform on MNIST 0-4 (the first half of the data).
2. I use afterward the standard finetuning approach to train this model to work on MNIST 5-9 (the second half of the data). 
3. I use finally LoRA finetuning to train the original model to work on MNIST 5-9 (the second half of the data).
