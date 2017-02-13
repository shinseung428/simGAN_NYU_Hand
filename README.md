# Simulated+Unsupervised (S+U) learning in TensorFlow
# NYU Hand Dataset

Another TensorFlow implementation of [Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/abs/1612.07828).

![model](./assets/SimGAN.png)

Thanks to [TaeHoon Kim](http://carpedm20.github.io), I was able to run simGAN that generated refined synthetic eye dataset.
This is just another version of his code that can generate NYU hand datasets.


The structure of the network is changed as it is described in the Apple paper. 
The only code added in this version is ./data/hand_data.py. 
Rest of the code runs in the same way as the original version. 
To set up the environment(or to run UnityEyes dataset), please follow instructions in this [link](https://github.com/carpedm20/simulated-unsupervised-tensorflow).


##Training Result

Given these synthetic images,

![NYU_hand1](./results/synt_1.png)
![NYU_hand2](./results/synt_2.png)
![NYU_hand3](./results/synt_3.png)
![NYU_hand4](./results/synt_4.png)

Result of 'lambda=0.1' with 'optimizer=sgd' after 10k steps.

![NYU_hand5](./results/refined_1.png)
![NYU_hand6](./results/refined_2.png)
![NYU_hand7](./results/refined_3.png)
![NYU_hand8](./results/refined_4.png)

![scalar_result_1](./results/scalar_result_1.png)




## Author

Seung Shin / [@shinseung428](http://shinseung428.github.io)
Taehoon Kim / [@carpedm20](http://carpedm20.github.io)

