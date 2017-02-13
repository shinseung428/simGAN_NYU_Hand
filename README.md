# Simulated+Unsupervised (S+U) learning in TensorFlow
# NYU Hand Dataset

Another TensorFlow implementation of [Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/abs/1612.07828).

Thanks to [TaeHoon Kim](http://carpedm20.github.io), I was able to run simGAN that generates refined synthetic eye dataset.  
This is just another version of his code that can generate [NYU hand datasets](http://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm).


The structure of the refiner/discriminator networks are changed as it is described in the Apple paper.  
The only code added in this version is ./data/hand_data.py.  
Rest of the code runs in the same way as the original version.  
To set up the environment(or to run UnityEyes dataset), please follow instructions in this [link](https://github.com/carpedm20/simulated-unsupervised-tensorflow)

###Notes
-NYU hand dataset is preprocessed(e.g. background removed)
-Image size set to 128x128
-Buffer/Batch size was reduced due to memory issues
-Local adversarial loss not implemented


##Training Result

Given these synthetic images,

![NYU_hand_synt_1](./results/synt_1.png)
![NYU_hand_synt_2](./results/synt_2.png)
![NYU_hand_synt_3](./results/synt_3.png)
![NYU_hand_synt_4](./results/synt_4.png)
![NYU_hand_synt_5](./results/synt_5.png)
![NYU_hand_synt_6](./results/synt_6.png)

###Test 1

'lambda=0.1' with 'optimizer=sgd' after 4k steps.  

![NYU_hand_ref_1](./results/refined_0.1_1.png)
![NYU_hand_ref_2](./results/refined_0.1_2.png)
![NYU_hand_ref_3](./results/refined_0.1_3.png)
![NYU_hand_ref_4](./results/refined_0.1_4.png)
![NYU_hand_ref_5](./results/refined_0.1_5.png)
![NYU_hand_ref_6](./results/refined_0.1_6.png)

![scalar_result_1](./results/scalar_result_1.png)

###Test 2

'lambda=0.5' with 'optimizer=sgd' after 4k steps.  

![NYU_hand_ref_7](./results/refined_0.5_1.png)
![NYU_hand_ref_8](./results/refined_0.5_2.png)
![NYU_hand_ref_9](./results/refined_0.5_3.png)
![NYU_hand_ref_10](./results/refined_0.5_4.png)
![NYU_hand_ref_11](./results/refined_0.5_5.png)
![NYU_hand_ref_12](./results/refined_0.5_6.png)

![scalar_result_2](./results/scalar_result_2.png)

###Test 3

'lambda=1.0' with 'optimizer=sgd' after 4k steps.  

![NYU_hand_ref_13](./results/refined_1.0_1.png)
![NYU_hand_ref_14](./results/refined_1.0_2.png)
![NYU_hand_ref_15](./results/refined_1.0_3.png)
![NYU_hand_ref_16](./results/refined_1.0_4.png)
![NYU_hand_ref_17](./results/refined_1.0_5.png)
![NYU_hand_ref_18](./results/refined_1.0_6.png)

![scalar_result_2](./results/scalar_result_2.png)


## Author

Seung Shin / [@shinseung428](http://shinseung428.github.io)


