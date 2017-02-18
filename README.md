# Simulated+Unsupervised (S+U) learning in TensorFlow
# NYU Hand Dataset

Another TensorFlow implementation of [Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/abs/1612.07828).

Thanks to [TaeHoon Kim](http://carpedm20.github.io), I was able to run simGAN that generates refined synthetic eye dataset.  
This is just another version of his code that can generate [NYU hand datasets](http://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm).


The structure of the refiner/discriminator networks are changed as it is described in the Apple paper.  
The only code added in this version is ./data/hand_data.py.  
Rest of the code runs in the same way as the original version.  
To set up the environment(or to run UnityEyes dataset), please follow instructions in this [link](https://github.com/carpedm20/simulated-unsupervised-tensorflow).

###Notes
-NYU hand dataset is preprocessed(e.g. background removed)  
-Image size set to 128x128  
-Buffer/Batch size was reduced due to memory issues  
-Changed the size of the refiner/discriminator network

##Results

Given these synthetic images,

![NYU_hand_synt_1](./results/synt_1.png)
![NYU_hand_synt_2](./results/synt_2.png)
![NYU_hand_synt_3](./results/synt_3.png)
![NYU_hand_synt_4](./results/synt_4.png)
![NYU_hand_synt_5](./results/synt_5.png)
![NYU_hand_synt_6](./results/synt_6.png)

###Test 1

'lambda=0.1' with 'optimizer=sgd'  
After 10k steps.  

![NYU_hand_ref_1](./results/refined_1.png)
![NYU_hand_ref_2](./results/refined_2.png)
![NYU_hand_ref_3](./results/refined_3.png)
![NYU_hand_ref_4](./results/refined_4.png)
![NYU_hand_ref_5](./results/refined_5.png)
![NYU_hand_ref_6](./results/refined_6.png)

Discriminator Loss  
![scalar_d_result_1](./results/scalar_discrim_result_1.png)

Refiner Loss  
![scalar_r_result_1](./results/scalar_refine_result_1.png)

###Test 2

'lambda=0.5' with 'optimizer=sgd'  
After ~10k steps.  

![NYU_hand_ref_7](./results/refined_1.1.png)
![NYU_hand_ref_8](./results/refined_2.1.png)
![NYU_hand_ref_9](./results/refined_3.1.png)
![NYU_hand_ref_10](./results/refined_4.1.png)
![NYU_hand_ref_11](./results/refined_5.1.png)
![NYU_hand_ref_12](./results/refined_6.1.png)

![scalar_result_2](./results/scalar_discrim_result_2.png)

![scalar_result_2](./results/scalar_refine_result_2.png)

###Test 3

'lambda=1.0' with 'optimizer=sgd' after 10k steps.  

![NYU_hand_ref_13](./results/refined_1.2.png)
![NYU_hand_ref_14](./results/refined_2.2.png)
![NYU_hand_ref_15](./results/refined_3.2.png)
![NYU_hand_ref_16](./results/refined_4.2.png)
![NYU_hand_ref_17](./results/refined_5.2.png)
![NYU_hand_ref_18](./results/refined_6.2.png)

![scalar_result_3](./results/scalar_discrim_result_3.png)

![scalar_result_3](./results/scalar_refine_result_3.png)

##Summary
Background of the refined images are darker.  
Some of the real image backgrounds were not properly removed while obtaining the arm hand segments from the real dataset.  
When the refiner tries to make refined synthetic images, it also changes the colour of the background to make it look like the ones in the real image dataset.

## Author

Seung Shin / [@shinseung428](http://shinseung428.github.io)


