# Towards Real-World 6G Drone Communication: Position and Camera Aided Beam Prediction
This is a python code package related to the following article:
G. Charan et al., "[Towards Real-World 6G Drone Communication: Position and Camera Aided Beam Prediction](https://ieeexplore.ieee.org/document/10000718),", in 2022 IEEE Global Communications Conference (GLOBECOM), 2022, pp. 2951-2956

# Abstract of the Article
Millimeter-wave (mmWave) and terahertz (THz) communication systems typically deploy large antenna arrays to guarantee sufficient receive signal power. The beam training overhead associated with these arrays, however, make it hard for these systems to support highly-mobile applications such as drone communication. To overcome this challenge, this paper proposes a machine learning based approach that leverages additional sensory data, such as visual and positional data, for fast and accurate mmWave/THz beam prediction. The developed framework is evaluated on a real-world multi-modal mmWave drone communication dataset comprising co-existing camera, practical GPS, and mmWave beam training data. The proposed sensing-aided solution achieves a top-1 beam prediction accuracy of 86.32% and close to 100% top-3 and top-5 accuracies, while considerably reducing the beam training overhead. This highlights a promising solution for enabling highly-mobile 6G drone communications.

# Code Package Content 
The scripts for generating the results of the ML solutions in the paper. This script adopts Scenario 23 of DeepSense6G dataset.

**To reproduce the results, please follow these steps:**

**Download Dataset and Code**
1. Download [the sensing aided drone beam prediction dataset of DeepSense 6G/Scenario 23](https://deepsense6g.net/scenario-23/).
2. Download (or clone) the repository into a directory.
3. Extract the dataset into the repository directory

**Generate Development Dataset**
1. Run downsample_pwr_gen_train_test_data.py to generate the train/val/test set. It provides an option to use both the original beams and the downsamples beams.
2. To generate the dataset with original (64) beams, update lines 94 and 155 to "df_new['unit1_beam'] = original_beam".
3. This code generates the development dataset for both vision and position-aided beam prediction and save in their respective folders.

**ML Model Training**
1. Run main_beam.py file under image_beam or pos_beam folder to train the respective models
2. This code will train the ML model, save the checkpoint and analysis files

**ML model Evaluation**
1. Run main_beam_eval.py file



If you have any questions regarding the code and used dataset, please write to DeepSense 6G dataset forum https://deepsense6g.net/forum/ or contact [Gouranga Charan](mailto:gcharan@asu.edu?subject=[GitHub]%20Beam%20prediction%20implementation).

# License and Referencing
This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). 
If you in any way use this code for research that results in publications, please cite our original article:
> G. Charan et al., "Towards Real-World 6G Drone Communication: Position and Camera Aided Beam Prediction," GLOBECOM 2022 - 2022 IEEE Global Communications Conference, Rio de Janeiro, Brazil, 2022, pp. 2951-2956, doi: 10.1109/GLOBECOM48099.2022.10000718.

If you use the [DeepSense 6G dataset](www.deepsense6g.net), please also cite our dataset article:
> A. Alkhateeb, G. Charan, T. Osman, A. Hredzak, J. Morais, U. Demirhan and N. Srinivas, “DeepSense 6G: large-scale real-world multi-modal sensing and communication datasets,” to be available on arXiv, 2022. [Online]. Available: https://www.DeepSense6G.net
