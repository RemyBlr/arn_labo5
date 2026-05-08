# PW5 - part 1

## Intro
L'objectif de ce TP est d'utiliser le transfer learning pour classifier trois espèces d'oiseaux : Mésange noire, Mésange charbonnière et Moineau. Le modèle de base utilisé est MobileNetV2, pré-entraîné sur ImageNet (1000 classes). Les couches de classification sont remplacées et entraînées sur un petit dataset collecté via Bing (50 images par classe).

### Different results
The basic configuration is used and every time i modified a parameter a took a new screenshot of the graph and the confusion matrix.

### 1 hidden layer, 256 neurons
<img src="../assets/part1/kfold_1_hl_256_p.PNG" width="400"/>
<img src="../assets/part1/matrix_1_hl_256_p.PNG" width="400"/>  

### 1 hidden layer, 128 neurons, freeze
<img src="../assets/part1/kfold_1_hl_128_p_f.PNG" width="400"/>
<img src="../assets/part1/matrix_1_hl_128_p_f.PNG" width="400"/>  

### 2 hidden layers, 128 neurons each, freeze
<img src="../assets/part1/kfold_2_hl_128_p_f.PNG" width="400"/>
<img src="../assets/part1/matrix_2_hl_128_p_f.PNG" width="400"/> 

### 2 hidden layers, 128 neurons each, freeze, 0.3 dropout
<img src="../assets/part1/kfold_2_hl_128_p_f_03_d.PNG" width="400"/>
<img src="../assets/part1/matrix_2_hl_128_p_f_03_d.PNG" width="400"/>

### 2 hidden layers, 128 neurons each, freeze, 0.3 dropout, 0.0001 learning rate
<img src="../assets/part1/kfold_2_hl_128_p_f_03_d_-4_lr.PNG" width="400"/>
<img src="../assets/part1/matrix_2_hl_128_p_f_03_d_-4_lr.PNG" width="400"/>  

### 2 hidden layers, 128 neurons each, freeze, 0.3 dropout, 0.00001 learning rate
<img src="../assets/part1/kfold_2_hl_128_p_f_03_d_-5_lr.PNG" width="400"/>
<img src="../assets/part1/matrix_2_hl_128_p_f_03_d_-5_lr.PNG" width="400"/>

### 2 hidden layers, 128 neurons each, freeze, 0.3 dropout, 0.0001 learning rate, 15 epochs
<img src="../assets/part1/kfold_2_hl_128_p_f_03_d_-4_lr_15_e.PNG" width="400"/>
<img src="../assets/part1/matrix_2_hl_128_p_f_03_d_-4_lr_15_e.PNG" width="400"/>

### 2 hidden layers, 128 neurons each, freeze, 0.3 dropout, 0.0001 learning rate, 10 epochs, 16 batch size
<img src="../assets/part1/kfold_2_hl_128_p_f_03_d_-4_lr_10_e_16_bs.PNG" width="400"/>
<img src="../assets/part1/matrix_2_hl_128_p_f_03_d_-4_lr_10_e_16_bs.PNG" width="400"/>

### 2 hidden layers, 128 neurons each, freeze, 0.3 dropout, 0.0001 learning rate, 15 epochs, 16 batch size, early stopping
<img src="../assets/part1/kfold_2_hl_128_p_f_03_d_-4_lr_15_e_16_bs_es.PNG" width="400"/>
<img src="../assets/part1/matrix_2_hl_128_p_f_03_d_-4_lr_15_e_16_bs_es.PNG" width="400"/>

### 2 hidden layers, 128 neurons each, freeze, 0.3 dropout, 0.0001 learning rate, 15 epochs, 16 batch size, early stopping, data augmentation
<img src="../assets/part1/kfold_2_hl_128_p_f_03_d_-4_lr_15_e_16_bs_es_da.PNG" width="400"/>
<img src="../assets/part1/matrix_2_hl_128_p_f_03_d_-4_lr_15_e_16_bs_es_da.PNG" width="400"/>

## Conclusion
La configuration finale combine : 2 couches denses (128 neurones), dropout 0.3, learning rate 1e-4, batch size 16, 15 epochs maximum avec early stopping (patience=2), et data augmentation (flip horizontal + rotation). Cette configuration atteint une val accuracy stable entre 90 et 100% sur la validation croisée 5-fold, avec un bon équilibre entre biais et variance. Le facteur le plus impactant s'est avéré être la data augmentation, particulièrement importante étant donné la petite taille du dataset (50 images par classe).