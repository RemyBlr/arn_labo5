# PW5 - part 1

## Different results
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

### Best configuration
The last configuration is clearly the best one.


