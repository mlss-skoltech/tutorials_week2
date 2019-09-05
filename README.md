# MLSS 2019 Skoltech tutorials - <b>week 2</b>
This is the official repository for Machine Learning Summer School 2019, which is taking place at Skoltech Institute of Science and Technology, Moscow, from 26.08 - 06.09.

This repository will contain all of the materials needed for MLSS tutorials for the second week (02.09 - 06.09). 

## The list of the current tutorials published (will be updated with time):
* DAY-6 (02.09): Alexnader Notchenko, Michael Bronstein- Graph Neural Networks: https://github.com/mlss-skoltech/tutorials_week2/tree/master/graph_neural_networks
* DAY-6 (02.09): Alexander Korotin, Matus Jan Telgarsky - Deep Learning Theory: https://github.com/mlss-skoltech/tutorials_week2/tree/master/deep_learning_theory
* DAY-7 (03.09): Nina Mazyavkina, Shimon Whiteson - Reinforcement Learning: https://github.com/mlss-skoltech/tutorials_week2/tree/master/reinforcement_learning
* DAY-7 (03.09): Justin Solomon - Geometric Techniques in ML: https://github.com/mlss-skoltech/tutorials_week2/tree/master/geometric_techniques_in_ML2
* DAY-8 (04.09): Rodrigo Rivera Castro, Dmitry Oganesyan, Alexander Korotin, Ulrich Bauer - Topological Data Analysis: https://github.com/mlss-skoltech/tutorials_week2/tree/master/topological_data_analysis
* DAY-8 (04.09): Alexey Artemov, Michael Bronstein - Graph Neural Networks: https://github.com/mlss-skoltech/tutorials_week2/tree/master/graph_neural_networks2
* DAY-9 (05.09): Rodrigo Rivera Castro, Dmitry Oganesyan, Alexander Korotin, Ulrich Bauer - Topological Data Analysis: https://github.com/mlss-skoltech/tutorials_week2/tree/master/topological_data_analysis
* DAY-9 (05.09): Nina Mazyavkina, Shimon Whiteson - Reinforcement Learning: https://github.com/mlss-skoltech/tutorials_week2/tree/master/reinforcement_learning2




# Running the tutorials on Google Colaboratory:
Most of the tutorials were created using Jupyter Notebooks. In order to reduce the time spent on installing various software, we have made sure that all of the tutorials are Google Colaboratory friendly. 

Colaboratory is a free Jupyter notebook environment that requires no setup and runs entirely in the cloud. With Colaboratory you can write and execute code, save and share your analyses, and access powerful computing resources, all for free from your browser. All of the notebooks already contain all the set-ups needed for each particular tutorial, so you will just be required to run the first several cells.

Here are the instructions on how open the notebooks in Colaboratory (tested on Google Chrome, version 76.0.):
* First go to https://colab.research.google.com/github/mlss-skoltech/
* In the pop-up window, sign-in into your GitHub account 
![image0](/img/img0.png)
* In the opened window, choose the notebook correspodning to the tutorial 
![image1](/img/img1.png)
* The selected notebook will open, now make sure that you are signed-in into your Google account
![image2](/img/img2.png)
* Try to run the first cell, you will get the following message:
![image3](/img/img3.png)
Press ```RUN ANYWAY```
* For the message ```Reset all runtimes``` press ```YES```
![image4](/img/img4.png)

In order to download all the material for the tutorial, make sure you run the cells containing the following code first (all of these cells are already added to the notebooks with the right paths):
* For downloading the github subdirectory containing the tutorial:

```!pip install --upgrade git+https://github.com/mlss-skoltech/tutorials.git#subdirectory=<name of tutorial subdirectory>```

* For declaring the data files' path: 
```
import pkg_resources
DATA_PATH = pkg_resources.resource_filename('name_of_the_installed_tutorial_package', 'data/')
```
# Using GPU with Google Colaboratory:
Sometimes for computationally hard tasks you will be required to use GPU instead of default CPU, in order to do this follow these steps:
* Go to ```Edit->Notebook Settings```
![image5](/img/img5.png)
* In the ```Hardware accelerator``` field choose ```GPU```
![image6](/img/img6.png)
![image7](/img/img7.png)

# Saving and downloading the notebooks
You can save your notebook in your Google Drive or simply download it, for that go to ```File->Save a copy in Drive``` or ```File->Download.ipynb```.
![image8](/img/img8.png)



If you would like to see more tutorials regarding Google Colaboratory have a look at this notebook: https://colab.research.google.com/notebooks/welcome.ipynb

# Contact 
If you have any questions/suggestions regarding this githup repository or have found any bugs, please write to me at N.Mazyavkina@skoltech.ru 

