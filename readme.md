## Original Source Code
Some parts of the code and trained face identification model are from [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe) repository which is released under the [MIT License](https://github.com/ZhaoJ9014/face.evoLVe/blob/master/LICENSE).
Some part of the code is also taken from the [this](the [github repo](https://github.com/spmallick/learnopencv/tree/master/Face-Recognition-with-ArcFace) github repo
## Installation
Install all dependencies
```sh
pip install -r requirements.txt
```
Although the gpu is not required to run all code (some may be), it will take less time if you run over gpu.

## Data preprocessing
### Data
The images should be arranged in this way: <br>
        root/dog/xxy.png <br>
        root/dog/[...]/xxz.png <br>
To prepare data with cropped and aligned faces from your original images, run: <br>
```sh
python face_alignment.py --tags images --crop_size 112
```
where _images_ is the folder name inside the data folder. <br>
_Note: crop_size argument must be either 112 or 224_

### Rename the file names
To rename morphed image names with respect to the subject name in the database, run: <br>
```sh
python rename_file.py
```

### Similarity visualization
To visualize similarity between faces using ArcFace, as shown in the Figure 10 in report, run the following: <br>
```sh
python similarity.py --tags images
```
### Cosine Similarity Score
The morphed images should be given the following names: 1-vs-2.png. and t he genuine images should be named like 1.JPG <br>
Run the following files after providing the respective filenames respectively to calculate the cosine score of genuine, morphed and imposter scores<br>
```sh
python cosine_similarity_genuine.py 
python cosine_similarity_morphed_attack.py
python cosine_similarity_imposter.py
```
### Distribution Plot
You must provide a csv file of cosine of angle for genuine, morphed attack, and impostor for distribution plots. <br>
For the distribution plot of genuine, morphed attack and impostor, run: <br>
```sh
python distribution_plot.py
```
### Test train split
Split the dataset such that there are 60/40 train test ratio. The split is done in such a way that no image in the train set can be used as a test set. <br>
To split the dataset, provide the respective path and run the following:
```sh
python train_test_split.py
```
### Train the model and predict 
To train the dataset with the model, select the network name and run the following: <br>
```sh
python fp_all_baseline_all.py
python fp_pad_baseline_inference.py
```
### APCER, BPCER ROC curve
To plot the DET curve run the following after training the model and predicting the labels <br>
```sh
python generate_apcer_bpcer_plots.py
```
