# Deep Learning Practicals – Viva Questions and Answers

This guide is based on the Jupyter notebooks contained in the uploaded `DL.zip` archive. Practicals are arranged by practical number and identified by their exact notebook filenames. Practical 14 has two notebook versions, shown separately as 14A and 14B. Numbers not present in the archive are not invented.

> **Note:** Some notebooks contain inconsistencies between their dataset and output-layer size. These are called out as code observations rather than silently corrected. Always describe what the notebook actually contains in your viva, and mention the correction if asked.

## Quick revision: common concepts

- **Epoch:** One complete pass through the training dataset.
- **Batch size:** Number of examples processed before a weight update.
- **Learning rate:** Controls the size of optimizer updates.
- **ReLU:** Activation that outputs `max(0, x)`.
- **Softmax:** Converts class scores into a probability distribution.
- **Normalization:** Rescales input values, for example pixel values from 0–255 to 0–1.
- **Overfitting:** Good performance on training data but weaker performance on unseen data.
- **Validation set:** Data used to monitor/tune a model, not to update its weights.
- **Test set:** Held-out data used for final performance estimation.
- **CNN:** Neural network that learns spatial features using convolution filters.
- **FNN:** Feedforward network in which information moves from input to output through layers.
- **Autoencoder:** Network trained to reconstruct its input.
- **CBOW:** Word2Vec-style method that predicts a target word from surrounding context words.

---

## Practical 1 – `Copy_of_DL_1practicalFNN.ipynb`
### FNN for Sign Language MNIST

**Q1. What is the aim?**
Train a feedforward neural network to classify Sign Language MNIST hand-gesture images into 24 encoded classes.

**Q2. What is the input representation?**
Each 28×28 grayscale image is normalized and flattened to 784 pixel values for Dense layers.

**Q3. Why map the labels?**
The original labels omit some alphabet classes; the code maps the observed classes to consecutive indices so they match the output neurons.

**Q4. What is the network architecture?**
Dense layers with 256, 128 and 64 ReLU units, followed by 24 Softmax outputs.

**Q5. Why sparse categorical crossentropy?**
The targets are integer class indices, not one-hot encoded vectors.

**Q6. What are the training settings?**
The notebook uses Adam, 15 epochs and batch size 128, with a validation split made using train_test_split.

**Q7. How is the result evaluated?**
The model is evaluated on the held-out test arrays and accuracy is reported.

---

## Practical 2 – `DL_2Practical_FNN.ipynb`
### FNN for Sign Language MNIST

**Q1. What is the objective?**
Build and evaluate an FNN for 24-class Sign Language MNIST recognition.

**Q2. What does drop(label, axis=1) do?**
It removes the target column from the pixel feature table; the label column is kept separately as y.

**Q3. Why use train_test_split?**
To create training and validation subsets from the training data.

**Q4. Why stratify the split?**
It approximately preserves the proportion of each class in both subsets.

**Q5. Describe the model.**
The network has 784 input features, Dense layers of 256, 128 and 64 ReLU neurons, and 24 Softmax output neurons.

**Q6. Why use sparse_categorical_crossentropy?**
Because the class targets are integer IDs.

**Q7. What do epochs and batch size mean here?**
15 epochs means 15 passes through training data; batch size 128 means 128 samples are processed per update.

**Q8. What does argmax do?**
It selects the index of the largest predicted class score for each example.

---

## Practical 3 – `DL_Pract_3FNN.ipynb`
### FNN for Sign Language MNIST

**Q1. What is the task?**
Classify static Sign Language MNIST images using a feedforward neural network.

**Q2. Why are pixels divided by 255?**
To scale 8-bit pixel intensities from 0–255 to the 0–1 range.

**Q3. Why reshape to (-1, 784)?**
A 28×28 image becomes a 784-element vector; -1 lets NumPy infer the number of samples.

**Q4. What is the model?**
Dense layers with 256, 128 and 64 ReLU units and a 24-unit Softmax output.

**Q5. Why use stratified validation splitting?**
To reserve validation examples while keeping class proportions approximately consistent.

**Q6. What does chr(classes[label] + 65) do?**
It converts the original class index into a letter code for displaying the alphabet label.

**Q7. Why use a grayscale colormap?**
The input images have one intensity channel, so cmap=gray displays them appropriately.

---

## Practical 4 – `DL_pract_4FNN.ipynb`
### FNN for Sign Language MNIST with graphs

**Q1. What is different from the previous FNN notebook?**
This version uses two hidden Dense layers, 128 and 64 neurons, rather than three hidden layers.

**Q2. What is the input size?**
784 features, corresponding to the 28×28 pixels of each image.

**Q3. Why are there 24 output units?**
The notebook maps the observed Sign Language classes to 24 output categories.

**Q4. Why call model.summary()?**
To inspect layer output shapes and trainable parameter counts.

**Q5. What do the accuracy plots show?**
Training and validation accuracy across epochs.

**Q6. What do the loss plots show?**
Training and validation loss across epochs; divergence can indicate overfitting.

**Q7. How are sample predictions displayed?**
The code predicts a few test samples, maps indices back to letters and compares predicted with actual letters.

---

## Practical 5 – `DL_5.ipynb`
### FNN for Sign Language MNIST

**Q1. What is the aim?**
Train a Dense feedforward network to classify Sign Language MNIST images.

**Q2. What does X contain and what does y contain?**
X contains pixel features; y contains the corresponding class labels.

**Q3. Why normalize pixel values?**
It makes the input scale smaller and usually helps optimization.

**Q4. What is the architecture?**
Three hidden Dense layers with 256, 128 and 64 ReLU units and a 24-class Softmax output.

**Q5. Why use Adam?**
Adam adaptively adjusts parameter update sizes using estimates of gradients and their moving averages.

**Q6. What is validation data used for?**
It monitors generalization during training without updating weights from those examples.

**Q7. How do you obtain test accuracy?**
Call model.evaluate on X_test and y_test and read the accuracy metric.

---

## Practical 6 – `DL_6.ipynb`
### CNN for Sign Language MNIST

**Q1. What is a CNN?**
A convolutional neural network learns local spatial patterns in images using shared filters.

**Q2. Why reshape images to 28×28×1?**
CNNs expect height, width and channel dimensions; grayscale images have one channel.

**Q3. What does Conv2D do?**
It applies learnable filters across the image to produce feature maps.

**Q4. What does MaxPooling2D do?**
It downsamples feature maps by retaining the maximum value in each pooling window.

**Q5. Why use Flatten?**
It converts the final feature maps into a vector for Dense layers.

**Q6. What is the output layer?**
The notebook uses 24 Softmax outputs for the 24 mapped Sign Language classes.

**Q7. Why sparse categorical crossentropy?**
The target labels are integer class indices.

---

## Practical 7 – `DL_7.ipynb`
### CNN for Sign Language MNIST with predictions

**Q1. What is the practical objective?**
Train a CNN for Sign Language recognition, evaluate it and show actual versus predicted examples.

**Q2. Why split training data into train and validation?**
Validation data measures performance on examples not used for weight updates.

**Q3. What does random_state=42 do?**
It makes the random split reproducible.

**Q4. Why use stratify?**
It keeps class proportions approximately similar across the split.

**Q5. What does model.predict return?**
A score or probability for every output class for each input image.

**Q6. How are predicted class IDs obtained?**
np.argmax(predictions, axis=1) selects the largest output for each sample.

**Q7. Why map IDs to label_names?**
It makes numeric predictions readable as alphabet labels.

---

## Practical 8 – `DL_8.ipynb`
### CNN for Sign Language MNIST and loss graph

**Q1. What data files are loaded?**
The notebook reads sign_mnist_train.csv and sign_mnist_test.csv.

**Q2. How are features and targets separated?**
The label column is removed to form X; the label column itself forms y.

**Q3. Why map labels to consecutive integers?**
The observed labels are converted to indices suitable for the network output.

**Q4. Why reshape to 28×28×1?**
To provide height, width and one grayscale channel to Conv2D.

**Q5. Describe the CNN.**
It uses convolution and max-pooling stages followed by Flatten and Dense classification layers.

**Q6. What is the role of validation_split?**
It reserves a portion of training data for validation during fitting.

**Q7. What do training and test accuracy represent?**
Training accuracy is measured on training examples; test accuracy is measured on the held-out test set.

---

## Practical 9 – `DL_9.ipynb`
### CNN for Sign Language recognition and sample display

**Q1. What is the input dataset?**
Sign Language MNIST training and testing CSV files.

**Q2. Why are J and Z absent from the static alphabet labels?**
Those gestures involve movement, while this dataset represents static hand positions.

**Q3. What does np.unique(y_train) do?**
It finds the distinct class labels present in the training labels.

**Q4. Why normalize and reshape?**
Normalization scales pixels to 0–1; reshaping makes each image 28×28×1 for the CNN.

**Q5. What does argmax(axis=1) do?**
It selects the predicted class index from each row of class scores.

**Q6. Why show actual and predicted labels?**
It gives a visual check of how the trained model classifies individual examples.

---

## Practical 10 – `DL_10.ipynb`
### CNN for Sign Language MNIST

**Q1. Which CNN layers are used?**
Two Conv2D layers with 32 and 64 filters, each followed by MaxPooling2D, then Flatten, Dense and Softmax output.

**Q2. What is a 3×3 kernel?**
A learnable filter that examines a 3×3 local region as it slides over the image.

**Q3. Why use ReLU?**
It introduces non-linearity by setting negative activations to zero.

**Q4. Why use MaxPooling?**
It reduces feature-map spatial size and computation.

**Q5. What is the optimizer and loss?**
The notebook compiles with Adam and sparse categorical crossentropy.

**Q6. What does evaluate report?**
The test loss and accuracy on the supplied test set.

**Q7. Important code check: how many classes?**
This notebook uses 24 Softmax outputs for the mapped Sign Language classes.

---

## Practical 11 – `DL_11.ipynb`
### FNN for Fashion-MNIST

**Q1. What is Fashion-MNIST?**
A dataset of 28×28 grayscale clothing images in ten categories.

**Q2. What is the FNN architecture?**
Flatten, Dense layers of 128 and 64 ReLU units, and a 10-unit Softmax output.

**Q3. Why ten output neurons?**
Fashion-MNIST contains ten clothing categories.

**Q4. Why sparse categorical crossentropy?**
The labels are integer category IDs.

**Q5. What is a confusion matrix?**
A table of actual versus predicted classes that reveals correct classifications and class confusions.

**Q6. What do precision and recall mean?**
Precision measures correctness among predicted positives; recall measures how many actual positives were found.

**Q7. Why use a classification report?**
It summarizes precision, recall, F1-score and support for each class.

---

## Practical 12 – `DL_12.ipynb`
### FNN for MNIST handwritten digits

**Q1. What is MNIST?**
A dataset of 28×28 grayscale handwritten digit images from 0 to 9.

**Q2. What is the input size after flattening?**
784 values per image.

**Q3. Why normalize the images?**
To scale pixel values from 0–255 to 0–1.

**Q4. Describe the network.**
Flatten, Dense layers with 128 and 64 ReLU units, and ten Softmax outputs.

**Q5. Why ten outputs?**
There are ten digit classes.

**Q6. Which loss is used?**
Sparse categorical crossentropy, because labels are integer digits.

**Q7. How is final performance measured?**
The notebook evaluates on the MNIST test set and reports loss and accuracy.

---

## Practical 13 – `DL_13.ipynb`
### FNN for Sign Language MNIST

**Q1. What is the task?**
Recognize Sign Language alphabet images using a feedforward neural network.

**Q2. Why map labels to 24 classes?**
The dataset has 24 static alphabet classes; the mapping makes output indices consecutive.

**Q3. Why flatten images?**
Dense layers require a vector, so 28×28 images become 784 features.

**Q4. Describe the model.**
Dense layers with 128 and 64 ReLU neurons followed by 24 Softmax outputs.

**Q5. What is validation_split?**
A fraction of training data held out to monitor performance during fitting.

**Q6. How are training and test performance compared?**
The notebook evaluates the model on both training and test arrays.

**Q7. Why convert class indices to letters?**
To make predictions understandable as alphabet symbols rather than numeric IDs.

---

## Practical 14A – `DL_14.ipynb`
### FNN for CIFAR-10

**Q1. What is CIFAR-10?**
A dataset of 32×32 RGB images in ten classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.

**Q2. How many input values are there per image?**
32×32×3 = 3072 pixel-channel values when flattened.

**Q3. Describe the FNN.**
Flatten, Dense layers of 256 and 128 ReLU units, and ten Softmax outputs.

**Q4. Why ten output neurons?**
CIFAR-10 has ten categories.

**Q5. Why normalize pixel values?**
To scale input values from 0–255 to 0–1.

**Q6. Why might CNNs perform better than FNNs on images?**
CNNs preserve spatial structure and learn local patterns with shared filters.

**Q7. What does model.evaluate do?**
It computes loss and the configured metrics on test examples.

---

## Practical 14B – `DL_14(using cifar-10 dataset).ipynb`
### FNN for CIFAR-10 loaded from CSV

**Q1. How is the dataset loaded?**
The notebook reads the cifar-10.csv file using Pandas.

**Q2. How are X and y formed?**
The label column is separated from the pixel columns.

**Q3. Why reshape to image dimensions?**
The flattened pixels are reshaped to 32×32×3 to represent RGB images.

**Q4. Why use train_test_split?**
To separate examples into training and testing sets.

**Q5. What is the model output size?**
Ten Softmax units for the ten CIFAR-10 classes.

**Q6. What does division by 255 do?**
It normalizes 8-bit image pixels to the 0–1 range.

**Q7. What must be correct when reshaping CSV pixels?**
The pixel column order and RGB channel ordering must match the reshape assumptions.

---

## Practical 15 – `cheet_15.ipynb`
### Comparing FNN architectures on MNIST

**Q1. What is the aim?**
Compare two FNN architectures for handwritten digit classification.

**Q2. What is Model 1?**
Flatten, Dense(128, ReLU), then Dense(10, Softmax).

**Q3. What is Model 2?**
Flatten, Dense(128, ReLU), Dense(64, ReLU), then Dense(10, Softmax).

**Q4. What is being compared?**
The test accuracy and test loss of a one-hidden-layer model and a two-hidden-layer model.

**Q5. Does adding a hidden layer always improve results?**
No. It can increase capacity but may also increase computation or overfitting.

**Q6. Why use Adam and sparse categorical crossentropy?**
Adam optimizes the weights; sparse categorical crossentropy fits integer digit labels.

**Q7. Why plot bar charts?**
They make the accuracy and loss comparison between the two models easy to see.

---

## Practical 16 – `cheet_16.ipynb`
### FNN for Fashion-MNIST with evaluation report

**Q1. What is the dataset?**
Fashion-MNIST: 28×28 grayscale clothing images across ten classes.

**Q2. Describe the network.**
Flatten, Dense(128, ReLU), Dense(64, ReLU), and Dense(10, Softmax).

**Q3. What are the training settings?**
Adam, sparse categorical crossentropy, 10 epochs and batch size 32 with 20% validation split.

**Q4. What is the purpose of class_names?**
It maps class IDs to readable clothing names for plots and reports.

**Q5. How is a confusion matrix created?**
Predicted class IDs are compared with true labels using confusion_matrix.

**Q6. What does the heatmap show?**
Counts of actual-versus-predicted class combinations.

**Q7. What is F1-score?**
The harmonic mean of precision and recall.

**Q8. What does history store?**
Per-epoch training and validation metrics returned by model.fit.

---

## Practical 17 – `cheet_17.ipynb`
### CNN for MNIST handwritten digits

**Q1. What is the task?**
Classify handwritten MNIST digits with a CNN.

**Q2. Why reshape to 28×28×1?**
The images are grayscale and Conv2D expects a channel dimension.

**Q3. What does Conv2D learn?**
Filters that respond to visual features such as edges and strokes.

**Q4. What does MaxPooling do?**
Reduces the width and height of feature maps.

**Q5. What is the output-class consistency issue in the notebook?**
The model definition shows 24 Softmax outputs although MNIST has 10 digit classes; a correct MNIST classifier should use 10 outputs.

**Q6. What is validation_split used for?**
It reserves a fraction of training examples for validation.

**Q7. How are predictions displayed?**
The notebook predicts a sample of test images, selects classes with argmax and plots the images with labels.

---

## Practical 18 – `cheet_18.ipynb`
### CNN for Fashion-MNIST

**Q1. What is Fashion-MNIST?**
Ten categories of 28×28 grayscale clothing images.

**Q2. Why is input shape 28×28×1?**
Height and width are 28 pixels and there is one grayscale channel.

**Q3. What do the convolutional layers do?**
They learn image features using 3×3 filters.

**Q4. What is MaxPooling2D?**
A downsampling layer that retains maximum activations in local windows.

**Q5. What is the output-size issue?**
The notebook uses 24 Softmax outputs, while Fashion-MNIST has ten classes; the output layer should have ten units.

**Q6. How are predictions converted to clothing names?**
argmax gives the class index and class_names maps that index to a category string.

**Q7. What does model.evaluate return?**
The loss and accuracy on the test data.

---

## Practical 19 – `cheet_19.ipynb`
### CNN for CIFAR-10 from CSV

**Q1. How is the data loaded?**
The notebook reads cifar-10.csv with Pandas and separates the label column from pixel features.

**Q2. What is the image shape?**
32×32×3 for RGB CIFAR-10 images.

**Q3. Why transpose after reshape?**
The CSV pixels are reshaped as channel-first and transposed to TensorFlow channels-last image format.

**Q4. What does stratify=y do?**
It preserves class proportions in the train/test split.

**Q5. What is the output-layer issue?**
The notebook specifies 24 output units even though CIFAR-10 has ten classes; the correct output size is ten.

**Q6. Why use CNN instead of FNN?**
Convolutions learn spatial image features while sharing filters across positions.

**Q7. What is displayed at the end?**
Predicted versus actual labels for sample images and example dataset images.

---

## Practical 20 – `cheet_20.ipynb`
### CNN for MNIST with accuracy and loss plots

**Q1. What is the objective?**
Train a CNN to classify MNIST handwritten digits and visualize learning curves.

**Q2. What preprocessing is applied?**
Pixel values are converted to float and divided by 255; images are reshaped to 28×28×1.

**Q3. What are the main CNN components?**
Conv2D, MaxPooling2D, Flatten and Dense classification layers.

**Q4. What does history.history contain?**
Training and validation loss and accuracy recorded at each epoch.

**Q5. What does the accuracy graph indicate?**
How classification accuracy changes over epochs for training and validation data.

**Q6. What does the loss graph indicate?**
How prediction error changes over epochs.

**Q7. What output-size issue should be noticed?**
The notebook uses 24 output neurons although MNIST has ten digit classes; the final layer should have ten units.

---

## Practical 21 – `cheet_21.ipynb`
### CNN for CIFAR-10 using tf.data pipeline

**Q1. What is the aim?**
Train a CNN on CIFAR-10 CSV data using TensorFlow dataset pipelines.

**Q2. Why load CSV values as uint8?**
Pixel values fit in 8-bit integers, reducing memory use before conversion.

**Q3. Why reshape and transpose?**
To convert channel-first flattened pixel data into TensorFlow's (height, width, channels) layout.

**Q4. What is tf.data.Dataset?**
TensorFlow's API for building efficient input pipelines.

**Q5. What does shuffle(5000) do?**
It randomizes examples using a buffer of 5000 elements.

**Q6. What does batch(32) do?**
It groups 32 examples into each batch.

**Q7. Why cast to float and divide by 255?**
Neural-network inputs are converted to floating point and normalized to 0–1.

**Q8. What does prefetch do?**
It prepares upcoming batches to overlap input processing with model computation.

**Q9. What output-size issue appears?**
The notebook model specifies 24 outputs although CIFAR-10 has ten classes; it should be ten.

---

## Practical 26 – `cheet_26.ipynb`
### VGG16 transfer learning by feature extraction

**Q1. What is transfer learning?**
Reusing a model trained on a large dataset for a different but related task.

**Q2. What is VGG16?**
A deep convolutional neural-network architecture with 16 weight layers.

**Q3. Why use weights=imagenet?**
It loads weights learned from the ImageNet dataset.

**Q4. Why include_top=False?**
It removes the original 1000-class classifier so a new task-specific classifier can be attached.

**Q5. Why freeze all VGG16 convolutional layers?**
The pre-trained convolutional base is used as a fixed feature extractor; its weights are not updated.

**Q6. What is the dataset/task?**
The notebook uses Caltech101 object-category images for a multi-class classifier.

**Q7. Why preprocess_input?**
It converts image values and channel order to the format expected by VGG16.

**Q8. What is the role of the new Dense Softmax layer?**
It maps extracted features to probabilities for the target classes.

**Q9. How is performance evaluated?**
The classifier is trained with a validation split and evaluated on held-out test data.

---

## Practical 27 – `cheet_27.ipynb`
### VGG16 fine-tuning on five Caltech101 categories

**Q1. What is the aim?**
Use ImageNet-pretrained VGG16 to classify five Caltech101 categories: airplanes, Faces, Motorbikes, Leopards and watch.

**Q2. What image size does VGG16 expect here?**
224×224 RGB images.

**Q3. What does include_top=False mean?**
Remove the original ImageNet classification head.

**Q4. How does this notebook differ from feature extraction?**
It freezes early VGG16 blocks but makes later layers trainable, so part of the pre-trained base is fine-tuned.

**Q5. What is GlobalAveragePooling2D?**
It averages each feature map spatially and converts feature maps into a feature vector.

**Q6. Why use Dropout(0.5)?**
It randomly drops half of the units during training to help reduce overfitting.

**Q7. Why use categorical crossentropy?**
The image labels are converted to one-hot vectors for the five classes.

**Q8. What is the output layer?**
Five Softmax units, one per target category.

**Q9. What caution is present in the notebook?**
It creates a validation dataset from the test dataset and later evaluates on that same dataset, so it is not a fully independent final test.

---

## Practical 28 – `cheet_28.ipynb`
### CBOW word prediction with one-hot targets

**Q1. What is NLP?**
Natural Language Processing is the field of working with human language using computers.

**Q2. What is CBOW?**
Continuous Bag of Words predicts a target word from its surrounding context words.

**Q3. What is tokenization?**
Splitting text into individual tokens, commonly words.

**Q4. Why lowercase and remove punctuation?**
To reduce superficial token differences and simplify the vocabulary.

**Q5. What is vocabulary?**
The set of unique tokens found in the training text.

**Q6. What is a context window?**
The number of neighboring words used to predict the target word.

**Q7. What does an Embedding layer do?**
It learns a dense vector representation for each vocabulary item.

**Q8. Why one-hot encode the target?**
It represents the target word as a vector with one active class position for categorical crossentropy.

**Q9. What does Softmax output?**
A probability distribution over the vocabulary for the predicted target word.

---

## Practical 29 – `DL28_29.ipynb`
### CBOW with a two-word context window

**Q1. What is the aim?**
Construct a Continuous Bag of Words model using a context window of two words on each side of the target word.

**Q2. What text corpus is used?**
A small manually defined sample corpus containing sentences about cats, dogs, deep learning, NLP and word embeddings.

**Q3. How is text cleaned?**
The notebook converts text to lowercase and removes punctuation using a regular expression.

**Q4. How is the vocabulary created?**
It takes the sorted set of unique tokens and assigns each word a unique integer ID.

**Q5. What does window_size=2 mean?**
Two words to the left and two words to the right are used as context, giving four context words.

**Q6. How are training samples generated?**
For each eligible center word, the code collects the four surrounding word IDs as context and stores the center word ID as target.

**Q7. What does GlobalAveragePooling1D do?**
It averages the embedding vectors for the context words into one context representation.

**Q8. What does the Embedding layer output?**
A trainable 50-dimensional vector for each input word ID.

**Q9. What is the training issue in this notebook?**
The fit call specifies epochs=00, which is zero epochs, so the model is not actually trained by that call.

**Q10. How is a target word predicted?**
The context is passed to the model, and argmax selects the vocabulary index with the highest predicted probability.

---

## Practical 30 – `DL_30.ipynb`
### CBOW model for next/target word prediction

**Q1. What is the learning task?**
Train a CBOW-style neural model to predict a target vocabulary word from context words.

**Q2. How is text prepared?**
The notebook cleans and tokenizes the text, builds a vocabulary and maps words to integer IDs.

**Q3. What are context and target arrays?**
Contexts contain neighboring word IDs; targets contain the word ID the model should predict.

**Q4. Why use an Embedding layer?**
It learns compact trainable vectors for words rather than treating word IDs as meaningful numeric magnitudes.

**Q5. What is the output layer?**
A Dense layer with vocabulary-size Softmax outputs.

**Q6. Why sparse categorical crossentropy?**
The target words are represented as integer vocabulary indices.

**Q7. What does Adam learning rate 0.005 control?**
It sets the scale of optimizer updates.

**Q8. What does model.evaluate measure?**
The loss and accuracy on the context-target examples supplied to evaluation.

---

## Practical 31 – `DL_31.ipynb`
### CBOW word embedding training and extraction

**Q1. What is the purpose?**
Train a CBOW model and inspect the learned word embeddings.

**Q2. What is a word embedding?**
A dense numeric vector learned to represent a word in a continuous space.

**Q3. What does word_to_index do?**
It maps each vocabulary word to its integer ID.

**Q4. What does index_to_word do?**
It maps integer IDs back to readable words.

**Q5. Why create context-target pairs?**
They provide supervised examples where surrounding words are inputs and the center word is the target.

**Q6. What does the Embedding layer learn?**
A trainable matrix whose rows are vector representations for vocabulary words.

**Q7. How can an embedding be retrieved?**
Use the embedding layer's learned weight matrix and select the row corresponding to a word ID.

**Q8. Why is the learning rate important?**
It controls how large parameter updates are during optimization.

---

## Practical 32 – `DL_32.ipynb`
### ECG anomaly detection using an autoencoder

**Q1. What is an autoencoder?**
A neural network trained to reconstruct its input.

**Q2. What is the aim?**
Learn normal ECG signal patterns and identify signals with unusually high reconstruction error.

**Q3. What are encoder and decoder?**
The encoder compresses input features into a smaller representation; the decoder reconstructs the original feature vector.

**Q4. What is the bottleneck?**
The narrow latent layer that forces the network to learn a compact representation.

**Q5. Why use MinMaxScaler?**
It scales each feature to a fixed range, typically 0 to 1.

**Q6. Why use mean squared error?**
It measures the squared difference between original and reconstructed values.

**Q7. How is the anomaly threshold selected?**
The notebook uses the 95th percentile of reconstruction errors on training/normal data.

**Q8. How are anomalies flagged?**
A sample whose reconstruction error exceeds the threshold is marked anomalous.

**Q9. Does the 95th percentile guarantee detection?**
No. It is a chosen thresholding rule and must be assessed against labeled validation/test examples.

---

## Practical 33 – `DL_33.ipynb`
### Energy anomaly detection using an autoencoder

**Q1. What is the goal?**
Use an autoencoder to learn normal energy data patterns and flag high reconstruction-error samples.

**Q2. Why scale the data?**
Scaling prevents features with large numeric ranges from dominating training.

**Q3. What is the encoder-decoder structure?**
The encoder compresses input features; the decoder expands the compressed representation back to the original feature count.

**Q4. What is the bottleneck layer?**
The smallest hidden representation, which captures compact information about the input.

**Q5. Why use MSE loss?**
The model reconstructs continuous numeric inputs, and MSE measures reconstruction differences.

**Q6. How is the threshold chosen?**
The notebook sets it at the 95th percentile of training reconstruction errors.

**Q7. What does errors > threshold mean?**
It returns a Boolean flag for samples whose reconstruction error is above the cutoff.

**Q8. Why plot a horizontal threshold line?**
It visually separates errors above the selected anomaly cutoff from lower errors.

---

## Practical 34 – `DL_34.ipynb`
### Credit-card fraud anomaly detection using an autoencoder

**Q1. What is the aim?**
Train an autoencoder on normal transaction features and flag possible fraudulent transactions using reconstruction error.

**Q2. Why use StandardScaler?**
It standardizes features by subtracting the mean and dividing by standard deviation.

**Q3. Why train primarily on normal transactions?**
The autoencoder learns the normal pattern; unusual transactions are expected to reconstruct less accurately.

**Q4. Describe the network.**
The notebook uses a compressed encoder-decoder Dense architecture to reconstruct the transaction feature vector.

**Q5. Why use linear output activation?**
The reconstruction output is intended to represent scaled numeric input features without restricting them to a probability.

**Q6. What is the loss function?**
Mean squared error between input and reconstructed transaction features.

**Q7. How is threshold selected?**
The notebook uses the 95th percentile of training reconstruction errors.

**Q8. How are predicted fraud flags produced?**
Errors greater than the threshold are converted to integer anomaly labels.

**Q9. Is accuracy alone enough for fraud detection?**
No. Fraud is often rare, so precision, recall, F1-score and the confusion matrix are also important.

---

## Important code observations to remember

- In `cheet_17.ipynb`, `cheet_18.ipynb`, `cheet_19.ipynb`, `cheet_20.ipynb` and `cheet_21.ipynb`, the notebook code specifies 24 output neurons in places where the named dataset (MNIST, Fashion-MNIST or CIFAR-10) has 10 classes. For those datasets, the correct output size is 10; verify label encoding and output dimensions before training.
- In `cheet_27.ipynb`, the same dataset is used as validation data during fitting and then evaluated again as test data. A separate validation set and untouched test set would provide a cleaner evaluation.
- In `DL28_29.ipynb`, the training call uses `epochs=00`, which means zero epochs; as written, that call does not train the model.
- A 95th-percentile reconstruction-error threshold in `DL_32.ipynb`, `DL_33.ipynb` and `DL_34.ipynb` is a heuristic cutoff, not a guarantee that every anomaly/fraud case will be detected.
- High accuracy alone may be misleading on imbalanced fraud datasets. Also inspect precision, recall, F1-score and the confusion matrix.

## Final rapid-fire viva questions

**Q. What is the difference between `categorical_crossentropy` and `sparse_categorical_crossentropy`?**  
Categorical crossentropy expects one-hot encoded targets; sparse categorical crossentropy expects integer class IDs.

**Q. What is the difference between `fit()`, `evaluate()` and `predict()`?**  
`fit()` trains the model, `evaluate()` calculates loss and metrics, and `predict()` produces model outputs for new inputs.

**Q. Why should test data not be used for training?**  
The test set should represent unseen data. Using it for training or repeated tuning makes the final evaluation biased.

**Q. What is the difference between classification and anomaly detection?**  
Classification predicts one of predefined classes; anomaly detection identifies observations that differ unusually from learned normal patterns.

**Q. Why is preprocessing important?**  
Models expect consistent input shapes and numeric ranges. Correct preprocessing helps training stability and ensures inputs match the model design.
