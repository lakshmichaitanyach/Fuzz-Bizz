
# coding: utf-8

# ## Logic Based FizzBuzz Function [Software 1.0]

# In[29]:


import pandas as pd

def fizzbuzz(n):
    
    # If the given nunmber is either divisible 3 and 5 the code will return "fizzbuzz" and if it is divisible by only 3 it will return "fuzz" or else if the number is divisible by 5 it will return "buzz"
    if n % 3 == 0 and n % 5 == 0:
        return 'FizzBuzz'
    elif n % 3 == 0:
        return 'Fizz'
    elif n % 5 == 0:
        return 'Buzz'
    else:
        return 'Other'


# ## Create Training and Testing Datasets in CSV Format

# In[30]:


def createInputCSV(start,end,filename):
    
    # lists are versatile and can be updated many times and the size is also variable and the input lists take input values and output list give output values
    inputData   = []
    outputData  = []
    
    # we need training data to train our machine learning alogorithm and the more is data the more is its accuracy
    for i in range(start,end):
        inputData.append(i)
        outputData.append(fizzbuzz(i))
    
    # the dataframe represents the given data into tabular form and to convert excel based analysis to python scripting
    dataset = {}
    dataset["input"]  = inputData
    dataset["label"] = outputData
    
    # Writing to csv
    pd.DataFrame(dataset).to_csv(filename)
    
    print(filename, "Created!")


# ## Processing Input and Label Data

# In[19]:


def processData(dataset):
    
    # we need to process the achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner and data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithms are executed in one data set
    data   = dataset['input'].values
    labels = dataset['label'].values
    
    processedData  = encodeData(data)
    processedLabel = encodeLabel(labels)
    
    return processedData, processedLabel


# In[20]:


def encodeData(data):
    
    processedData = []
    
    for dataInstance in data:
        
        # number of inputs of neurons
        processedData.append([dataInstance >> d & 1 for d in range(10)])
    
    return np.array(processedData)


# In[21]:


from keras.utils import np_utils

def encodeLabel(labels):
    
    processedLabel = []
    
    for labelInstance in labels:
        if(labelInstance == "FizzBuzz"):
            # Fizzbuzz
            processedLabel.append([3])
        elif(labelInstance == "Fizz"):
            # Fizz
            processedLabel.append([1])
        elif(labelInstance == "Buzz"):
            # Buzz
            processedLabel.append([2])
        else:
            # Other
            processedLabel.append([0])

    return np_utils.to_categorical(np.array(processedLabel),4)


# ## Model Definition

# In[22]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard

import numpy as np

input_size = 10
drop_out = 0.2
first_dense_layer_nodes  = 256
second_dense_layer_nodes = 4

def get_model():
    
    # the models are used to predict and analyze the given training set and train the algorithm accordingly
    # dense is implementation where activation is application of activation to the output elements
    # it allows us to model layer by layer and sharing of layers or  multiple inputs and outputs are not used in this model
    model = Sequential()
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))
    
    # dropout in ml is dropping out some of the layers in the neural network which avoids overfitting and also regularizes the network by reducing interdepency among the neurons
    model.add(Dropout(drop_out))
    
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('softmax'))
    # for multiclass classification we used softmax
    
    model.summary()
    
    # as it is one-hot endings of multiclass classification we used cat_crossentropy
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


# # <font color='blue'>Creating Training and Testing Datafiles</font>

# In[23]:


# Create datafiles
createInputCSV(101,1001,'training.csv')
createInputCSV(1,101,'testing.csv')


# # <font color='blue'>Creating Model</font>

# In[24]:


model = get_model()


# # <font color = blue>Run Model</font>

# In[25]:


validation_data_split = 0.2
num_epochs = 10000
model_batch_size = 128
tb_batch_size = 32
early_patience = 100

tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')

# Read Dataset
dataset = pd.read_csv('training.csv')

# Process Dataset
processedData, processedLabel = processData(dataset)
history = model.fit(processedData
                    , processedLabel
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                   )


# # <font color = blue>Training and Validation Graphs</font>

# In[26]:


get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.DataFrame(history.history)
df.plot(subplots=True, grid=True, figsize=(10,15))


# # <font color = blue>Testing Accuracy [Software 2.0]</font>

# In[27]:


def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return "Other"
    elif encodedLabel == 1:
        return "Fizz"
    elif encodedLabel == 2:
        return "Buzz"
    elif encodedLabel == 3:
        return "FizzBuzz"


# In[28]:


wrong   = 0
right   = 0

testData = pd.read_csv('testing.csv')

processedTestData  = encodeData(testData['input'].values)
processedTestLabel = encodeLabel(testData['label'].values)
predictedTestLabel = []

for i,j in zip(processedTestData,processedTestLabel):
    y = model.predict(np.array(i).reshape(-1,10))
    predictedTestLabel.append(decodeLabel(y.argmax()))
    
    if j.argmax() == y.argmax():
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy: " + str(right/(right+wrong)*100))

# Please input your UBID and personNumber 
testDataInput = testData['input'].tolist()
testDataLabel = testData['label'].tolist()

testDataInput.insert(0, "UBID")
testDataLabel.insert(0, "lakshmic")

testDataInput.insert(1, "personNumber")
testDataLabel.insert(1, "50290974")

predictedTestLabel.insert(0, "")
predictedTestLabel.insert(1, "")

output = {}
output["input"] = testDataInput
output["label"] = testDataLabel

output["predicted_label"] = predictedTestLabel

opdf = pd.DataFrame(output)
opdf.to_csv('output.csv')

