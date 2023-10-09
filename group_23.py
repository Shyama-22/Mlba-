'''Importing the very basic libraries for data extraction'''
import numpy as np
import argparse
import pandas as pd
'''
Importing difference Libraries needed for our work . 

'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM


# Encode labels
label_encoder = LabelEncoder()  # this variable is used for label encoding


def our_model(X_train,y_train,X_test,tokenizer,shape_X_len):
    # Build the model
    
    len_word = len(tokenizer.word_index)+1
    
    # Create a Sequential model
    model = Sequential()
    
    # Define the activation function for the first Dense layer    
    activation_layer1= 'relu'
    
    # Add an Embedding layer to the model    
    model.add(Embedding(input_dim=len_word, output_dim=32, input_length=shape_X_len))
    
    # Add a Bidirectional LSTM layer with 64 units and return sequences    
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    
    # Add another Bidirectional LSTM layer with 32 units    
    model.add(Bidirectional(LSTM(32)))
    
    # Define the activation function for the second Dense layer
    activation_layer2='sigmoid'
    
    # Add a Dense layer with 64 units and the specified activation function    
    model.add(Dense(64, activation=activation_layer1))
    # Add a Dropout layer with a dropout rate of 0.5
    model.add(Dropout(0.5))

    # Add the final Dense layer with 1 unit and the specified activation function
    model.add(Dense(1, activation=activation_layer2))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=15, batch_size=64)


    # Make predictions on the test data
    predictions = model.predict(X_test)
    
    threshold = 0.5  #Threshold thatwhen greater than 0.5 it should give 1 otherwise 0 . 
    
    
    predicted_labels = (predictions > threshold).astype(int)
    
    return predicted_labels   # returning out predicted labels 
  
  


def parse_arguments():
    # Create an ArgumentParser object and provide a description of the program.
    
    parser = argparse.ArgumentParser(description="MLBA Code")
    
    # Define a help message for the "--train_file" argument.
    st1 = 'Path to the training data file'
    parser.add_argument("--train_file", type=str, required=True, help=st1)
    # Define a help message for the "--test_file" argument.
    st2 = 'Path to the test data file'
    parser.add_argument("--test_file", type=str, required=True, help=st2)
    # Define a help message for the "--output_file" argument.
    st3= 'Path to save the output predictions'
    parser.add_argument("--output_file", type=str, required=True, help=st3)
    
    # Parse the command-line arguments and return the result.
    return parser.parse_args()
  
  
  
if __name__ == "__main__":
  
  # Parse command-line arguments
  args = parse_arguments()
  
  '''
    Loading the dataset in our system for working as data and test_data
  '''
  data = pd.read_csv(args.train_file)  # Since training dataset is in 'train.csv'
  test_data = pd.read_csv(args.test_file)  # Since testing dataset is in 'test.csv'

 
  '''
  Preprocessing our data ,
  since data is in text sequence .
  '''
  #Preprocessing on training data 

  tokenizer = Tokenizer(char_level=True)

  data_X = data['Sequence']  # Getting all values of Column Sequence 

  tokenizer.fit_on_texts(data_X)

  X = tokenizer.texts_to_sequences(data_X)  # Each character in text is replaced with its corresponding Integer

  X = pad_sequences(X) # Ensuring all sequences have the same length so padding it 

  shape_X_len = X.shape[1] # Getting the shape of sequence 
  
  y = label_encoder.fit_transform(data['Label'])

  y = data['Label']   # Getting our test Labels

  # Preprocessing the test data
  data_Y = test_data['Sequence']
  X_test = tokenizer.texts_to_sequences(data_Y)
  X_test = pad_sequences(X_test, maxlen=shape_X_len)

  # Split the data into training and validation sets
  X_train = X
  y_train = y

  '''
  Output that our_model will be predicting
  '''
  predicted_labels = our_model(X_train,y_train,X_test,tokenizer,shape_X_len)  # Uses our_model function 

  # Convert back to original labels
  predicted_labels = label_encoder.inverse_transform(predicted_labels.flatten())

  '''
  Dataframe creation for saving the file as required
  '''
  data_dictionary = {'ID':test_data['ID'],'Label':predicted_labels}
  submission_df = pd.DataFrame(data_dictionary)  # Create a DataFrame for submission

  # Save the submission file
  file_name = args.output_file
  submission_df.to_csv(file_name, index=False)