import tensorflow as tf
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, LSTM, Dense , Bidirectional ,  Concatenate
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from operator import itemgetter
import numpy as np
import pandas as pd
import os
import csv
import fileinput
import sys
import pickle
from customfunction import text2sequence , sequence2tokenization , encoder_decoder_data

sent_length=70
num_unique_tokens = 1004
latent_dim=256
batch_size=64
epochs = 10
emb_size = 50

filename1 = sys.argv[1]
filename2 = sys.argv[2]
valid_data = pd.read_csv('G:/IISc/2nd-semester/ASE with ML/Assignment/DeepFixLite/'+ filename1)

pickle_in1 = open("token2index.pickle" , "rb")
pickle_in2 = open("index2token.pickle" , "rb")
token2index = pickle.load(pickle_in1)
index2token = pickle.load(pickle_in2)

input_valid_texts = valid_data['sourceLineTokens']
num_valid_samples = len(input_valid_texts)
padded_input_valid_texts = sequence2tokenization(input_valid_texts , token2index , sent_length)
padded_target_valid_texts = padded_input_valid_texts
encoder_input_valid_data ,_,_ = encoder_decoder_data(padded_input_valid_texts , padded_target_valid_texts ,num_valid_samples ,sent_length ,num_unique_tokens )


#//////////////////////////////////////////////////////////////////////
sample_model = tf.keras.models.load_model("G:/IISc/2nd-semester/ASE with ML/Assignment/DeepFixLite/BD_LSTM_model")

#Take the input from the first layer of the model
encoder_inputs = sample_model.input[0]
output_encoder, forward_h, forward_c, backward_h, backward_c  = sample_model.layers[3].output
state_h = sample_model.layers[5]
state_c = sample_model.layers[6]
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
encoder_states = [state_h, state_c]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_input = sample_model.input[1]
decoder_state_input_h = Input(shape=(latent_dim * 2),name = "inp_1")
decoder_state_input_c = Input(shape=(latent_dim * 2),name = "inp_2")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_embedding= sample_model.layers[4]
x = decoder_embedding(decoder_input)
decoder_lstm = sample_model.layers[7]
decoder_outputs, state_h, state_c = decoder_lstm(x, initial_state= decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_dense = sample_model.layers[8]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_input] + decoder_states_inputs, [decoder_outputs] + decoder_states)

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    # target_seq = np.zeros((1, 1, num_unique_tokens))
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    # target_seq[0, 0, 1] = 1.0
    target_seq[0, 0] = 1

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:

        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        #print("sampled_token_index",sampled_token_index)
        sampled_char = index2token[sampled_token_index]
        #print("sampled_char",sampled_char)
        

        # # Exit condition: either hit max length
        # # or find stop character.
        
        if sampled_char == 'EOS' or len(decoded_sentence) > sent_length:
          stop_condition = True
        else:
          decoded_sentence.append(sampled_char)
        


        # # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # # Update states
        states_value = [h, c]
    return decoded_sentence

# for k in range(len(encoder_input_valid_data)):
row_list = []
headerList = ['sourceLineTokens', 'fixedTokens']
row_list.append(headerList)

if os.path.exists('G:/IISc/2nd-semester/ASE with ML/Assignment/DeepFixLite/'+ filename2):
    os.remove('G:/IISc/2nd-semester/ASE with ML/Assignment/DeepFixLite/'+ filename2)

for k in range(len(encoder_input_valid_data)):
    seq_index = k
    input_seq = encoder_input_valid_data[seq_index: seq_index + 1]
    #print('-',seq_index)
    #print('Input sentence:      ', input_valid_texts[seq_index])
    decoded_sentence = decode_sequence(input_seq)

    #print('Decoded sentence:    ', decoded_sentence)
    s=eval(input_valid_texts[seq_index]) 
    desired_token = []     
    for token in s:
      if(token in token2index.keys()):
        if(token2index[token] == 3):
          desired_token.append(token)
      else:
        desired_token.append(token)
    
    decod = []
    #print("length " = len(decod))
    idx = 0
    s=(decoded_sentence) 
    for token in s:
      if(token == "OOV_Token" and idx < len(desired_token)):
        decod.append(desired_token[idx])
        idx += 1
      elif(token != "PAD"):
        decod.append(token)
        
    #print('New Decoded sentence:',decod)
    list1 = []
    list2 = []
    finalList = []
    list1 = input_valid_texts[seq_index]
    list2 = decod
    finalList = [list1 , list2]
    row_list.append(finalList)
    # print(seq_index)

with open('G:/IISc/2nd-semester/ASE with ML/Assignment/DeepFixLite/'+ filename2, 'w', newline='') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
    writer.writerows(row_list)
 
