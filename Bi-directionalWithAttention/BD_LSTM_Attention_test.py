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
sample_model = tf.keras.models.load_model("G:/IISc/2nd-semester/ASE with ML/Assignment/DeepFixLite/Attention_model")

#Take the input from the first layer of the model
encoder_model = Model(encoder_inputs, outputs = [encoder_outputs1, final_enc_h, final_enc_c])

decoder_state_h = Input(shape=(512,))
decoder_state_c = Input(shape=(512,))
decoder_hidden_state_input = Input(shape=(36,512))

dec_states = [decoder_state_h, decoder_state_c]

dec_emb2 = dec_emb_layer(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=dec_states)

# Attention inference
attention_result_inf, attention_weights_inf = attention_layer([decoder_hidden_state_input, decoder_outputs2])

decoder_concat_input_inf = Concatenate(axis=-1, name='concat_layer')([decoder_outputs2, attention_result_inf])

dec_states2= [state_h2, state_c2]

decoder_outputs2 = decoder_dense(decoder_concat_input_inf)

decoder_model= Model(
                    [decoder_inputs] + [decoder_hidden_state_input, decoder_state_h, decoder_state_c],
                     [decoder_outputs2]+ dec_states2)

def decode_sequence(input_seq):
  # Encode the input as state vectors.
  enc_output, enc_h, enc_c = encoder_model.predict(input_seq)

  # Generate empty target sequence of length 1.
  # target_seq = np.zeros((1, 1, num_unique_tokens))
  target_seq = np.zeros((1, 1))
  # Populate the first character of target sequence with the start character.
  # target_seq[0, 0, 1] = 1.0
  target_seq[0, 0] = 1

  # Sampling loop for a batch of sequences
  # (to simplify, here we assume a batch of size 1).
  stop_condition = False
  decoded_sentence = ''
  while not stop_condition:

    output_tokens, h, c = decoder_model.predict([target_seq] + [enc_output, enc_h, enc_c ])
    # Sample a token
    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    #print("sampled_token_index",sampled_token_index)
    if sampled_token_index == 0:
      break
    else:
      # convert max index number to marathi word
      sampled_char = index2token[sampled_token_index]
    #print("sampled_char",sampled_char)        
    if sampled_char == 'EOS' or len(decoded_sentence) > sent_length:
      stop_condition = True
    else:
      decoded_sentence += ' '+sampled_char
        


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
 
