#!/usr/bin/env python
# coding: utf-8

### Python Project 16: Building a Spam Filter with Naive Bayes
# In this project, we're going to build a spam filter for SMS messages using the multinomial Naive Bayes algorithm. 
# Our goal is to write a program that classifies new messages with an accuracy greater than 80% — 
# so we expect that more than 80% of the new messages will be classified correctly as spam or ham (non-spam).

# ### Exploring the Dataset
# We'll now start by reading in the dataset: "`SMSSpamCollection`"

import pandas as pd
# The data points are tab separated, so we'll need to use the sep='\t' parameter
# The dataset doesn't have a header row, which means we need to use the header=None parameter, 
# otherwise the first row will be wrongly used as the header row.
# Naming the columns as Label and SMS.
sms_spam = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['Label', 'SMS'])

print(sms_spam.shape)
sms_spam.head()

# Lets find what percentage of the messages is spam and what percentage is ham ("ham" means non-spam).
sms_spam['Label'].value_counts(normalize=True)

#### Training and Test Set
# We're now going to split our dataset into a training and a test set, where the training set accounts for 80% of the data, 
# and the test set for the remaining 20%.

# Randomize the dataset
data_randomized = sms_spam.sample(frac=1, random_state=1)

# Calculate index for split
training_test_index = round(len(data_randomized) * 0.8)

# Training/Test split
training_set = data_randomized[:training_test_index].reset_index(drop=True)
test_set = data_randomized[training_test_index:].reset_index(drop=True)

print(training_set.shape)
print(test_set.shape)

# We'll now analyze the percentage of spam and ham messages in the training and test sets. 

training_set['Label'].value_counts(normalize=True)
test_set['Label'].value_counts(normalize=True)

# The results look good! We'll now move on to cleaning the dataset.

#### Data Cleaning
# We'll begin with removing all the punctuation and bringing every letter to lower case.
# Before cleaning
training_set.head()

# After cleaning
training_set['SMS'] = training_set['SMS'].str.replace('\W', ' ')
training_set['SMS'] = training_set['SMS'].str.lower()
training_set.head()

#### Creating the Vocabulary
# Let's now move to creating the vocabulary, which in this context means a list with all the unique words in our training set.
training_set['SMS'] = training_set['SMS'].str.split()

vocabulary = []
for sms in training_set['SMS']:
    for word in sms:
        vocabulary.append(word)
        
vocabulary = list(set(vocabulary))
len(vocabulary)
# It looks like there are 7,783 unique words in all the messages of our training set.

#### The Final Training Set
# We're now going to use the vocabulary we just created to make the data transformation we want.
word_counts_per_sms = {unique_word: [0] * len(training_set['SMS']) for unique_word in vocabulary}

for index, sms in enumerate(training_set['SMS']):
    for word in sms:
        word_counts_per_sms[word][index] += 1

word_counts = pd.DataFrame(word_counts_per_sms)
word_counts.head()

# Concatenating the DataFrame we just built above with the DataFrame containing the training set 
# (this way, we'll also have the `Label` and the `SMS` columns). Use the `pd.concat()` function.
training_set_clean = pd.concat([training_set, word_counts], axis=1)
training_set_clean.head()


#### Calculating Constants First
# We're now done with cleaning the training set, and we can begin creating the spam filter. 
# - P(Spam) and P(Ham)
# - NSpam, NHam, NVocabulary
# We'll also use Laplace smoothing and set $\alpha = 1$.

# Isolating spam and ham messages first
spam_messages = training_set_clean[training_set_clean['Label'] == 'spam']
ham_messages = training_set_clean[training_set_clean['Label'] == 'ham']

# P(Spam) and P(Ham)
p_spam = len(spam_messages) / len(training_set_clean)
p_ham = len(ham_messages) / len(training_set_clean)

# N_Spam
n_words_per_spam_message = spam_messages['SMS'].apply(len)
n_spam = n_words_per_spam_message.sum()

# N_Ham
n_words_per_ham_message = ham_messages['SMS'].apply(len)
n_ham = n_words_per_ham_message.sum()

# N_Vocabulary
n_vocabulary = len(vocabulary)

# Laplace smoothing
alpha = 1

#### Calculating Parameters
# Calculating the parameters P(w_i|Spam) and P(w_i|Ham). 
# Each parameter will thus be a conditional probability value associated with each word in the vocabulary.

# Initiate parameters
parameters_spam = {unique_word:0 for unique_word in vocabulary}
parameters_ham = {unique_word:0 for unique_word in vocabulary}

# Calculate parameters
for word in vocabulary:
    n_word_given_spam = spam_messages[word].sum()   # spam_messages already defined in a cell above
    p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
    parameters_spam[word] = p_word_given_spam
    
    n_word_given_ham = ham_messages[word].sum()   # ham_messages already defined in a cell above
    p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
    parameters_ham[word] = p_word_given_ham


#### Classifying A New Message
# Now that we have all our parameters calculated, we can start creating the spam filter. 
# The spam filter can be understood as a function that:
# - Takes in as input a new message (w1, w2, ..., wn).
# - Calculates P(Spam|w1, w2, ..., wn) and P(Ham|w1, w2, ..., wn).
# - Compares the values of P(Spam|w1, w2, ..., wn) and P(Ham|w1, w2, ..., wn), and:
#     - If P(Ham|w1, w2, ..., wn) > P(Spam|w1, w2, ..., wn), then the message is classified as ham.
#     - If P(Ham|w1, w2, ..., wn) < P(Spam|w1, w2, ..., wn), then the message is classified as spam.
#     - If P(Ham|w1, w2, ..., wn) = P(Spam|w1, w2, ..., wn), then the algorithm may request human help.

import re
def classify(message):
    '''
    message: a string
    '''
    message = re.sub('\W', ' ', message)
    message = message.lower().split()
    
    p_spam_given_message = p_spam
    p_ham_given_message = p_ham

    for word in message:
        if word in parameters_spam:
            p_spam_given_message *= parameters_spam[word]
            
        if word in parameters_ham:
            p_ham_given_message *= parameters_ham[word]
            
    print('P(Spam|message):', p_spam_given_message)
    print('P(Ham|message):', p_ham_given_message)
    
    if p_ham_given_message > p_spam_given_message:
        print('Label: Ham')
    elif p_ham_given_message < p_spam_given_message:
        print('Label: Spam')
    else:
        print('Equal proabilities, have a human classify this!')
        
        
# Testing above function (Test 1)
classify('WINNER!! This is the secret code to unlock the money: C3421.')
# Testing above function (Test 2)
classify("Sounds good, Tom, then see u there")

#### Measuring the Spam Filter's Accuracy
# The two results above look promising, but let's see how well the filter does on our test set, which has 1,114 messages. 
# We'll start by writing a function that returns classification labels instead of printing them.
def classify_test_set(message):    
    '''
    message: a string
    '''
    
    message = re.sub('\W', ' ', message)
    message = message.lower().split()
    
    p_spam_given_message = p_spam
    p_ham_given_message = p_ham

    for word in message:
        if word in parameters_spam:
            p_spam_given_message *= parameters_spam[word]
            
        if word in parameters_ham:
            p_ham_given_message *= parameters_ham[word]
    
    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    elif p_spam_given_message > p_ham_given_message:
        return 'spam'
    else:
        return 'needs human classification'


# Now that we have a function that returns labels instead of printing them, we can use it to create a new column in our test set.
test_set['predicted'] = test_set['SMS'].apply(classify_test_set)
test_set.head()

# Now, we'll write a function to measure the accuracy of our spam filter to find out how well our spam filter does.
correct = 0
total = test_set.shape[0]
    
for row in test_set.iterrows():
    row = row[1]
    if row['Label'] == row['predicted']:
        correct += 1
        
print('Correct:', correct)
print('Incorrect:', total - correct)
print('Accuracy:', correct/total)

# ### Result Analysis
# The accuracy is close to 98.74%, which is really good. Our spam filter looked at 1,114 messages that it hasn't seen in training, 
# and classified 1,100 correctly.
