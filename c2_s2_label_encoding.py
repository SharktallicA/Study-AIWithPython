# LABEL ENCODING
# Converting data labels from words into numbers

import numpy as np
from sklearn import preprocessing

input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']

# Create and train label encoder obj
encoder =  preprocessing.LabelEncoder()
encoder.fit(input_labels)

# Map words to numbers
print("\nLabel mapping:")
for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)

# Test: Encode a set of labels
test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels)
print("\nLabels:", test_labels)
# Expect [1, 2, 0] since they correspond to the numerical order the three test labels ['green', 'red', 'black'] were originally trained in
print("Encoded values:", list(encoded_values))

# Test: Decode a set of values
encoded_values = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded values:", encoded_values)
# Expect ['white', 'black', 'yellow', 'green'] since that is what corresponds to the given numerical labels 
print("Decoded labels: ", list(decoded_list))