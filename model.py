import numpy as np
import pickle

loaded_model = pickle.load(open('F:/ML_PROJECTS/diabetes prediction/trained_model.sav','rb'))

 
input_data=(1,89,66,23,94,28.1,0.167,21)

input_data=np.asarray(input_data) #convert to numpy

# reshape as we are predecting for one instance
input_data=input_data.reshape(1,-1)

# standardize input
# std=sc.transform(input_data)
# print(std)

pred=loaded_model.predict(input_data)
print(pred)


if(pred[0]==0):
    print('diabetic')
else :
    print('not diabetic')