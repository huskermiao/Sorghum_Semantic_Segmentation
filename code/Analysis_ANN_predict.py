"""
make predictions using trained model
"""
import sys
import numpy as np
import pandas as pd

def Predict(h5_model_name, testing_data_fn, output_fn):
    """
    h5_model_name: trained model name
    testing_data_fn: testing data in npy fromat with the same dimension as the data used for training
    output_fn: file name for the prediction output 
    """
    from keras.models import load_model
    my_model = load_model(model)
    test_npy = np.load(npy)
    npy_shape = test_npy.shape
    print('testing data shape:', npy_shape)
    npy_dim = len(npy_shape)
    npy_2d = test_npy.reshape(npy_shape[0]*npy_shape[1], npy_shape[2]) if npy_dim==3 else test_npy
    npy_2d = npy_2d/255
    pre_prob = my_model.predict(npy_2d)
    predictions = pre_prob.argmax(axis=1) # this is a numpy array
    if npy_dim == 3:
        import scipy.misc as sm
        predictions = predictions.reshape(npy_shape[0], npy_shape[1])
        df = pd.DataFrame(predictions)
        df1 = df.replace(0, 255).replace(1, 127).replace(2, 253).replace(3, 190)#0: background
        df2 = df.replace(0, 255).replace(1, 201).replace(2, 192).replace(3, 174)#1: leaf
        df3 = df.replace(0, 255).replace(1, 127).replace(2, 134).replace(3, 212)#2: stem
        arr = np.stack([df1.values, df2.values, df3.values], axis=2)#3: panicle 
        sm.imsave('%s.png'%output_fn, arr)
        print('Done, check %s.png!'%output_fn)
    elif npy_dim == 2:
        np.savetxt(output_fn, predictions)
        print('Done, check %s!'%output_fn)
    else:
        sys.exit('only 2 anb 3 dim array supported!')

if __name__ == "__main__":
    if len(sys.argv)==4:
        Predict(*sys.argv[1:])
    else:
        print('usg: python Analysis_ANN_predict.py h5_model_name testing_data_fn output_fn')
