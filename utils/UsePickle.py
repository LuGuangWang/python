import pickle
import pandas

datas = {'row1':['wlg',20],'row2':['seven',30]}
df=pandas.DataFrame.from_dict(datas,orient='index',columns=['name','age'])

## The pickle module implements binary protocols for serializing and de-serializing a Python object structure

with open("../test_data/reviews.pkl",'wb') as f:
    pickle.dump(df,f,pickle.HIGHEST_PROTOCOL)

with open("../test_data/reviews.pkl",'rb') as f:
    fr = pickle.load(f)
    print(fr)