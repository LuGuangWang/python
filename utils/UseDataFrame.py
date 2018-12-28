import pandas

datas = {'row1':['wlg',20],'row2':['seven',30]}

df = pandas.DataFrame.from_dict(datas,orient='index',columns=['name','age'])

df=df.reset_index(drop=True)

print(df)