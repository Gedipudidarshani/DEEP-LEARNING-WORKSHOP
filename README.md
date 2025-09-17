# DEEP-LEARNING-WORKSHOP
## TITLE:Binary Classification with Neural Networks on the Census Income Dataset

## PROGRAM:
```
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
%matplotlib inline

df = pd.read_csv('income.csv')
```
```
print(len(df))
df.head()
```
<img width="851" height="218" alt="image" src="https://github.com/user-attachments/assets/f2f5af0e-9542-4daf-9372-7636b638b09c" />

```
df['label'].value_counts()
```

<img width="810" height="74" alt="image" src="https://github.com/user-attachments/assets/c72f5bf4-a09a-494c-a656-a9b33467213b" />

```
df.columns
```

<img width="824" height="61" alt="image" src="https://github.com/user-attachments/assets/7c7feb72-45ba-4eff-808f-c90ed711943e" />

```
cat_cols = ['sex', 'education', 'marital-status', 'workclass', 'occupation']
cont_cols = ['age', 'hours-per-week']
y_col = ['label']
print(f'cat_cols  has {len(cat_cols)} columns')
print(f'cont_cols has {len(cont_cols)} columns')
print(f'y_col     has {len(y_col)} column')
```

<img width="845" height="63" alt="image" src="https://github.com/user-attachments/assets/d269bb83-b24b-4d76-b6df-2e8f9200c1e0" />

```
for col in cat_cols:
    df[col] = df[col].astype('category')
cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
print(emb_szs)
```

<img width="793" height="35" alt="image" src="https://github.com/user-attachments/assets/ad8fc0f6-935c-458e-8147-0900269eba99" />

```
cats = np.stack([df[col].cat.codes.values for col in cat_cols], axis=1)
cats[:5]
cats = torch.tensor(cats, dtype=torch.int64)
```

<img width="854" height="101" alt="image" src="https://github.com/user-attachments/assets/56408088-c759-4833-9f38-9236fb8090a5" />

```
conts = np.stack([df[col].values for col in cont_cols], axis=1)
conts[:5]
conts = torch.tensor(conts, dtype=torch.float32)
```


```
y = torch.tensor(df[y_col].values, dtype=torch.int64).flatten()
b = 30000  # total records
t = 5000   # test size

cat_train = cats[:b-t]
con_train = conts[:b-t]
y_train = y[:b-t]

cat_test = cats[b-t:]
con_test = conts[b-t:]
y_test = y[b-t:]

torch.manual_seed(33)

```
```
class TabularModel(nn.Module):

    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        # Call the parent __init__
        super().__init__()
        
        # Set up the embedding, dropout, and batch normalization layer attributes
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        # Assign a variable to hold a list of layers
        layerlist = []
        
        # Assign a variable to store the number of embedding and continuous layers
        n_emb = sum((nf for ni,nf in emb_szs))
        n_in = n_emb + n_cont
        
        # Iterate through the passed-in "layers" parameter (ie, [200,100]) to build a list of layers
        for i in layers:
            layerlist.append(nn.Linear(n_in,i)) 
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1],out_sz))
        
        # Convert the list of layers into an attribute
        self.layers = nn.Sequential(*layerlist)
    
    def forward(self, x_cat, x_cont):
        # Extract embedding values from the incoming categorical data
        embeddings = []
        for i,e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))
        x = torch.cat(embeddings, 1)
        # Perform an initial dropout on the embeddings
        x = self.emb_drop(x)
        
        # Normalize the incoming continuous data
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        
        # Set up model layers
        x = self.layers(x)
        return x
```
```
model = TabularModel(emb_szs, n_cont=len(cont_cols), out_sz=2, layers=[50], p=0.4)
model 
```

```
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

```
```
import time
start_time = time.time()

epochs = 300
losses = []

for i in range(epochs):
    i+=1
    y_pred = model(cat_train, con_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss)
    
    # a neat trick to save screen space:
    if i%25 == 1:
        print(f'epoch: {i:3}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'epoch: {i:3}  loss: {loss.item():10.8f}') # print the last line
print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed

```
```
plt.plot([loss.item() for loss in losses])
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.title("Training Loss")
plt.show()

```

```
with torch.no_grad():
    y_val = model(cat_test, con_test)
    loss = criterion(y_val, y_test)
print(f'CE Loss: {loss:.8f}')

```
```
correct = 0
for i in range(len(y_test)):
    if y_val[i].argmax().item() == y_test[i].item():
        correct += 1

accuracy = correct / len(y_test) * 100
print(f'{correct} out of {len(y_test)} = {accuracy:.2f}% correct')

```
