#This is not included in the readme, but im gonna make it better than the rest
# im gonna use a boiler plat like the otherones, and start with like feature engineering, and changing the epochs and lr, and some layers.
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

df1 = pd.read_csv("data/APPL.csv")
df2 = pd.read_csv("data/nvda.csv")
df3 = pd.read_csv("data/google.csv")
df4 = pd.read_csv("data/MSFT.csv")
df5 = pd.read_csv("data/AMZN.csv")

df1 = df1.drop(columns=["Open", "High", "Low"])
df1 = df1.iloc[2:].reset_index(drop=True)

df2 = df2.drop(columns=["Open", "High", "Low"])
df2 = df2.iloc[2:].reset_index(drop=True)


df3 = df3.drop(columns=["Open", "High", "Low"])
df3 = df3.iloc[2:].reset_index(drop=True)

df4 = df4.drop(columns=["Open", "High", "Low"])
df4 = df4.iloc[2:].reset_index(drop=True)


df5 = df5.drop(columns=["Open", "High", "Low"])
df5 = df5.iloc[2:].reset_index(drop=True)

def custom_func(x):
    x = str(x)
    return float(x.split("-")[0] + x.split("-")[1] + x.split("-")[2])

df1.rename(columns={"Price": "Date"}, inplace=True)
df1["Date"] = df1["Date"].apply(custom_func)
df1["Close"] = df1["Close"].astype(float)


df2.rename(columns={"Price": "Date"}, inplace=True)
df2["Date"] = df2["Date"].apply(custom_func)
df2["Close"] = df2["Close"].astype(float)


df3.rename(columns={"Price": "Date"}, inplace=True)
df3["Date"] = df3["Date"].apply(custom_func)
df3["Close"] = df3["Close"].astype(float)



df4.rename(columns={"Price": "Date"}, inplace=True)
df4["Date"] = df4["Date"].apply(custom_func)
df4["Close"] = df4["Close"].astype(float)


df5.rename(columns={"Price": "Date"}, inplace=True)
df5["Date"] = df5["Date"].apply(custom_func)
df5["Close"] = df5["Close"].astype(float)

def getPast10(index):
    values = []
    for i in range(10, 0, -1):
        values.append(float(df.iloc[index - i]["Close"]))
    return values

def getPast10Volumes(index):
    values = []
    for i in range(10, 0, -1):
        values.append(float(df.iloc[index - i]["Volume"]))
    return values

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)
        self.out = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x


torch.manual_seed(41)
model = Model()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

df = None
for i in range(4):
    if i == 0:
        df = df1
    elif i == 1:
        df = df2
    elif i ==2:
        df = df3
    elif i == 3:
        df = df4
    elif i == 4:
        df = df5

    X = []
    y = []

    for i in range(10, len(df)):
        X.append(getPast10(i)+getPast10Volumes(i))
        y.append(float(df1.iloc[i]["Close"]))

    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


    epochs = 100

    for i in range(epochs):
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch {i}, Loss: {loss.item()}")

    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = criterion(test_pred, y_test)

    print(f"Test Loss: {test_loss.item()}")
