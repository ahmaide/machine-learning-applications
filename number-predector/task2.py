import numpy as np
import task1 as t1
import time
x = np.linspace(-3,3,100)
y = np.sin(x - 1)*0.8
z = np.column_stack((x,y))
np.random.seed(12)
np.random.shuffle(z)
cut = np.split(z,[20,100])
test = cut[0]
train = cut[1]

def get_mse(pr,ts):
    err = 0
    for i in range(20):
        diff = pr[i] - ts[i][1]
        diff = diff*diff
        err+=diff
    err = err/20
    return err


#Unlimited Application
start = time.time()
rgtree = t1.RegressionTree(train)
end = time.time()
pred = rgtree.predict(test)
error = get_mse(pred,test)

print("********UNLIMITED TEST********")
print("Height:")
print(rgtree.height)
print("Time to Build:")
print(end - start)
print("Error:")
print(error)

unHeight = rgtree.height

##Limit Height 1/2
start = time.time()
rgtree = t1.RegressionTree(train,max_height=unHeight*0.5)
end = time.time()
pred = rgtree.predict(test)
error = get_mse(pred,test)

print("********Height Limit 1/2********")
print("Height:")
print(rgtree.height)
print("Time to Build:")
print(end - start)
print("Error:")
print(error)

##Limit Height 3/4
start = time.time()
rgtree = t1.RegressionTree(train,max_height=unHeight*0.75)
end = time.time()
pred = rgtree.predict(test)
error = get_mse(pred,test)

print("********Height Limit 3/4********")
print("Height:")
print(rgtree.height)
print("Time to Build:")
print(end - start)
print("Error:")
print(error)

##Leaf Limit 2
start = time.time()
rgtree = t1.RegressionTree(train,leaf_size=2,limit="leaf size")
end = time.time()
pred = rgtree.predict(test)
error = get_mse(pred,test)

print("********Leaf Limit 2********")
print("Height:")
print(rgtree.height)
print("Time to Build:")
print(end - start)
print("Error:")
print(error)

##Leaf Limit 4
start = time.time()
rgtree = t1.RegressionTree(train,leaf_size=4,limit="leaf size")
end = time.time()
pred = rgtree.predict(test)
error = get_mse(pred,test)

print("********Leaf Limit 4********")
print("Height:")
print(rgtree.height)
print("Time to Build:")
print(end - start)
print("Error:")
print(error)

##Leaf Limit 8
start = time.time()
rgtree = t1.RegressionTree(train,leaf_size=8,limit="leaf size")
end = time.time()
pred = rgtree.predict(test)
error = get_mse(pred,test)

print("********Leaf Limit 8********")
print("Height:")
print(rgtree.height)
print("Time to Build:")
print(end - start)
print("Error:")
print(error)