import sys
import csv
import math
import numpy as np

train_input = sys.argv[1]
test_input = sys.argv[2]
train_out = sys.argv[3]
test_out = sys.argv[4]
metrics_out = sys.argv[5]
num_epoch = int(sys.argv[6])
hidden_units = int(sys.argv[7])
init_flag = int(sys.argv[8])
learning_rate = float(sys.argv[9])

alphastart = np.zeros((hidden_units,1))
betastart = np.zeros((10,1))

if init_flag == 1:
    alpha = np.random.uniform(-0.1, 0.1, (hidden_units, 128))  
    beta = np.random.uniform(-0.1, 0.1, (10, hidden_units))    
else:
    alpha = np.zeros((hidden_units, 128))
    beta = np.zeros((10, hidden_units))
alpha = np.concatenate([alphastart,alpha], axis=1)
beta = np.concatenate([betastart,beta], axis=1)


with open(train_input,'r') as f:
    reader = csv.reader(f,delimiter = ',')
    xTrain = []
    yTrain = []
    for row in reader:
        label = row[0]
        feature = row[1:]
        xTrain.append(feature)
        yTrain.append(label)
xTrain = np.asarray(xTrain,'f')
yTrain = np.asarray(yTrain,'f') 

with open(test_input,'r') as f:
    reader = csv.reader(f,delimiter = ',')
    xTest = []
    yTest  = []
    for row in reader:
        label = row[0]
        feature = row[1:]
        xTest .append(feature)
        yTest .append(label)
xTest  = np.asarray(xTest ,'f')
yTest  = np.asarray(yTest ,'f')  

  

def LinearForward1(x,al):

    a = np.dot(al,x)
    return a

def LinearForward2(z,be):

    b = np.dot(be,z)
    return b
           

def LinearBackward1(a, w, b, g_b):#x, al, a, g_a

    g_w = np.dot(g_b.reshape(-1, 1), a.T.reshape(1, -1))
    g_x = np.dot(w.T, g_b)

    return g_w, g_x
    

def LinearBackward2(a, w, b, g_b): #z, be, b, g_b  #change T

    g_w = np.dot(g_b.reshape(-1, 1), a.T.reshape(1, -1))
    g_a = np.dot(w.T, g_b)

    return g_w, g_a #g_beta,g_z
    #div_l_beta = np.dot(div_l_b.reshape(-1, 1), z.T.reshape(1, -1))
    #div_l_z = np.dot(beta_star.T, div_l_b)       
def SigmoidForward(a):
    e = np.exp(np.multiply(a,-1))
    denom = np.add(e,1)
    score = np.divide(1,denom)
    return score

def SigmoidBackward(a,z,g_z):#a,z,g_z

    g_a = g_z * z * (1-z)
    return g_a
    #div_l_z * z * (1 - z)

def SoftmaxForward(b):
    yhat = np.exp(b)/np.sum(np.exp(b))
    return yhat

def SoftmaxBackward(b,y,yhat,g_yhat):
    g_b = yhat - y
    return g_b
    
def CrossEntropyForward(y,yhat):
    yt = -np.transpose(y) 
 
    multiple = np.dot(yt,np.log(yhat))  
    return multiple

def CrossEntropyBackward(y, yhat, J, g_J):
    g_yhat = -(np.dot(g_J, np.divide(y, yhat)))
    return g_yhat

    

def NNForward(x,y,al,be):
   
    a = LinearForward1(x,al)
    z = SigmoidForward(a)
    z = np.append(1.0,z)
    b = LinearForward2(z,be)
    yhat = SoftmaxForward(b)
    J = CrossEntropyForward(y,yhat)
    o = [x,a,z,b,yhat,J] 
    return o
     
def NNBackward(x,y,al,be,o):
    g_J = 1
    a = o[1]
    z = o[2]
    b = o[3]
    yhat = o[4]
    J = o[5]
    
    g_yhat = CrossEntropyBackward(y, yhat, J, g_J)
    g_b = SoftmaxBackward(b,y,yhat,g_yhat)
    be = be[:, 1:]
    g_beta,g_z = LinearBackward2(z, be, b, g_b)

    z = z[1:]
    g_a = SigmoidBackward(a,z,g_z)
    #question on demension of a 
    g_alpha,g_x = LinearBackward1(x, al, a, g_a)
    
    return g_alpha, g_beta
    

def ComputeCrossEntropy(xTrain,yTrain,xTest,yTest,al,be,epoch):
    train_com = np.array([])
    test_com = np.array([])
    
    for i in range(len(yTrain)):
        y = np.zeros((10))
        y[int(yTrain[i])] = 1.0
        x = np.append(1.0,xTrain[i])
        o = NNForward(x,y,al,be)
        train_com = np.append(train_com,o[5])        
    
    
    for i in range(len(yTest)):
        y = np.zeros((10))
        y[int(yTest[i])] = 1.0
        x = np.append(1.0,xTest[i])
        o = NNForward(x,y,al,be)
        test_com = np.append(test_com,o[5])        
     
    
    current_epoch_string = "epoch=" + str(epoch) + " crossentropy(train): " + str(np.mean(train_com)) + "\n"
    current_epoch_string += "epoch=" + str(epoch) + " crossentropy(test): " + str(np.mean(test_com)) + "\n"   
    
    return current_epoch_string



    
def SGD(xTrain,yTrain,xTest,yTest,num_epoch,init_flag,alpha,beta):
    metrics_string = ""
    count = 0 
       
    while count < num_epoch:
        for i in range(len(yTrain)):
            
            y = np.zeros((10))
            y[int(yTrain[i])] = 1.0
            
            x = np.append(1.0,xTrain[i])
            
            o = NNForward(x,y,alpha,beta)  
                     
            g_alpha, g_beta = NNBackward(x,y,alpha,beta,o) 
               
            alpha = alpha - learning_rate * g_alpha
            #alpha_change = np.multiply(learning_rate,g_alpha)

            #alpha = np.subtract(alpha,alpha_change)  
            beta = beta - learning_rate * g_beta
            #beta_change = np.multiply(learning_rate,g_beta) 
            
            #beta = np.subtract(beta,beta_change) 
        metrics_string += ComputeCrossEntropy(xTrain,yTrain,xTest,yTest,alpha,beta,(count+1))              
    
        count += 1
              
    return alpha,beta,metrics_string

def predictLabel(xT,yT,al,be,output):
    wr = open(output,'w')
    count_r = 0.0
    count_w = 0.0
    predictlabel = []
    for i in range(len(yT)):
        y = yT[i]
        x = np.append(1.0,xT[i])  
        o = NNForward(x,y,al,be) 
        
        prediction = np.argmax(o[4])
        predictlabel.append(prediction)        
        if prediction == y:
            count_r += 1.0
        if prediction != y: 
            count_w += 1.0
    for item in predictlabel:
        wr.write(str(item))
        wr.write('\n')
    error = str(count_w/(count_w + count_r))
    return error


trained_alpha, trained_beta,metrics_string = SGD(xTrain,yTrain,xTest,yTest,num_epoch,init_flag,alpha,beta)  
  
                
train_error = predictLabel(xTrain,yTrain,trained_alpha, trained_beta,train_out)
test_error = predictLabel(xTest,yTest,trained_alpha, trained_beta,test_out)


error = "error(train): " + str(train_error) + '\n' + "error(test): " + str(test_error) + '\n'
metrics_string += error

with open(metrics_out, 'w') as file:
    file.write(metrics_string)




  
