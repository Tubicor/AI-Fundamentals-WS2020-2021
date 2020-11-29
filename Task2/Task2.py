import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class PlotModes:    
    def __init__(self,numSamples,numModes,fig,variance):
        self.totalSamples = numSamples*numModes
        self.samples = np.empty((1,3))#[x,y]
        meansX = np.random.uniform(0,0.7,numModes)
        meansY = np.random.uniform(0,0.7,numModes)
        variances = np.random.uniform(0,variance,numModes)
        clasDev = np.random.choice(2,numModes)
        for i in range(0,numModes):
            #print(i)
            samplesX = np.random.normal(meansX[i],np.sqrt(variances[i]),numSamples)
            samplesY = np.random.normal(meansY[i],np.sqrt(variances[i]),numSamples)
            for j in range(0,numSamples):
                color = 'b'
                if(i%2 == 1):
                    color = 'r'
                fig.add_subplot(111).plot(samplesX[j],samplesY[j],color,marker='x')
                self.samples = np.append(self.samples,np.asarray([[i%2,samplesX[j],samplesY[j]]]),axis=0)
        self.samples = self.samples[1:self.totalSamples+1]#delete first element            
                
class Neuron:
    def __init__(self,func,derivative,learningRate):
        self.func = func
        self.derivative = derivative
        self.state = 0
        self.learningRate = learningRate
        self.w = np.random.rand(2)
        theta = 0.5#random
        self.w = np.r_[self.w,[theta]]
        self.w = np.r_[0,self.w]
    def compute(self,x):
        print("x:",x)
        print("w:",self.w)
        state = x @ self.w
        #print("state",self.state)
        return np.array(list(map(self.func,state)));
    def update(self,error,x):
        self.state = x @ self.w
        update = sum(np.asarray([(self.learningRate*error*np.array(list(map(self.derivative,self.state))))[i]*x[i] for i in range (0,len(x))]))
        #print("old w",self.w)        
        update[0] = 0
        #print("update",update)
        #self.w += np.mean(update)
        self.w = np.add(self.w,update)
        #print("new w",self.w)
    def resetWeights(self):
            self.w = np.random.rand(2)
            theta = 0.5
            self.w = np.r_[self.w,[theta]]
            self.w = np.r_[0,self.w]
    def setFunc(self,func,dfunc):
            self.func = func
            self.derivative = dfunc

def plot_decision_boundary(x, model,canvas, steps=1000, cmap='Paired'):
    # define bounds of the domain
    min1, max1 = x[:, 1].min(), x[:, 1].max()+0.03
    min2, max2 = x[:, 2].min(), x[:, 2].max()+0.03
    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.01)
    x2grid = np.arange(min2, max2, 0.01)
    ## create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)
    ## flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2, = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    r0 = np.zeros(r1.shape)
    rm1 = np.ones(r1.shape)*-1
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r0,r1,r2,rm1))
    yhat = neuron.compute(grid)
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)
    # plot the grid of x, y and z values as a surface
    fig.add_subplot(111).contourf(xx, yy, zz,1)
    canvas.draw()
   
def trainNeuron(iterations,batchsize,neuron):
    gradMSE = lambda t,p:(t-p)
    for i in range(1,iterations):
        neuron.learningRate = (1-(i/iterations))*learningRate
        #print(i/iterations*100,"%")
        randint = np.random.randint(0,data.totalSamples);
        randSamples = data.samples[randint:randint+batchsize]
        lastElementOfBatchNum = randint+batchsize-data.totalSamples;
        if lastElementOfBatchNum > 0:
            randSamples = np.append(randSamples,data.samples[0:lastElementOfBatchNum],axis=0)
        randSamples = np.c_[randSamples,np.ones(batchsize)*-1]
        prediction = neuron.compute(randSamples)
        #print("\nprediction",prediction)
        #print("\nexpection",randSamples[:,0])
        error = gradMSE(randSamples[:,0],prediction)
        #print("mean Error",MSE(randSamples[:,0],prediction))
        #print("error",error)
        neuron.update(error,randSamples)

if __name__ == '__main__':

    FunctionNames=[
        "Heaviside",
        "sigmoid",
        "sin",
        "tanh",
        "sign",
        "ReLu",
        "lekyReLu"]
    Functions=[
        #Heaviside 1
        lambda s:np.heaviside(s,1),
        lambda s : 1,
        #Sigmoid
        lambda s: 1.0/(1+np.exp(betaSigmoid*s)),
        lambda s: Functions[2](s)*(1-Functions[2](s)),
        #sin
        np.sin,
        np.cos,
        #tanh
        np.tanh,
        lambda s : 1/np.cos(s)**2,
        #sign
        lambda s: -1 if s<0 else (0 if s==0 else 1),
        lambda s: 1, #what is the deriviative of sign?
        #Relu
        lambda s: s if s>0 else 0,
        lambda s: 1 if s>0 else 0,
        #leakyRelu
        lambda s: s if s>0 else 0.01*s,
        lambda s: 1 if s>0 else 0.01        
    ]

    #GUI Layout
    root = tk.Tk()
    plotFrame =tk.Frame(root)
    plotFrame.grid(column=0,row=0)
    settingFrame =tk.Frame(root,width=100)
    settingFrame.grid(column=1,row=0)
    #GUI Plot
    fig = plt.Figure(figsize=(5,4),dpi=100)
    ax1 = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig,plotFrame)
    canvas.get_tk_widget().grid()
    #GUI dropdownMenu
    currentFunc = tk.StringVar(settingFrame)
    funcDropMenu=tk.OptionMenu(settingFrame,currentFunc,*FunctionNames)
    currentFunc.set(FunctionNames[0])
    funcDropMenu.grid(column=0,row=0)

    #Data
    numSamples= 10
    numModes = 2
    data = PlotModes(200,2,fig,0.05)
    np.random.shuffle(data.samples)
    #Functions
    
    betaSigmoid = -6
    learningRate = 0.5
    neuron = Neuron(Functions[0],Functions[1],learningRate)
    
    #GUI Widgets
    bApply = tk.Button(settingFrame,text="apply Func",command=lambda: neuron.setFunc(Functions[FunctionNames.index(currentFunc.get())*2],Functions[FunctionNames.index(currentFunc.get())*2+1]))
    bApply.grid(column=1,row=0)
    numTrain = tk.Entry(master=settingFrame,text="repititions")
    numTrain.insert(0,"500")
    numTrain.grid(column=1,row=1)
    bTrain = tk.Button(master=settingFrame,text="train",command=lambda: trainNeuron(int(numTrain.get()),32,neuron))
    bTrain.grid(column=0,row=1)    
    bEvaluate = tk.Button(master=settingFrame,text="evaluate",command = lambda: plot_decision_boundary(data.samples,neuron,canvas))
    bEvaluate.grid()
    bReset = tk.Button(master=settingFrame,text="reset Weights",command = lambda: neuron.resetWeights())
    bReset.grid()
    
    
    root.mainloop()
