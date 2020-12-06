from abc import ABC,abstractmethod
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class Gui():
    def __init__(self,FunctionNames,Functions):
        #Unset
        self.network = None
        self.data = None
        #GUI Layout
        self.root = tk.Tk()
        self.plotFrame =tk.Frame(self.root)
        self.plotFrame.grid(column=0,row=0)
        self.settingFrame =tk.Frame(self.root,width=100)
        self.settingFrame.grid(column=1,row=0)
        #GUI Plot
        self.fig = plt.Figure(figsize=(5,4),dpi=100)
        self.ax1 = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig,self.plotFrame)
        self.canvas.get_tk_widget().grid()
        #GUI Number of neurons in Hidden Layer
        
        self.batchSize = tk.Entry(self.settingFrame,text="Batchsize")
        self.batchSize.insert(0,"32")
        self.batchSize.grid(column=0,row=0)
        lbatch = tk.Label(self.settingFrame,text="Batchsize                    ")
        lbatch.grid(column=1,row=0)
        self.iterations = tk.Entry(self.settingFrame,text="Iterations of Training")
        self.iterations.insert(0,"5000")
        self.iterations.grid(column=0,row=1)        
        lbatch = tk.Label(self.settingFrame,text="Iterations of Training   ")
        lbatch.grid(column=1,row=1)
        self.numNeurons = tk.Entry(self.settingFrame,text="")
        self.numNeurons.insert(0,"16")
        self.numNeurons.grid(column=0,row=2)
        lbatch = tk.Label(self.settingFrame,text="Neurons in hidden Layer")
        lbatch.grid(column=1,row=2)
        ##GUI dropdownMenu
        #self.currentFunc = tk.StringVar(self.settingFrame)
        #self.funcDropMenu=tk.OptionMenu(self.settingFrame,self.currentFunc,*FunctionNames)
        #self.currentFunc.set(FunctionNames[0])
        #self.funcDropMenu.grid(column=0,row=0)
        #GUI Buttons
        self.buttonApply = tk.Button(self.settingFrame,text="apply & create new network",command= lambda: self.network.changeHiddenLayer(int(self.numNeurons.get()),Functions[2],Functions[3]))
        self.buttonApply.grid(column=0)
        self.buttonTrain = tk.Button(self.settingFrame,text="train",command= lambda:self.network.train(self.data.samples,int(self.iterations.get()),int(self.batchSize.get())))
        self.buttonTrain.grid(column=0)
        self.buttonEvaluate = tk.Button(self.settingFrame,text="evaluate",command= lambda:self.plot_decision_boundary(self.data.samples,self.network))
        self.buttonEvaluate.grid(column=0)
    
    def loop(self):
        self.root.mainloop()
    def newNetwork(self,width):
        self.network = Network(width,Functions[2],Functions[3])
    def plot_decision_boundary(self, x, model, steps=1000, cmap='Paired'):
        # define bounds of the domain
        min1, max1 = x[:, 2].min(), x[:, 2].max()+0.03
        min2, max2 = x[:, 3].min(), x[:, 3].max()+0.03
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
        grid = np.hstack((r1,r2))
        ###print("compute:",network.compute(grid))
        yhat = np.array([ 1-i[0] +i[1] for i in network.compute(grid)])
        #hat = network.compute(grid)
        # reshape the predictions back into a grid
        zz = yhat.reshape(xx.shape)
        # plot the grid of x, y and z values as a surface
        self.fig.add_subplot(111).contourf(xx, yy, zz,2)
        self.canvas.draw() 

class PlotModes:    
    def __init__(self,numSamples,numModes,fig,variance):
        self.totalSamples = numSamples*numModes
        self.samples = np.empty((1,4))#[x,y]
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
                self.samples = np.append(self.samples,np.asarray([[i%2,(i+1)%2,samplesX[j],samplesY[j]]]),axis=0)
        self.samples = self.samples[1:self.totalSamples+1]#delete first element  

class Layer(ABC):
    def __inti__(self):
        super().__init__()
    @abstractmethod
    def forward(self):
        pass
    @abstractmethod
    def backward(self):
        pass
    @abstractmethod
    def adjust(self):
        pass

class Linear(Layer):    
    def __init__(self,numNeurons,numInput):
        self.w = np.random.rand(numInput+1,numNeurons)#weights for each neuron for each input + theta for each neuron
        self.grad = []
        self.x = []
        self.numNeurons = numNeurons
        self.numInput = numInput
        super().__init__()

    def forward(self,x):
        #store x and add -1 for theta
        self.x = x      
        #multiply each weight with the corresponding weight for each sample
        return self._augment(x) @ self.w
        
    def backward(self,grad):
        #store grad
        self.grad = grad 
        #remove the theta values for next layer
        return (grad @ self.w.T)[:,:-1]
    
    def adjust(self,eta):
        #print(eta * self._augment(self.x).T @self.grad)
        self.w -= eta * self._augment(self.x).T @self.grad
    
    def _augment(self,arr):
        return np.c_[arr,np.ones((arr.shape[0]))*-1]    

class Activation(Layer):
    def __init__(self,numNeurons,func,dfunc):
        self.numNeurons = numNeurons
        self.s = []
        self.func = func
        self.dfunc= dfunc
        super().__init__()

    def forward(self,s):
        #store s
        self.s = s
        #f(s)
        return self.func(self.s)
    def backward(self,grad): 
        #f'(s)*grad
        return self.dfunc(self.s)*grad
    def adjust(self,eta):
        pass

class Network():
    def __init__(self,width,func,dfunc):
        self.rows = 2
        self.width = width
        self.layers =[]
        #first layer 2 Neurons and each 2 inputs
        self.layers += [Linear(2,2)]
        self.layers += [Activation(2,func,dfunc)]
        #secound layer var Neurons and each 2 inputs
        self.layers += [Linear(width,2)]
        self.layers += [Activation(width,func,dfunc)]
        #last layer 2 Neurons and each var inputs
        self.layers += [Linear(2,width)]
        self.layers += [Activation(2,func,dfunc)]

    def changeHiddenLayer(self,width,func,dfunc):
        self.rows = 2        
        self.width = width
        self.layers =[]
        #first layer 2 Neurons and each 2 inputs
        self.layers += [Linear(2,2)]
        self.layers += [Activation(2,func,dfunc)]
        #secound layer var Neurons and each 2 inputs
        self.layers += [Linear(width,2)]
        self.layers += [Activation(width,func,dfunc)]
        #last layer 2 Neurons and each var inputs
        self.layers += [Linear(2,width)]
        self.layers += [Activation(2,func,dfunc)]

    def compute(self,samples):
        store = samples
        for layer in self.layers:
            store = layer.forward(store)
        return store;

    def train(self,samples,iterations,batchsize):
        lastElementOfBatchNum = 0
        etaStart = 5/self.width
        index = 0
        #for iterations
        for i in range(0,iterations):
            #variable Learningrate
            eta = etaStart-(i/iterations)*etaStart
            #Gets random Batch from samples
            batchSample = samples[index:index+batchsize]
            #if the random index makes the batch overstand the Array of the samples add the rest of the beginning of the Array
            if index >= len(samples):
                index = index - len(samples)
                batchSample = np.append(batchSample,samples[0:index],axis=0)

            input = batchSample[:,2:]

            target = batchSample[:,:2]
            #print("target",target)
            #let the network predict
            predictions = self.compute(input)
            #print("prediction",predictions)
            #calculate the error
            grad = 2*(predictions-target)
            #print("grad",grad)
            #print the Loss
            print("Loss: ",np.sum((predictions-target)**2))

            #backprobagate of the layers
            for layer in self.layers[::-1]:
                grad = layer.backward(grad)
            #adjust the weights of the layer
            for layer in self.layers:
                layer.adjust(eta)
            index +=batchsize
   
if __name__ == "__main__":
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
        lambda s: 1.0/(1+np.exp(-s)),
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
    
    #instanciate the Gui
    gui = Gui(FunctionNames,Functions)
    #width of the hidden layer
    width = 32
    #instanciate the Network
    network = Network(width,Functions[2],Functions[3])
    gui.network = network
    #instanciate the random data
    data = PlotModes(100,10,gui.fig,0.0005)
    gui.data = data
    #shuffel the random data samples
    np.random.shuffle(data.samples)

    gui.loop()
   
