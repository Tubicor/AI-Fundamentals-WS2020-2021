from abc import ABC,abstractmethod
import numpy as np
from Task2 import PlotModes
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class Gui():
    def __init__(self,FunctionNames,Functions):
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
        #GUI dropdownMenu
        self.currentFunc = tk.StringVar(self.settingFrame)
        self.funcDropMenu=tk.OptionMenu(self.settingFrame,self.currentFunc,*FunctionNames)
        self.currentFunc.set(FunctionNames[0])
        self.funcDropMenu.grid(column=0,row=0)
    def loop(self):
        self.root.mainloop()
    def plot_decision_boundary(self, x, model, steps=1000, cmap='Paired'):
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
        grid = np.hstack((r1,r2))
        yhat = np.sum(network.compute(grid),axis=1)/2
        # reshape the predictions back into a grid
        zz = yhat.reshape(xx.shape)
        # plot the grid of x, y and z values as a surface
        self.fig.add_subplot(111).contourf(xx, yy, zz)
        self.canvas.draw()     

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
        self.w = np.random.rand(numNeurons,numInput+1)#weights for each neuron for each input + theta for each neuron
        self.grad = []
        self.x = []
        self.numNeurons = numNeurons
        self.numInput = numInput
        super().__init__()

    def forward(self,x):
        #store x and add -1 for theta
        self.x = np.c_[x,np.ones((len(x)))*-1]        
        #multiply each weight with the corresponding weight for eacht sample
        return np.asarray([[singleInput@singleWeight for singleWeight in self.w] for singleInput in self.x])
        
    def backward(self,grad):
        #store grad
        self.grad = grad 
        #multiply grad for each sample of batchsize and connection with corresponding weight
        prod = np.asarray([[grad[i,j]*self.w[j] for j in range(0,len(self.w))] for i in range(0,len(grad))])
        #sum all mistakes and remove the missfitting for theta
        return np.sum(prod,axis=1)[:,0:-1]
    
    def adjust(self,eta):
        update = []
        #for each sample of the batch
        for i in range(0,len(self.x)):
            #multiply grad with every input for sample i
            update += [np.array([eta*self.grad[i]*xi for xi in self.x[i]])]
        #sum all updates up and transpone from shape(weight,neuron) to (neuron,weight)
        update = np.sum(np.asarray(update),axis=0).transpose()
        #add the update to the weights of the layer
        self.w = np.add(self.w,update)

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
        etaStart = 0.1
        #for iterations
        for i in range(0,iterations):
            #variable Learningrate
            eta = etaStart-(i/iterations)*etaStart
            
            #Gets random Batch from samples
            randIndex = np.random.randint(0,data.totalSamples);
            randSamples = data.samples[randIndex:randIndex+batchsize]
            lastElementOfBatchNum = randIndex+batchsize-data.totalSamples;
            #if the random index makes the batch overstand the Array of the samples add the rest of the beginning of the Array
            if lastElementOfBatchNum > 0:
                randSamples = np.append(randSamples,data.samples[0:lastElementOfBatchNum],axis=0)

            #let the network predict
            predictions = self.compute(randSamples[:,1:]);
            #calculate the error
            grad = 2*(randSamples[:,:1]-predictions)

            #backprobagate of the layers
            for layer in self.layers[::-1]:
                grad = layer.backward(grad)
            #adjust the weights of the layer
            for layer in self.layers:
                layer.adjust(eta)      
   
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
    width = 16
    #instanciate the Network
    network = Network(width,Functions[0],Functions[1])
    #instanciate the random data
    data = PlotModes(200,5,gui.fig,0.0005)
    #shuffel the random data samples
    np.random.shuffle(data.samples)
    
    #train network for 500 iterations with 32 batchsize
    network.train(data.samples,500,32)
    #print(network.compute(data.samples[:,1:]))
    
    gui.plot_decision_boundary(data.samples,network)
    gui.loop()
    