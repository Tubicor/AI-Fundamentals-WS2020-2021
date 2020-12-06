
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def plotModes(numSamples,numModes):
    meansX = np.random.uniform(0,1.0,numModes)
    meansY = np.random.uniform(0,1.0,numModes)
    variances = np.random.uniform(0,0.1,numModes)
    clasDev = np.random.choice(2,numModes)
    for i in range(0,numModes):
        #print(i)
        samplesX = np.random.normal(meansX[i],np.sqrt(variances[i]),numSamples)
        samplesY = np.random.normal(meansY[i],np.sqrt(variances[i]),numSamples)
        for j in range(0,numSamples):
            color = 'b'
            if(clasDev[i] == 1):
                color = 'r'
            axes.plot(samplesX[j],samplesY[j],color,marker='x')


def updateM(val):
    axes.clear()
    global numModes
    numModes = val
    plotModes(numSamples,numModes)

def updateS(val):
    axes.clear()
    global numSamples
    numSamples = val
    plotModes(numSamples,numModes)


if __name__ == '__main__':
    #Layout
    global axes
    fig, axes = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    #variables
    global numSamples
    numSamples= 100
    global numModes
    numModes = 2
    plotModes(numSamples,numModes)

    #sliders
    axes.margins(x=0)
    axcolor = 'lightgoldenrodyellow'
    axSamp = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axModes = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    sSamp = Slider(axSamp, 'Samples', 5, 200, valinit=numSamples, valstep=5)
    sModes = Slider(axModes, 'Modes', 2, 20, valinit=numModes,valstep = 1)
    sSamp.on_changed(updateS)
    sModes.on_changed(updateM)


    plt.show()
