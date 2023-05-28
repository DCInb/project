import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# define the toy model class
class toy_model:

    # define intial parameters
    def __init__(self, N, perc, lengthLoud, lengthNotice, strengthLoud, strengthNotice, prability = 0.5, tcool=1.0, tmax=10.0, dt=0.1, biasMax=0.5, biasMin=-0.25):
        self.N = N # the size of the map
        self.perc = perc # the percentage of brave people
        self.lengthLoud = lengthLoud # the length of applause affected by loud people
        self.lengthNotice = lengthNotice # the length of applause affected by stop people
        self.strengthLoud = strengthLoud # the strength of applause affected by loud people
        self.strengthNotice = strengthNotice    # the strength of applause affected by stop people
        self.prability = prability  # the probability width of applause
        self.tcool = tcool # the cooling time
        self.tmax = tmax # the maximum time
        self.dt = dt    # the time step
        self.biasMax = biasMax  # the maximum bias
        self.biasMin = biasMin  # the minimum bias for initial map

        self.t = np.array([0])   # the time array
        self.tstart = np.zeros((self.N, self.N))  # the start time of applause

        # define the initial map and bias
        self.map = np.zeros((2, self.N, self.N))    # the map of applause
        self.bais = self.biasMin*np.ones((self.N, self.N))  # the bias of applause
        self.braveindex = np.random.randint(self.N, size=(2, int(self.perc*self.N**2)))  # the index of brave people
        self.bais[self.braveindex[0, :], self.braveindex[1, :]] = 0   # the initial bias of brave people

        # define the decrease embarrassment cost matrix and increase embarrassment cost matrix
        lx = np.arange(-self.lengthLoud, self.lengthLoud+1)
        ly = np.arange(-self.lengthLoud, self.lengthLoud+1)
        lvx, lvy = np.meshgrid(lx, ly)
        self.loud = np.exp(-(lvx**2+lvy**2)/self.lengthLoud**2)*self.strengthLoud
        nx = np.arange(-self.lengthNotice, self.lengthNotice+1)
        ny = np.arange(-self.lengthNotice, self.lengthNotice+1)
        nvx, nvy = np.meshgrid(nx, ny)
        self.notice = np.exp(-(nvx**2+nvy**2)/self.lengthNotice**2)*self.strengthNotice

        # store the physical parameters
        self.clap = np.sum(self.map[0])/self.N**2
        self.stop = np.sum(self.map[1])/self.N**2
    # define the function to update the map
    def step(self):

        # update the time
        self.t = np.append(self.t, self.t[-1]+self.dt)

        # store the map
        map_old = self.map.copy()

        # update the map and start time of applause
        random = 0.25+np.random.rand(self.N, self.N)*self.prability+self.bais    # the random matrix to decide the map update
        clap = np.where(random > 0.5, 1, -1)    # the matrix index of update map
        self.map[0,clap==1] = 1  # update the map applauding
        self.map[1,clap==-1] = self.map[0,clap==-1]  # update the map stop applauding
        self.tstart[map_old[0] != self.map[0]] = self.t[-1]   # update the start time of applause

        # update the bias
        dec = sp.signal.convolve2d(self.map[0], self.loud, mode='same', boundary='fill', fillvalue=0)  # the decrease embarrassment cost
        inc = sp.signal.convolve2d(self.map[1], self.notice, mode='same', boundary='fill', fillvalue=0)   # the increase embarrassment cost
        self.bais = np.where(self.map[0]==0, dec+self.bais, self.bais)  # update the bias
        self.bais = np.where(self.map[0]==1, self.biasMax-inc-(self.biasMax-self.biasMin)*(self.t[-1]-self.tstart)/self.tcool, self.bais)  # update the bias
        self.bais = np.where(self.map[1]==1, 0, self.bais)  # update the bias

        # store the physical parameters
        self.clap = np.append(self.clap, np.sum(self.map[0])/self.N**2)
        self.stop = np.append(self.stop, np.sum(self.map[1])/self.N**2)

    # define the function to plot the proportion of applauding people and stop applauding people
    def plot_proportion(self, figname= 'proportion.png'):
        plt.figure(figsize=(10, 10))
        plt.plot(self.t, self.clap, label='started applauding')
        plt.plot(self.t, self.stop, label='stopped applauding')
        plt.plot(self.t, self.clap-self.stop, label='applauding')
        plt.xlabel('time')
        plt.ylabel('proportion of individuals')
        plt.legend()
        plt.savefig(figname)
        plt.show()

# define the main function to run the model
def main():
    # define the model
    model = toy_model(10, 0.05, 5, 3, 0.001, 0.01, prability = 0.5, tcool=5.5, tmax=10.0, dt=0.1, biasMax=0.5, biasMin=-0.25)
    # run the model
    for i in range(int(model.tmax/model.dt)):
        model.step()
    # plot the proportion of applauding people and stop applauding people
    model.plot_proportion(figname='N10Loud3Notice1Strength0.5.png')

if __name__ == '__main__':
    main()
