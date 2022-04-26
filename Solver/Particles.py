import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import gamma


class Particles():

    def __init__(self, time, LiquidProperty, Mesh):
        # Simulation parameters
        self.N = None  # Number of particles
        self.Tstart = time.Tstart  # current time of the simulation
        self.TEnd = time.Tend  # time at which simulation ends
        self.TimeStep = time.TimeStep
        self.dt = time.dt  # timestep
        self.m = LiquidProperty.m  # mass
        self.R = LiquidProperty.R  # liquid droplet radius
        self.h = LiquidProperty.h  # smoothing length
        self.k = LiquidProperty.k  # stiffness constant
        self.nu = LiquidProperty.nu  # damping
        self.rho0 = LiquidProperty.rho0
        self.Xstart = Mesh.start_x
        self.Xend = Mesh.end_x
        self.Ystart = Mesh.start_y
        self.Yend = Mesh.end_y
        self.pos = None  # randomly selected positions and velocities
        self.vel = None
        self.acc = None
        self.P = None
        self.rho = None
        self.w = None

        #targets
        self.npWtarget =None
        self.npXtarget = None
        self.npYtarget = None
        self.npUtarget = None
        self.npVtarget = None
        self.npAccXtarget = None
        self.npAccYtarget = None
        self.npRhotareget = None
        self.npPtarget = None

        # The number of colliding with rollers
        self.npColliding = None
        self.npCollidingTotal = None

    def printParticles(self):
        print("Number of particles", self.N)
        print("current time of the simulation", self.Tstart)
        print("time at which simulation ends", self.TEnd)
        print("timestep", self.dt)
        print("star radius", self.R)
        print("smoothing length", self.h)
        print("equation of state constant", self.k)
        print("damping", self.nu)

    def generateInitConditions(self, startGenerateX, endGenerateX, startGenerateY, endGenerateY, column, row):

        """
        pos     is a matrix of (x,y) positions
        vel     is a matrix of (x,y) velocity
        """
        self.N = column * row
        X = np.linspace(startGenerateX, endGenerateX, column).reshape(column,1)
        Y = np.linspace(startGenerateY, endGenerateY, row).reshape((row,1))
        posX = (X * np.ones_like(Y).T).flatten()
        posY = (Y * np.ones_like(X).T).T.flatten()

        self.pos = np.vstack((posX,posY)).T
        self.vel = np.zeros(self.pos.shape)
        self.npColliding = np.zeros_like(self.pos)

    def KernelFunction(self, x, y):

        """
        Gausssian Smoothing kernel (2D)
        x     is a vector/matrix of x positions
        y     is a vector/matrix of y positions
        z     is a vector/matrix of z positions
        h     is the smoothing length
        w     is the evaluated smoothing function
        alpha is constant
        coreZone is 0 <= q < 1/2
        sideZone is 1/2 <= q < 1
        """

        q = (np.sqrt(x ** 2 + y ** 2)) / self.h

        # save initial Numpy array shape
        indx = q.shape[0]
        indy = q.shape[1]

        # calculate alpha
        alpha = 40/(7 * np.pi * self.h)

        # flatten q and make w array which has same shape as q
        # if q is over than 1 the values of w array are 0
        q = q.flatten()
        w = np.zeros_like(q)

        # if q is smaller than 1
        sideZone = np.asarray(np.where(q <= 1)).flatten()
        w[sideZone] = alpha * 2 * (1-q[sideZone])**3

        # if q is smaller than 1/2
        coreZone = np.asarray(np.where(q < 1/2)).flatten()
        w[coreZone] = alpha * (6 * (q[coreZone]**3 - q[coreZone]**2)+1)

        # reshape to make shape as origin q array
        w = w.reshape(indx, indy)
        self.w = w

        return w

    def gradKernelFunction(self, x, y):
        """
        Gradient of the Gausssian Smoothing kernel (3D)
        x     is a vector/matrix of x positions
        y     is a vector/matrix of y positions
        z     is a vector/matrix of z positions
        h     is the smoothing length
        wx, wy, wz     is the evaluated gradient
        """

        q = (np.sqrt(x ** 2 + y ** 2)) / self.h

        # save initial Numpy array shape
        indx = q.shape[0]
        indy = q.shape[1]

        # calculate alpha
        alpha = 40 / (7 * np.pi * self.h)

        # flatten q and make w array which has same shape as q
        # if q is over than 1 the values of w array are 0
        q = q.flatten()
        w = np.zeros_like(q)

        # if q is smaller than 1
        sideZone = np.asarray(np.where(q <= 1)).flatten()
        w[sideZone] = alpha * -6 * (1 - q[sideZone]) ** 2

        # if q is smaller than 1/2
        coreZone = np.asarray(np.where(q < 1 / 2)).flatten()
        w[coreZone] = alpha * (18 * q[coreZone]**2 - 12 * q[coreZone])

        # reshape to make shape as origin q array
        w = w.reshape(indx, indy)
        wx = w * x
        wy = w * y

        return wx, wy

    def getPairwiseSeparations(self, ri, rj):
        """
        Get pairwise desprations between 2 sets of coordinates
        ri    is an M x 2 matrix of positions
        rj    is an N x 2 matrix of positions
        dx, dy, dz   are M x N matrices of separations
        """

        M = ri.shape[0]
        N = rj.shape[0]

        # positions ri = (x,y,z)
        rix = ri[:, 0].reshape((M, 1))
        riy = ri[:, 1].reshape((M, 1))

        # other set of points positions rj = (x,y,z)
        rjx = rj[:, 0].reshape((N, 1))
        rjy = rj[:, 1].reshape((N, 1))

        # matrices that store all pairwise particle separations: r_i - r_j
        dx = rix - rjx.T
        dy = riy - rjy.T

        return dx, dy

    def getDensity(self, r, pos):
        """
        Get Density at sampling loctions from SPH particle distribution
        r     is an M x 2 matrix of sampling locations
        pos   is an N x 2 matrix of SPH particle positions
        m     is the particle mass
        h     is the smoothing length
        rho   is M x 1 vector of accelerations
        """

        M = r.shape[0]

        dx, dy = self.getPairwiseSeparations(r, pos)

        rho = np.sum(self.m * self.KernelFunction(dx, dy), 1).reshape((M, 1))

        return rho

    def getPressure(self, rho):
        """
        k is user-defined stiffness constant that scales pressure, pressure gradient, and the resulting pressure force
        position based fluid PBF [Macklin 2013]
        """

        P = self.k*((rho / self.rho0)-1)

        return P
    def getViscosityAcc(self):

        """
        dx, dy        are matrix of distances differences
        nu            is  kinematic viscosity
        aVisX, aVisY  are matrix of acceleration of viscosity
        """

        dx, dy = self.getPairwiseSeparations(self.pos, self.pos)
        vx, vy = self.getPairwiseSeparations(self.vel, self.vel)
        nu = self.nu
        dWx, dWy = self.gradKernelFunction(dx, dy)

        aVisX = 2 * nu * (self.m / self.rho) * vx * (dWx / (dx * dx + 0.01 * self.h))
        aVisY = 2 * nu * (self.m / self.rho) * vx * (dWy / (dy * dy + 0.01 * self.h))

        aVisX = np.sum(aVisX, 1).reshape(self.N, 1)
        aVisY = np.sum(aVisY, 1).reshape(self.N, 1)

        return aVisX, aVisY


    def getAcc(self):
        """
        Calculate the acceleration on each SPH particle
        pos   is an N x 2 matrix of positions
        vel   is an N x 2 matrix of velocities
        m     is the particle mass
        h     is the smoothing length
        k     equation of state constant
        n     polytropic index
        nu    viscosity
        a     is N x 2 matrix of accelerations
        """

        # Calculate densities at the position of the particles
        self.rho = self.getDensity(self.pos, self.pos)

        # Get the pressures
        self.P = self.getPressure(self.rho)

        # Get pairwise distances and gradients
        dx, dy = self.getPairwiseSeparations(self.pos, self.pos)
        dWx, dWy = self.gradKernelFunction(dx, dy)

        # Add Pressure contribution to accelerations
        ax = - (np.sum(self.m * (self.P / self.rho ** 2 + self.P.T / self.rho.T ** 2) * dWx, 1).reshape((self.N, 1)))
        ay = - (np.sum(self.m * (self.P / self.rho ** 2 + self.P.T / self.rho.T ** 2) * dWy, 1).reshape((self.N, 1)))

        # Add Gravity
        ay -= 9.8

        # Add viscosity
        aVisX, aVisY = self.getViscosityAcc()

        ax += aVisX
        ay += aVisY

        # pack together the acceleration components
        acc = np.hstack((ax, ay,))

        return acc

    def outMesh(self):

        """
        x               is a vector of x position
        y               is a vector of y position
        outPos          is a vector of index out of position
        N               is the number of total particles
        pas             is a matrix positions of total particles
        vel             is a matrix velocities of total particles
        acc             is a matrix accelerations of total particles
        """
        x = self.pos[:, 0]
        y = self.pos[:, 1]

        # looking for index which is out of positions
        outPosX = np.unique(np.append(np.asarray(np.where(self.Xstart > x)), np.asarray(np.where(self.Xend < x))))
        outPosY = np.unique(np.append(np.asarray(np.where(self.Ystart > y)), np.asarray(np.where(self.Yend < y))))
        outPos = np.unique(np.append(outPosX, outPosY))

        # delete particles out of positions
        self.N -= outPos.shape[0]
        self.vel = np.vstack((np.delete(self.vel[:, 0], outPos), np.delete(self.vel[:, 1], outPos))).T
        self.pos = np.vstack((np.delete(self.pos[:, 0], outPos), np.delete(self.pos[:, 1], outPos))).T
        self.acc = np.vstack((np.delete(self.acc[:, 0], outPos), np.delete(self.acc[:, 1], outPos))).T

    def addParticles(self, newParticles):

        """
        newParticles    is the number of particles that we add to the simulation basket
        newPos          is a matrix positions of new particles
        newVel          is a matrix velocities of new particles
        N               is the number of total particles
        pas             is a matrix positions of total particles
        vel             is a matrix velocities of total particles
        acc             is a matrix accelerations of total particles
        """

        # generate new particles
        # np.random.seed(14)  # set the random number generator seed
        newPos = np.random.rand(newParticles, 2)  # randomly selected positions and velocities
        newPos = 1 * (newPos - 0.5)
        newVel = np.zeros(newPos.shape)

        # add new particles to matrix
        self.N += newParticles
        self.pos = np.append(self.pos, newPos, axis=0)
        self.vel = np.append(self.vel, newVel, axis=0)
        self.acc = self.getAcc()

    def addRollerEffect(self, Roller1, Roller2):

        """
        Roller1         is a object of the Roller class
        Roller2         is a object of the Roller class
        CenterX         is the x position of the Roller object
        CenterY         is the y position of the Roller object
        radius          is radius of the Roller object
        AngularVelocity is angular velocity of the Roller object
        """

        # set the vector of each position
        x = self.pos[:, 0]
        y = self.pos[:, 1]

        # get the distance between the position of each particles
        D1 = np.sqrt((x - Roller1.CenterX) ** 2 + (y - Roller1.CenterY) ** 2)
        D2 = np.sqrt((x - Roller2.CenterX) ** 2 + (y - Roller2.CenterY) ** 2)

        # find the indexes which is on the each rollers
        onRoller1 = np.asarray(np.where(D1 <= Roller1.Radius)).flatten()
        onRoller2 = np.asarray(np.where(D2 <= Roller2.Radius)).flatten()

        # save colliding particles
        self.npColliding = np.zeros_like(self.npColliding)
        self.npColliding[onRoller1] = 1
        self.npColliding[onRoller2] = 1


        # change the velocities vector of the particles on the rollers.
        velx = self.vel[:, 0]
        vely = self.vel[:, 1]

        velx[onRoller1] = (y - Roller1.CenterY)[onRoller1] * -Roller1.AngularVelocity
        vely[onRoller1] = (Roller1.CenterX - x)[onRoller1] * -Roller1.AngularVelocity
        velx[onRoller2] = (y - Roller2.CenterY)[onRoller2] * -Roller2.AngularVelocity
        vely[onRoller2] = (Roller2.CenterX - x)[onRoller2] * -Roller2.AngularVelocity


        velx = velx.reshape(-1, 1)
        vely = vely.reshape(-1, 1)

        self.vel = np.hstack((velx, vely))

    def saveData(self, timestep):

        npW = np.expand_dims(self.w,axis=0)
        npX = np.expand_dims(self.pos[:, 0], axis=0)
        npY = np.expand_dims(self.pos[:, 1], axis=0)
        npVelx = np.expand_dims(self.vel[:, 0], axis=0)
        npVely = np.expand_dims(self.vel[:, 1], axis=0)
        npAccx = np.expand_dims(self.acc[:, 0], axis=0)
        npAccy = np.expand_dims(self.acc[:, 1], axis=0)
        nprho = np.expand_dims(self.rho[:,0], axis=0)
        npP = np.expand_dims(self.P[:,0], axis=0)
        npColliding = np.expand_dims(self.npColliding[:,0],axis=0)

        if timestep == 0:
            self.npWtarget = npW
            self.npXtarget = npX
            self.npYtarget = npY
            self.npUtarget = npVelx
            self.npVtarget = npVely
            self.npAccXtarget = npAccx
            self.npAccYtarget = npAccy
            self.npRhotareget = nprho
            self.npPtarget = npP
            self.npCollidingTotal = npColliding
            return
        self.npWtarget = np.append(self.npWtarget, npW, axis=0)
        self.npXtarget = np.append(self.npXtarget,npX,axis=0)
        self.npYtarget = np.append(self.npYtarget,npY,axis=0)
        self.npUtarget = np.append(self.npUtarget,npVelx,axis=0)
        self.npVtarget = np.append(self.npVtarget,npVely,axis=0)
        self.npAccXtarget = np.append(self.npAccXtarget,npAccx,axis=0)
        self.npAccYtarget = np.append(self.npAccYtarget,npAccy,axis=0)
        self.npRhotareget = np.append(self.npRhotareget,nprho,axis=0)
        self.npPtarget = np.append(self.npPtarget,npP,axis=0)
        self.npCollidingTotal = np.append(self.npCollidingTotal,npColliding,axis=0)


    def simulateParticles(self, Roller1, Roller2):

        plotRealTime = True  # switch on for plotting as the simulation goes along

        # prep figure
        fig = plt.figure(figsize=(10, 10), dpi=80)
        ax = plt.subplot()

        # Simulation Main Loop
        for timestep in range(self.TimeStep):

            # kick
            self.vel += self.acc * self.dt
            self.addRollerEffect(Roller1, Roller2)  # particles are not influenced on the roller area.

            # drift
            self.pos += self.vel * self.dt

            # update accelerations
            self.getAcc()

            # get density for plottiny
            rho = self.getDensity(self.pos, self.pos)

            # save dataset
            self.saveData(timestep)

            # plot in real time (only use when  you want to plot the simulation)
        #     if plotRealTime:
        #         plt.sca(ax)
        #         plt.cla()
        #         cval = np.maximum(rho , 0).flatten()
        #         plt.scatter(self.pos[:, 0], self.pos[:, 1], c=cval, s=10, alpha=0.5)
        #         ax.set(xlim=(-0.10, 0.10), ylim=(-0.20, 0.05))
        #         ax.set_aspect('equal', 'box')
        #         ax.set_xticks([-0.10, 0, 0.10])
        #         ax.set_yticks([-0.20, 0, 0.05])
        #         ax.set_facecolor('black')
        #         ax.set_facecolor((.1, .1, .1))
        #
        #         plt.pause(0.001)
        #
        # # Save figure
        # # plt.savefig(f'{saveAddress}/sph.png', dpi=240)
        plt.close(fig)
        # plt.show()


def main():
    return 0


if __name__ == "__main__":
    main()
