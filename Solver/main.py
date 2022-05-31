"""
Reference:

* coding style
Philip Mocz (2020) Princeton Univeristy, @PMocz
https://github.com/pmocz/sph-python

* kernel function
Smoothed Particle Hydrodynamics
Techniques for the Physics Based Simulation of Fluids and Solids
Dan Koschier1, Jan Bender2, Barbara Solenthaler3, and Matthias Teschner4
"""
import os


def main():

    import numpy as np
    from Particles import Particles
    from FluidProperties import FluidProperties
    from Time import Time
    from Mesh import Mesh
    from Roller import Roller
    import time as SimulationTime

    # simulation start
    start =SimulationTime.time()


    # smoothed length
    h=0.01
    # overview
    NData = 100
    NParticles = 100
    NTimeSteps = 200
    rollRadius = 0.048

    # set the random number generator seed
    np.random.seed(42)

    # random parameters
    rnMass = np.random.rand(NData) * 0.0001 + 0.0001
    rnLiquidViscosity = (np.random.rand(NData) * 3)* 10**-7
    rnAngularVelocity1 = np.random.rand(NData) * 30 + 10
    rnAngularVelocity2 = np.random.rand(NData) * 30 + 10

    # datasets
    npInputDataset = None
    npXTargetDataset = None
    npYTargetDataset = None
    npUTargetDataset = None
    npVTargetDataset = None
    npAccXTargetDataset = None
    npAccYTargetDataset = None
    npRhoTargetDataset = None
    npPTargetDataset = None
    npCollidingTotal = None

    os.makedirs(f"./DATA", exist_ok=True)

    for i in range(NData):

        print("NData",i)

        # set the random number generator seed
        np.random.seed(42)

        # simulation parameters
        time = Time(start_time=0, end_time=0.2, time_step=NTimeSteps)
        liquid_property = FluidProperties(mass=rnMass[i], SmoothingLength=h, LiquidViscosity=rnLiquidViscosity[i], LiquidDropletRadius=1,
                                          restDensity=1000, stiffnessConstant=1)
        mesh = Mesh(start_x=-0.10, end_x= 0.10 , start_y= -0.20, end_y= 0.05)
        Roller1 = Roller(CenterX=0.05, CenterY=-0.06, Radius=rollRadius, AngularVelocity= rnAngularVelocity1[i])
        Roller2 = Roller(CenterX=-0.05, CenterY=-0.06, Radius=rollRadius, AngularVelocity=rnAngularVelocity2[i])
        particles = Particles(time=time, LiquidProperty=liquid_property,Mesh=mesh,)

        # Generate Initial Conditions
        particles.generateInitConditions(startGenerateX=-0.020, endGenerateX=0.020, startGenerateY=-0.020, endGenerateY=0.020, column=10, row=10)

        # add Input initial x,y data
        X = np.linspace(-0.020, 0.020, 10).reshape(10,1)
        Y = np.linspace(-0.020, 0.020, 10).reshape(10,1)
        posX = (X * np.ones_like(Y).T).flatten().reshape(-1,1)
        posY = (Y * np.ones_like(X).T).T.flatten().reshape(1,-1)


        # calculate initial gravitational accelerations
        particles.acc = particles.getAcc()

        # simulation
        particles.simulateParticles(Roller1=Roller1, Roller2=Roller2)

        # Input data
        npInput = np.full((NTimeSteps,NParticles,1), rnMass[i])
        npInput = np.append(npInput, np.full((NTimeSteps,NParticles,1),rnLiquidViscosity[i]), axis=-1)
        npInput = np.append(npInput,np.full((NTimeSteps,NParticles,1), rnAngularVelocity1[i] * rollRadius), axis=-1)# "*rollRadius" required to transfer to rotating speed
        npInput = np.append(npInput, np.full((NTimeSteps,NParticles,1), rnAngularVelocity2[i]* rollRadius), axis=-1)
        npInput = np.expand_dims(npInput,axis=0)

        # Target data

        npXTarget = np.expand_dims(particles.npXtarget,axis=0)
        npYTarget = np.expand_dims(particles.npYtarget,axis=0)
        npUTarget = np.expand_dims(particles.npUtarget,axis=0)
        npVTarget = np.expand_dims(particles.npVtarget,axis=0)
        npAccXTarget = np.expand_dims(particles.npAccXtarget,axis=0)
        npAccYTarget = np.expand_dims(particles.npAccYtarget,axis=0)
        npRhoTarget = np.expand_dims(particles.npRhotareget,axis=0)
        npPTarget = np.expand_dims(particles.npPtarget,axis=0)
        npCollinding = np.expand_dims(particles.npCollidingTotal, axis=0)

        # save as dataset
        if i == 0:
            npInputDataset = npInput
            npXTargetDataset = npXTarget
            npYTargetDataset = npYTarget
            npUTargetDataset = npUTarget
            npVTargetDataset = npVTarget
            npAccXTargetDataset = npAccXTarget
            npAccYTargetDataset = npAccYTarget
            npRhoTargetDataset = npRhoTarget
            npPTargetDataset = npPTarget
            npCollidingTotal = npCollinding
            continue

        npInputDataset = np.append(npInputDataset, npInput, axis=0)
        npXTargetDataset = np.append(npXTargetDataset, npXTarget,axis=0)
        npYTargetDataset = np.append( npYTargetDataset, npYTarget,axis=0)
        npUTargetDataset = np.append(npUTargetDataset ,npUTarget,axis=0)
        npVTargetDataset = np.append(npVTargetDataset, npVTarget,axis=0)
        npAccXTargetDataset = np.append(npAccXTargetDataset,npAccXTarget,axis=0)
        npAccYTargetDataset = np.append(npAccYTargetDataset,npAccYTarget,axis=0)
        npRhoTargetDataset = np.append(npRhoTargetDataset,npRhoTarget,axis=0)
        npPTargetDataset = np.append(npPTargetDataset ,npPTarget,axis=0)
        npCollidingTotal = np.append(npCollidingTotal,npCollinding,axis=0)

    # save arrays

    np.save(f"./DATA/npInputDataset.npy", npInputDataset)
    np.save(f"./DATA/npXTargetDataset.npy", npXTargetDataset)
    np.save(f"./DATA/npYTargetDataset.npy", npYTargetDataset)
    np.save(f"./DATA/npUTargetDataset.npy", npUTargetDataset)
    np.save(f"./DATA/npVTargetDataset.npy", npVTargetDataset)
    np.save(f"./DATA/npAccxTargetDataset.npy", npAccXTargetDataset)
    np.save(f"./DATA/npAccYTargetDataset.npy", npAccYTargetDataset)
    np.save(f"./DATA/npRhoTargetDataset.npy", npRhoTargetDataset)
    np.save(f"./DATA/npPTargetDataset.npy", npPTargetDataset)
    np.save(f"./DATA/npCollidingTotal.npy", npCollidingTotal)
    np.save(f"./DATA/npRotatingVL.npy", rnAngularVelocity1*rollRadius)
    np.save(f"./DATA/npRotatingVR.npy", rnAngularVelocity2*rollRadius)


    # simulation end
    end = SimulationTime.time()

    print(f"simulation time: {end -start:.2f} ")
    return 0


if __name__ == "__main__":
    main()
