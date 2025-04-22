# -*- coding: utf-8 -*-
"""
Created in Apr 2020
Heavily modified in December 2024

@author: Stefan Haessler + Andre Kalouguine
"""
#%% Modules
import ARTcore.ModuleMirror as mmirror
import ARTcore.ModuleSupport as msupp
import ARTcore.ModuleProcessing as mp
import ARTcore.ModuleMask as mmask
import ARTcore.ModuleSource as mos
import ARTcore.ModuleOpticalChain as moc
import ARTcore.ModuleGeometry as mgeo
import ARTcore.ModuleDetector as mdet
import ART 

import matplotlib.pyplot as plt
import numpy as np

# %% Source definition
N_rays = 1000

#Spectrum = mos.UniformSpectrum(lambdaMin=30e-6, lambdaMax=800e-6)
Wavelength = 50e-6  #single wavelength 50 nm
PowerDistribution = mos.GaussianPowerDistribution(1, 50e-3, 30e-3/2) #Gaussian power distrib with 50 micron source beam waist, and 15 mrad 1/e^2 divergence half-angle
Origins = mos.PointRayOriginsDistribution(mgeo.Origin) #all rays origins are at the same point, the origin [0,0,0]
Directions = mos.ConeRayDirectionsDistribution(mgeo.Vector([1,0,0]), 30e-3) #the ray directions are a cone with 30 mrad opening half-angle
Source = mos.SimpleSource(Wavelength, PowerDistribution, Origins, Directions) #build a simple ray-source out of the above distributions

SourceRays = Source(N_rays)

# %% Define the optical elements
SupportMask = msupp.SupportRoundHole(Radius=30, RadiusHole=15, CenterHoleX=0, CenterHoleY=0) 
Mask = mmask.Mask(SupportMask)
MaskSettings = {
    'OpticalElement' : Mask,
    'Distance' : 400,
    'IncidenceAngle' : 0,
    'IncidencePlaneAngle' : 0,
    'Description' : "Mask for selecting rays",
    'Alignment' : 'support_normal',
}

Focal = 500
AngleIncidence = 80 #in deg
OptimalMajorRadius, OptimalMinorRadius = mmirror.ReturnOptimalToroidalRadii(Focal, AngleIncidence)

SupportToroidal = msupp.SupportRectangle(150, 32)

ToroidalMirrorA = mmirror.MirrorToroidal(SupportToroidal, OptimalMajorRadius, OptimalMinorRadius)
ToroidalASettings = {
    'OpticalElement' : ToroidalMirrorA,
    'Distance' : Focal-MaskSettings['Distance'],
    'IncidenceAngle' : AngleIncidence,
    'IncidencePlaneAngle' : 0,
    'Description' : "First parabola for collimation",
}

ToroidalMirrorB = mmirror.MirrorToroidal(SupportToroidal,OptimalMajorRadius, OptimalMinorRadius)
ToroidalBSettings = {
    'OpticalElement' : ToroidalMirrorB,
    'Distance' : None,
    'IncidenceAngle' : AngleIncidence,
    'IncidencePlaneAngle' : 180,
    'Description' : "First parabola for collimation",
}

Det = mdet.InfiniteDetector(-1)
Detectors = {
    "Focus": Det
}

ChainDescription = "2 toroidal mirrors in f-d-f config, i.e. approx. collimation, propagation, and the refocus "

Distances = np.linspace(Focal-200, Focal+200, 20)
FocalDistances = []
FocalSizes = []

for d in Distances:
    ToroidalBSettings['Distance'] = d
    print(d)
    AlignedOpticalElements = mp.OEPlacement([MaskSettings, ToroidalASettings, ToroidalBSettings])
    AlignedOpticalChain = moc.OpticalChain(Source(1000), AlignedOpticalElements, Detectors, ChainDescription)
    RayListAnalysed = AlignedOpticalChain.get_output_rays()[-1]
    Det.autoplace(RayListAnalysed, 390)
    Det.optimise_distance(RayListAnalysed, [200,600], Det._spot_size, maxiter=10, tol=1e-14)
    FocalDistances.append(Det.distance)
    DetectorPointList2D = AlignedOpticalChain.get2dPoints()
    DetectorPointList2DCentre = DetectorPointList2D - np.mean(DetectorPointList2D, axis=0)
    FocalSpotSizeSD = np.std(DetectorPointList2DCentre.norm)
    FocalSizes.append(FocalSpotSizeSD)

optimalDistance = Distances[np.argmin(FocalSizes)]

fig, ax = plt.subplots()
ax.plot(Distances, FocalSizes)
ax.set_xlabel('Distance between PM and Toroidal B [mm]')
ax.set_ylabel('Spot size [mm]')
ax.scatter(optimalDistance, np.min(FocalSizes), color='red')
plt.tight_layout()

ToroidalBSettings['Distance'] = optimalDistance
AlignedOpticalElements = mp.OEPlacement([MaskSettings, ToroidalASettings, ToroidalBSettings])
AlignedOpticalChain = moc.OpticalChain(Source(5000), AlignedOpticalElements, Detectors, ChainDescription)
rays= AlignedOpticalChain.get_output_rays()
Det.autoplace(rays[-1], FocalDistances[np.argmin(FocalSizes)])
Det.optimise_distance(rays[-1], [200,600], Det._spot_size, maxiter=10, tol=1e-14)

AlignedOpticalChain.render(EndDistance=Det.distance+10, OEpoints=5000, cycle_ray_colors=True, impact_points=True, DetectedRays=True)
AlignedOpticalChain.drawSpotDiagram(ColorCoded="Delay")
AlignedOpticalChain.drawCaustics()

print("Optimum found for following parameters:")
print(AlignedOpticalChain)
plt.show()