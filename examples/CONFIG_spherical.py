# -*- coding: utf-8 -*-
"""
Created in Apr 2020

@author: Stefan Haessler
"""
#%% Modules
#import copy

import numpy as np
import ARTcore.ModuleMirror as mmirror
import ARTcore.ModuleSupport as msupp
import ARTcore.ModuleProcessing as mp
import ARTcore.ModuleMask as mmask
import ARTcore.ModuleSource as mos
import ARTcore.ModuleOpticalChain as moc
import ART.ModuleAnalysisAndPlots as maap
import ARTcore.ModuleGeometry as mgeo
import ARTcore.ModuleDetector as mdet
import ART.ModuleTolerancing as mtol
from ART.ARTmain import run_ART
from copy import copy
import matplotlib.pyplot as plt
from scipy.stats import linregress
import ART.ModuleAnalysis as man
import time


#%%########################################################################
Wavelength = 45e-6  # 45 nm, central wavelength
PowerDistribution = mos.GaussianPowerDistribution(1, 2, 50e-3)
Positions = mos.PointRayOriginsDistribution(mgeo.Origin)
Directions = mos.ConeRayDirectionsDistribution(mgeo.Vector([1,0,0]), 5e-3) # 5 mrad
Source = mos.SimpleSource(Wavelength, PowerDistribution, Positions, Directions)

ChainDescription = "Single spherical mirror with short RoC, used to demagnify gas HHG"

# %% Define the optical elements

SupportSphMirror = msupp.SupportRound(50.8)
RoC = 600
SphMirror = mmirror.MirrorSpherical(SupportSphMirror, Radius=RoC)
SphMirrorSettings = {
    'OpticalElement' : SphMirror,
    'Distance' : 4000, # 4 meters from source
    'IncidenceAngle' : 3,
    'IncidencePlaneAngle' : 0,
    'Alignment': "support_normal",
    'Description' : "Spherical mirror",
}

Det = mdet.InfiniteDetector(-1)
Detectors = {
    "Focus": Det
}

OpticsList = [SphMirrorSettings]

AlignedOpticalElements = mp.OEPlacement(OpticsList) # Align the optical elements

AlignedOpticalChain = moc.OpticalChain(Source(2000), AlignedOpticalElements, Detectors, ChainDescription) # Create the optical chain

AlignedOpticalChain.get_output_rays()

rays= AlignedOpticalChain.get_output_rays()

Det.autoplace(rays[-1], 400)
Det.optimise_distance(AlignedOpticalChain.get_output_rays()[-1], [200,650], Det._spot_size, maxiter=10, tol=1e-16)


f,D = AlignedOpticalChain.drawSpotDiagram(ColorCoded="Delay")
AlignedOpticalChain.drawDelaySpots(0.1)
fig = AlignedOpticalChain.render(EndDistance=500, OEpoints=5000, cycle_ray_colors=True, impact_points=True, DetectedRays=True)
print(f"Beamline transmission: {round(AlignedOpticalChain.getETransmission(),3)}%")
