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
from ART.ARTmain import run_ART
from copy import copy
import matplotlib.pyplot as plt
from scipy.stats import linregress
import ART.ModuleAnalysis as man
import time


#%%########################################################################
Spectrum = mos.UniformSpectrum(lambdaMin=30e-6, lambdaMax=800e-6)
#Spectrum = mos.SingleWavelengthSpectrum(800e-6)
PowerDistribution = mos.GaussianPowerDistribution(1, 2, 50e-3)
Positions = mos.PointRayOriginsDistribution(mgeo.Origin)
Directions = mos.ConeRayDirectionsDistribution(mgeo.Vector([1,0,0]), 50e-3)
Source = mos.SimpleSource(Spectrum, PowerDistribution, Positions, Directions)

ChainDescription = "2 equal large-off-axis-angle parabolas for collimation and refocusing "

# %% Define the optical elements
SupportMask = msupp.SupportRoundHole(30, 6.25, 0, 0)
Mask = mmask.Mask(SupportMask)
MaskSettings = {
    'OpticalElement' : Mask,
    'Distance' : 200,
    'IncidenceAngle' : 0,
    'IncidencePlaneAngle' : 0,
    'Description' : "Mask for blocking rays",
    'Alignment' : 'support_normal',
}


SupportCollimatingParabola = msupp.SupportRectangle(35,35)
offAxisAngle = 150 #in deg
FocalEffective = 400 # in mm
CollimatingParabola = mmirror.MirrorParabolic(SupportCollimatingParabola, FocalEffective=FocalEffective, OffAxisAngle=offAxisAngle)
CollimatingParabolaSettings = {
    'OpticalElement' : CollimatingParabola,
    'Distance' : FocalEffective-MaskSettings['Distance'],
    'IncidenceAngle' : 0,
    'IncidencePlaneAngle' : 0,
    'Alignment': "towards_focusing",
    'Description' : "First parabola for collimation",
}

SupportPlane = msupp.SupportRound(50.8)
PlaneMirror = mmirror.MirrorPlane(SupportPlane)
PlaneMirrorSettings = {
    'OpticalElement' : PlaneMirror,
    'Distance' : 300,
    'IncidenceAngle' : 75,
    'IncidencePlaneAngle' : 0,
    'Description' : "Plane mirror for reflection",
    'Alignment' : 'support_normal',
}

SupportFocusingParabola = msupp.SupportRectangle(35,35)
offAxisAngle = 150 #in deg
FocalEffective = 400 # in mm
FocusingParabola = mmirror.MirrorParabolic(SupportFocusingParabola, FocalEffective=FocalEffective, OffAxisAngle=offAxisAngle)
FocusingParabolaSettings = {
    'OpticalElement' : FocusingParabola,
    'Distance' : 300,
    'IncidenceAngle' : 0,
    'IncidencePlaneAngle' : 0,
    'Description' : "Second parabola for refocusing",
    'Alignment' : 'support_normal',
}

Det = mdet.InfiniteDetector(-1)
Detectors = {
    "Focus": Det
}

OpticsList = [MaskSettings,CollimatingParabolaSettings, PlaneMirrorSettings, FocusingParabolaSettings]

AlignedOpticalElements = mp.OEPlacement(OpticsList) # Align the optical elements

AlignedOpticalChain = moc.OpticalChain(Source(2000), AlignedOpticalElements, Detectors, ChainDescription) # Create the optical chain

AlignedOpticalChain.get_output_rays()

rays= AlignedOpticalChain.get_output_rays()

Det.autoplace(rays[-1], 410)
Det.optimise_distance(AlignedOpticalChain.get_output_rays()[-1], [200,600], Det._spot_size, maxiter=10, tol=1e-16)


AlignedOpticalChain.drawSpotDiagram()
AlignedOpticalChain.render(EndDistance=500, OEpoints=5000, cycle_ray_colors=True, impact_points=True, DetectedRays=True)
print(f"Beamline transmission: {round(AlignedOpticalChain.getETransmission(),3)}%")