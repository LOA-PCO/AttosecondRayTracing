"""
Provides functions for analysing the alignment tolerance of a system.

Mostly it consists in mis-aligning the optical elements of the system and checking the impact on the focal spot and delays.

There are two parts to this module:
- Misaligning/deteriorating the optical elements of the system
- Analysing the impact of these misalignments on the focal spot and delays

The end goal is to have a simple function "GetTolerance" that will return the tolerance of the system to misalignments of each optical element.

Created in Nov 2024

@author: Andre Kalouguine
"""
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from scipy.stats import linregress

import ARTcore.ModuleOpticalChain as moc


def GetTolerance(OpticalChain,
                 time_error=1.0,
                 Detector = "Focus",
                 n_iter = 5,
                 elements_to_misalign = None):
    """
    Returns the tolerance of the system to misalignments of each optical element.
    It first constructs a normalisation vector:
    For each optical element, for each degree of freedom, it iteratively calculates the required amplitude of misalignment to reach the time_error.
    This gives a n-dimensional vector. The smaller the value, the more sensitive the system is to misalignments of this axis.
    """
    # Define the normalisation vector
    normalisation_vector = np.zeros(len(OpticalChain.optical_elements) * 6)*np.nan
    durations = np.zeros(len(OpticalChain.optical_elements) * 6)*np.nan
    misalignments = ["rotate_roll_by", "rotate_pitch_by", "rotate_yaw_by", "shift_along_normal", "shift_along_major", "shift_along_cross"]
    if elements_to_misalign is None:
        elements_to_misalign = range(len(OpticalChain.optical_elements))
    if isinstance(Detector, str):
        Det = OpticalChain.detectors[Detector]
    else:
        Det = Detector
    for i in elements_to_misalign:
        for j in range(6):
            # Misalign the optical element
            amplitude = 1e-3
            for k in range(n_iter):
                try:
                    misaligned_optical_chain = copy(OpticalChain)
                    r_before = misaligned_optical_chain[i].r
                    q_before = misaligned_optical_chain[i].q
                    misaligned_optical_chain[i].__getattribute__(misalignments[j])(amplitude)
                    r_after = misaligned_optical_chain[i].r
                    q_after = misaligned_optical_chain[i].q
                    rays = misaligned_optical_chain.get_output_rays()
                    Det.optimise_distance(rays[Det.index], [Det.distance-100, Det.distance+100], Det._spot_size, maxiter=10, tol=1e-10)
                except:
                    print(f"OE {i} failed to misalign {misalignments[j]} by {amplitude}")
                    amplitude /= 10
                    continue
                rays = misaligned_optical_chain.get_output_rays(force=True)
                duration = np.std(Det.get_Delays(rays[Det.index]))
                if len(rays[Det.index]) <= 50:
                    amplitude /= 2
                    continue
                if duration > time_error:
                    amplitude /= 3
                elif duration < time_error / 10:
                    amplitude *= 3
                elif duration < time_error / 1.5:
                    amplitude *= 1.2
                else:
                    break
            if not (time_error/2 < duration < time_error*2):
                print(f"OE {i} failed to misalign {misalignments[j]} by {amplitude}: duration = {duration}")
                if duration > time_error:
                    amplitude = np.nan
                else:
                    amplitude = 0.1
            normalisation_vector[i*6+j] = amplitude
            durations[i*6+j] = duration
    return normalisation_vector, durations
