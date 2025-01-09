"""
Adds plotting and visualisation methods to various classes from ARTcore.
Most of the code is simply moved here from the class definitions previously.

Created in July 2024

@author: André Kalouguine + Stefan Haessler + Anthony Guillaume
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import pyvista as pv
import pyvistaqt as pvqt
import colorcet as cc

import ARTcore.ModuleSupport as msup
import ARTcore.ModuleMirror as mmir
import ARTcore.ModuleGeometry as mgeo
import ARTcore.ModuleMask as mmask
from ARTcore.ModuleGeometry import Point, Vector, Origin
import itertools

# %% Adding support drawing for mirror projection

def SupportRound_ContourSupport(self, Figure):
    """Draw support contour in MirrorProjection plots."""
    axe = Figure.add_subplot(111, aspect="equal")
    axe.add_patch(patches.Circle((0, 0), self.radius, alpha=0.08))
    return axe

msup.SupportRound._ContourSupport = SupportRound_ContourSupport

def SupportRoundHole_ContourSupport(self, Figure):
    """Draws support contour in MirrorProjection plots."""
    axe = Figure.add_subplot(111, aspect="equal")
    axe.add_patch(patches.Circle((0, 0), self.radius, alpha=0.08))
    axe.add_patch(patches.Circle((self.centerholeX, self.centerholeY), self.radiushole, color="white", alpha=1))
    return axe

msup.SupportRoundHole._ContourSupport = SupportRoundHole_ContourSupport

def SupportRectangle_ContourSupport(self, Figure):
    """Draws support contour in MirrorProjection plots."""
    axe = Figure.add_subplot(111, aspect="equal")
    axe.add_patch(patches.Rectangle((-self.dimX * 0.5, -self.dimY * 0.5), self.dimX, self.dimY, alpha=0.08))
    return axe

msup.SupportRectangle._ContourSupport = SupportRectangle_ContourSupport

def SupportRectangleHole_ContourSupport(self, Figure):
    """Draws support contour in MirrorProjection plots."""
    axe = Figure.add_subplot(111, aspect="equal")
    axe.add_patch(patches.Rectangle((-self.dimX * 0.5, -self.dimY * 0.5), self.dimX, self.dimY, alpha=0.08))
    axe.add_patch(patches.Circle((self.centerholeX, self.centerholeY), self.radiushole, color="white", alpha=1))
    return axe

msup.SupportRectangleHole._ContourSupport = SupportRectangleHole_ContourSupport

def SupportRectangleRectHole_ContourSupport(self, Figure):
    """Draws support contour in MirrorProjection plots."""
    axe = Figure.add_subplot(111, aspect="equal")
    axe.add_patch(patches.Rectangle((-self.dimX * 0.5, -self.dimY * 0.5), self.dimX, self.dimY, alpha=0.08))
    axe.add_patch(
        patches.Rectangle(
            (-self.holeX * 0.5 + self.centerholeX, -self.holeY * 0.5 + self.centerholeY),
            self.holeX,
            self.holeY,
            color="white",
            alpha=1,
        )
    )
    return axe

msup.SupportRectangleRectHole._ContourSupport = SupportRectangleRectHole_ContourSupport

# %% Support sampling and meshing 

def sample_support(Support, Npoints):
    """
    This function samples a regular grid on the support and filters out the points that are outside of the support.
    For that it uses the _estimate_size method of the support.
    It then generates a regular grid over that area and filters out the points that are outside of the support.
    If less than half of the points are inside the support, it will increase the number of points by a factor of 2 and try again.
    """
    if hasattr(Support, "size"):
        size = Support.size
    else:
        size_x = Support._estimate_size()
        size = np.array([2*size_x, 2*size_x])
    enough = False
    while not enough:
        X = np.linspace(-size[0] * 0.5, size[0] * 0.5, int(np.sqrt(Npoints)))
        Y = np.linspace(-size[1] * 0.5, size[1] * 0.5, int(np.sqrt(Npoints)))
        X, Y = np.meshgrid(X, Y)
        Points = np.array([X.flatten(), Y.flatten()]).T
        Points = [p for p in Points if p in Support]
        if len(Points) > Npoints * 0.5:
            enough = True
        else:
            Npoints *= 2
    return mgeo.PointArray(Points)

def sample_z_values(Support, Npoints):
    Npoints_init = Npoints
    if hasattr(Support, "size"):
        size = Support.size
    else:
        size_x = Support._estimate_size()
        size = np.array([2*size_x, 2*size_x])
    enough = False
    while not enough:
        X = np.linspace(-size[0] * 0.5, size[0] * 0.5, int(np.sqrt(Npoints)))
        Y = np.linspace(-size[1] * 0.5, size[1] * 0.5, int(np.sqrt(Npoints)))
        X, Y = np.meshgrid(X, Y)
        Points = np.array([X.flatten(), Y.flatten()]).T
        Points = [p for p in Points if p in Support]
        if len(Points) > Npoints_init * 0.5:
            enough = True
        else:
            Npoints *= 2
    Points = np.array([X.flatten(), Y.flatten()]).T
    mask = np.array([p in Support for p in Points]).reshape(X.shape)
    return X,Y,mask


# %% Mesh/pointcloud generation for different mirror/mask types

# For the mirrors that are already implemented in ARTcore
# we can use the method _zfunc to render the mirror surface. 
# For that we sample the support and then use the _zfunc method to get the z-values of the surface.
# We then use the pyvista function add_mesh to render the surface.

def _RenderMirror(Mirror, Npoints=10000):
    """
    This function renders a mirror in 3D.
    It samples the support of the mirror and uses the _zfunc method to get the z-values of the surface.
    It draws a small sphere at each point of the support and connects them to form a surface.
    """
    Points = sample_support(Mirror.support, Npoints)
    Points += Mirror.r0[:2]
    Z = Mirror._zfunc(Points)
    Points = mgeo.PointArray([Points[:, 0], Points[:, 1], Z]).T
    Points = Points.from_basis(*Mirror.basis)
    mesh = pv.PolyData(Points)
    return mesh


def _RenderMirrorSurface(Mirror, Npoints=10000):
    """
    This function renders a mirror in 3D.
    It samples the support of the mirror and uses the _zfunc method to get the z-values of the surface.
    It draws a small sphere at each point of the support and connects them to form a surface.
    """
    X,Y,mask = sample_z_values(Mirror.support, Npoints)
    shape = X.shape
    X += Mirror.r0[0]
    Y += Mirror.r0[1]
    Z = Mirror._zfunc(mgeo.PointArray([X.flatten(), Y.flatten()]).T).reshape(X.shape)
    #Z += Mirror.r0[2]
    Z[~mask] = np.nan
    Points = mgeo.PointArray([X.flatten(), Y.flatten(), Z.flatten()]).T
    Points = Points.from_basis(*Mirror.basis)
    X,Y,Z = Points.T
    X = X.reshape(shape)
    Y = Y.reshape(shape)
    Z = Z.reshape(shape)
    mesh = pv.StructuredGrid(X, Y, Z)
    return mesh

mmir.MirrorPlane._Render = _RenderMirror
mmir.MirrorParabolic._Render = _RenderMirror
mmir.MirrorCylindrical._Render = _RenderMirror
mmir.MirrorEllipsoidal._Render = _RenderMirror
mmir.MirrorSpherical._Render = _RenderMirror
mmir.MirrorToroidal._Render = _RenderMirror

mmask.Mask._Render = _RenderMirror

mmir.MirrorPlane._Render = _RenderMirrorSurface
mmir.MirrorParabolic._Render = _RenderMirrorSurface
mmir.MirrorCylindrical._Render = _RenderMirrorSurface
mmir.MirrorEllipsoidal._Render = _RenderMirrorSurface
mmir.MirrorSpherical._Render = _RenderMirrorSurface
mmir.MirrorToroidal._Render = _RenderMirrorSurface

mmask.Mask._Render = _RenderMirrorSurface

# %% Optical element rendering
def _RenderOpticalElement(fig, OE, OEpoints, draw_mesh = False, color="blue", index=0):
    """
    This function renders the optical elements in the 3D plot.
    It receives a pyvista figure handle, and draws the optical element on it.
    """
    mesh = OE._Render(OEpoints)
    fig.add_mesh(mesh, color = color, name = f"{OE.description} {index}")    

def _RenderDetector(fig, Detector, size = 40, name = "Focus", detector_meshes = None):
    """
    Unfinished
    """
    Points = [np.array([-size/2,size/2,0]),np.array([size/2,size/2,0]),np.array([size/2,-size/2,0]),np.array([-size/2,-size/2,0])]
    Points = mgeo.PointArray(Points)
    Points = Points.from_basis(*Detector.basis)
    Rect = pv.Rectangle(Points[:3])
    if detector_meshes is not None:
        detector_meshes += [Rect]
    fig.add_mesh(Rect, color="green", name=f"Detector {name}")

# %% Support rendering using the sdf method

def _RenderSupport(Support, Npoints=10000):
    """
    This function renders a support in 3D.
    """
    X,Y,mask = sample_z_values(Support, Npoints)
    shape = X.shape
    Z = np.zeros_like(X)
    Z[~mask] = np.nan
    mesh = pv.StructuredGrid(X, Y, Z)
    return mesh
    
msup.SupportRound._Render = _RenderSupport
msup.SupportRoundHole._Render = _RenderSupport
msup.SupportRectangle._Render = _RenderSupport
msup.SupportRectangleHole._Render = _RenderSupport
msup.SupportRectangleRectHole._Render = _RenderSupport


# %% Standalone optical element rendering

def _RenderMirror(Mirror, Npoints=1000, draw_support = False, draw_points = False, draw_vectors = True , recenter_support = True):
    """
    This function renders a mirror in 3D.
    """
    mesh = Mirror._Render(Npoints)
    p = pv.Plotter()
    p.add_mesh(mesh)
    if draw_support:
        support = Mirror.support._Render()
        if recenter_support:
            support.translate(Mirror.r0,inplace=True)
        p.add_mesh(support, color="gray", opacity=0.5)
    
    if draw_vectors:
        # We draw the important vectors of the optical element
        # For that, if we have a "vectors" attribute, we use that
        #  (a dictionary with the vector names as keys and the colors as values)
        # Otherwise we use the default vectors: "support_normal", "majoraxis"
        if hasattr(Mirror, "vectors"):
            vectors = Mirror.vectors
        else:
            vectors = {"support_normal_ref": "red", "majoraxis_ref": "blue"}
        for vector, color in vectors.items():
            if hasattr(Mirror, vector):
                p.add_arrows(Mirror.r0, 10*getattr(Mirror, vector), color=color)
    
    if draw_points:
        # We draw the important points of the optical element
        # For that, if we have a "points" attribute, we use that
        #  (a dictionary with the point names as keys and the colors as values)
        # Otherwise we use the default points: "centre_ref"
        if hasattr(Mirror, "points"):
            points = Mirror.points
        else:
            points = {"centre_ref": "red"}
        for point, color in points.items():
            if hasattr(Mirror, point):
                p.add_mesh(pv.Sphere(radius=1, center=getattr(Mirror, point)), color=color)
    else:
        p.add_mesh(pv.Sphere(radius=1, center=Mirror.r0), color="red")
    p.show()
    return p

mmir.MirrorPlane.render = _RenderMirror
mmir.MirrorParabolic.render = _RenderMirror
mmir.MirrorParabolic.vectors = {
    "support_normal": "red",
    "majoraxis": "blue",
    "towards_focusing_ref": "green"
}
mmir.MirrorParabolic.points = {
    "centre_ref": "red",
    "focus_ref": "green"
}
mmir.MirrorCylindrical.render = _RenderMirror
mmir.MirrorEllipsoidal.render = _RenderMirror
mmir.MirrorEllipsoidal.vectors = {
    "support_normal_ref": "red",
    "majoraxis_ref": "blue",
    "towards_image_ref": "green",
    "towards_object_ref": "green",
    "centre_normal_ref": "purple"}

mmir.MirrorSpherical.render = _RenderMirror
mmir.MirrorToroidal.render = _RenderMirror

mmask.Mask.render = _RenderMirror

# %% Ray rendering

def _RenderRays(RayListHistory, EndDistance, maxRays=150):
    """
    Generates a list of Pyvista-meshes representing the rays in RayListHistory.
    This can then be employed to render the rays in a Pyvista-figure.

    Parameters
    ----------
        RayListHistory : list(list(Ray))
            A list of lists of objects of the ModuleOpticalRay.Ray-class.

        EndDistance : float
            The rays of the last ray bundle are drawn with a length given by EndDistance (in mm).

        maxRays : int, optional
            The maximum number of rays to render. Rendering all the traced rays is a insufferable resource hog
            and not required for a nice image. Default is 150.
    
    Returns
    -------
        meshes : list(pvPolyData)
            List of Pyvista PolyData objects representing the rays
    """
    meshes = []
    # Ray display
    for k in range(len(RayListHistory)):
        x = []
        y = []
        z = []
        if k != len(RayListHistory) - 1:
            knums = list(
                map(lambda x: x.number, RayListHistory[k])
            )  # make a list of all ray numbers that are still in the game
            if len(RayListHistory[k + 1]) > maxRays:
                rays_to_render = np.random.choice(RayListHistory[k + 1], maxRays, replace=False)
            else:
                rays_to_render = RayListHistory[k + 1]

            for j in rays_to_render:
                indx = knums.index(j.number)
                i = RayListHistory[k][indx]
                Point1 = i.point
                Point2 = j.point
                x += [Point1[0], Point2[0]]
                y += [Point1[1], Point2[1]]
                z += [Point1[2], Point2[2]]

        else:
            if len(RayListHistory[k]) > maxRays:
                rays_to_render = np.random.choice(RayListHistory[k], maxRays, replace=False)
            else:
                rays_to_render = RayListHistory[k]

            for j in rays_to_render:
                Point = j.point
                Vector = j.vector
                x += [Point[0], Point[0] + Vector[0] * EndDistance]
                y += [Point[1], Point[1] + Vector[1] * EndDistance]
                z += [Point[2], Point[2] + Vector[2] * EndDistance]
        points = np.column_stack((x, y, z))
        meshes += [pv.line_segments_from_points(points)]
    return meshes

