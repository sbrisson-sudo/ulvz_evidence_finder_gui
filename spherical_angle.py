#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 2022
@author: Sylvain Brisson sylvain.brisson@ens.fr
"""

import numpy as np
from numpy import cos,sin,pi
from numpy.core.umath_tests import inner1d
import matplotlib.pyplot as plt

import pandas as pd

import os

import scipy as sp

def _fast_cross(a, b):
    """
    This is a reimplementation of `numpy.cross` that only does 3D x
    3D, and is therefore faster since it doesn't need any
    conditionals.
    """

    cp = np.empty(np.broadcast(a, b).shape)
    aT = a.T
    bT = b.T
    cpT = cp.T

    cpT[0] = aT[1]*bT[2] - aT[2]*bT[1]
    cpT[1] = aT[2]*bT[0] - aT[0]*bT[2]
    cpT[2] = aT[0]*bT[1] - aT[1]*bT[0]

    return cp

def two_d(vec):
    """
    Reshape a one dimensional vector so it has a second dimension
    """
    shape = list(vec.shape)
    shape.append(1)
    shape = tuple(shape)
    return np.reshape(vec, shape)

def _cross_and_normalize(A, B):
    T = _fast_cross(A, B)
    # Normalization
    l = np.sqrt(np.sum(T ** 2, axis=-1))
    l = two_d(l)
    # Might get some divide-by-zeros
    with np.errstate(invalid='ignore'):
        TN = T / l
    # ... but set to zero, or we miss real NaNs elsewhere
    TN = np.nan_to_num(TN)
    return TN


def spherical_angle(A,B,C,degrees = True):
    """Compute the angle between the two great circles BA and BC, 
    Points position given in spherical (lat,lon in degrees) coordinates.
    """
    
    A = np.asanyarray(A)
    B = np.asanyarray(B)
    C = np.asanyarray(C)

    A, B, C = np.broadcast_arrays(A, B, C)
    
    A = np.asanyarray(_sph2cart(A))
    B = np.asanyarray(_sph2cart(B))
    C = np.asanyarray(_sph2cart(C))

    ABX = _fast_cross(A, B)
    ABX = _cross_and_normalize(B, ABX)
    BCX = _fast_cross(C, B)
    BCX = _cross_and_normalize(B, BCX)
    X = _cross_and_normalize(ABX, BCX)
    diff = inner1d(B, X)
    inner = inner1d(ABX, BCX)
    with np.errstate(invalid='ignore'):
        angle = np.arccos(inner)
    angle = np.where(diff < 0.0, (2.0 * np.pi) - angle, angle)

    if degrees:
        angle = np.rad2deg(angle)
        
    # return in [-pi,pi] rather than [0,2pi]
    if angle > 180.:
        angle = angle - 360.
        
    return angle

def spherical_angle_xyz(A,B,C,degrees = True):
    """Compute the angle between the two great circles BA and BC, 
    Points position given in cartesian coordinates.
    """
    
    A = np.asanyarray(A)
    B = np.asanyarray(B)
    C = np.asanyarray(C)

    A, B, C = np.broadcast_arrays(A, B, C)

    ABX = _fast_cross(A, B)
    ABX = _cross_and_normalize(B, ABX)
    BCX = _fast_cross(C, B)
    BCX = _cross_and_normalize(B, BCX)
    X = _cross_and_normalize(ABX, BCX)
    diff = inner1d(B, X)
    inner = inner1d(ABX, BCX)
    with np.errstate(invalid='ignore'):
        angle = np.arccos(inner)
    angle = np.where(diff < 0.0, (2.0 * np.pi) - angle, angle)

    if degrees:
        angle = np.rad2deg(angle)
        
    # return in [-pi,pi] rather than [0,2pi]
    if angle > 180.:
        angle = angle - 360.
        
    return angle
        
def _sph2cart(latlon):
    """Angles given in degrees"""
    
    lat,lon = latlon
    
    colat = 90. - lat 
    
    colat,lon = colat*pi/180., lon*pi/180.
    
    x = sin(colat)*cos(lon)
    y = sin(colat)*sin(lon)
    z = cos(colat)
    
    return x,y,z


        
        
# for testing

def read_receivers(file):
    
    """Load the receivers information from a receivers.dat or a receivers.csv"""

    data = pd.read_csv(file, header = 2, sep = "\s+")
    data.rename(columns = {'lon:':'lon'}, inplace = True)

    return data

if __name__ == "__main__":
    
    # test : read position stations with ULVZ at (60°lat, 0°lon) and event at (0°,0°)
    
    stations = read_receivers("test/receivers.dat")
    
    ulvz_latlon = (60., 0.)
    event_latlon = (0.,0.)
    
    ulvz_xyz = _sph2cart(ulvz_latlon)
    event_xyz = _sph2cart(event_latlon)

    for index, row in stations.iterrows():
        
        coord_latlon = (row["lat"], row["lon"])
        coord_xyz = _sph2cart((row["lat"], row["lon"]))
        angle = spherical_angle_xyz(ulvz_xyz, event_xyz, coord_xyz)
        
        angle = spherical_angle(ulvz_latlon, event_latlon, coord_latlon)
        
        print(row["lat"], row["lon"], angle)
        