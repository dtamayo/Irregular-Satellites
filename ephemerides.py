import spiceypy as spice
import numpy as np
import astropy.units as u
from astropy.constants import c
kms = u.km/u.s
AUday = u.AU/u.day

def load_spice_kernels():
    '''
    Loads required spice kernels. Must be called prior to calling any spice functions.
    Loads de435.bsp for the planet barycenter ephemerides, 310 for Jupiter and the Galilean 
    satellites individually, 341 for known irregular satellites.
    Last kernel loaded has priority in the case of conflicts--did not find noticeable 
    differences when permuting the order for our applications.
    '''
    spice.kclear()
    spice.furnsh('data/naif0009.tls.txt')
    spice.furnsh('data/jup341.bsp')
    spice.furnsh('data/jup310.bsp')
    spice.furnsh('data/de435.bsp')

def build_spice_get_target(target, cor, obs):
    """
    We want the location of a target object (target) as seen from a specific observer (obs).
    This is a factory function that builds a function for doing that using SPICE.
    Specifically, returns a function spice_get_target(t), which when passed a time (as a julian date JD) 
    returns the targets' xyz, vxvyvz, and (RA, Dec) relative to the observer (obs).
    
    Arguments:
    
    target (str): NAIF code for the object we are trying to observe. Can be object name, or its
    corresponding NAIF code (https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html)

    cor (str): Corrections for the apparent position of the object. Available arguments are 'NONE'
    (actual geometric position), 'LT' (correction for light travel time), 'LT+S' (correction for
    light travel time and stellar aberration).
    
    obs (str): NAIF code for the observer. Can be object name, or its corresponding NAIF code
    (https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html). Can't take observatory
    codes, see 

    Returns:
    
    spice_get_target (function): Function for getting target position/velocity/(RA, Dec) using SPICE.
    See docstring below for spice_get_target in ephemerides.py
    """    
    
    def spice_get_target(t):
        """
        Returns target's observed position vector, velocity vector, and (RA/Dec)
        
        Arguments:
        
        t (float): Time of observation, expressed as a Julian Date (JD) 
        
        Returns:
    
        xyz (numpy array): Array of target's observed position vector [x, y, z] in AU
        (relative to the observer specified in build_spice_get_target)
        
        vxvyvz (numpy array): Array of target's observed velocity vector [vx, vy, vz] in AU/day
        (relative to the observer specified in build_spice_get_target)
        
        radec (numpy array): Array of target's observed right ascension / declination [RA, Dec] in deg
        (relative to the observer specified in build_spice_get_target)
        """
        
        state,lighttime = spice.spkezr(target,t,'J2000',cor,obs)
        pos,lighttime = spice.spkpos(target,t,'J2000',cor,obs)
        dist,ra,dec = spice.recrad(pos) 
        xyz = np.array([state[0],state[1],state[2]])*u.km.to(u.AU)
        vxvyvz = np.array([state[3],state[4],state[5]])*kms.to(AUday)
        radec = np.array([ra,dec])*u.rad.to(u.deg)
        return xyz, vxvyvz, radec
    return spice_get_target

def CordConv(xyz):
    '''
    This function takes in a position vector of a body relative to an observer and returns a radec.

    Arguments:

    xyz: numpy array

    Should be three values, the x,y,z of the position vector

    Returns:

    radec: numpy array

    Two values, the first for right ascension and the second for declination

    '''
    DEC = -(np.arccos(xyz[2]/np.linalg.norm(xyz))-np.pi/2)
    RA = (np.arctan2(xyz[1],xyz[0]))
    while (RA > 2*np.pi):
        RA -= 2*np.pi
    while (RA < 0):
        RA += 2*np.pi
    return np.array([RA*180/np.pi,DEC*180/np.pi])

def build_astroquery_get_target(name,cor,loc):
    """
    This wrapper function automates the creation of objects through the JPL Horizons database.

    Arguments:

    name: str

    Stipulates the target object in Horizons. The major bodies in the Solar System have an id based on their position.
    Hence '5' refers to Jupiter and '3' to Earth. A single number designator refers to a barycenter and a designator
    such as '599' to the planetary center. For minor bodies in the Solar System, the id_type in the Horizons
    must be changed to "minorbody"

    cor: str

    Refers to the type of correction that the object has. Available arguments are "geometric","astrometric" and
    "apparent"

    loc: strj

    Designates the location of observation. Names that start with "g@#" refer to barycenters where the number designates the
    body that the observer is based at. Hence "g@0" refers to the Solar System barycenter. Also takes Earth location designators.
    Observatories are named after their code. Hence, Pan-Starrs observatory is referred as "f51"

    Returns:

    get_target_xyz function
    """
    def astroquery_get_target(t):
        """
        Returns the vectors of the Horizons body at a certain time t.

        Arguments:

        t: days

        Julian date of observation

        Returns:

        xyz: numpy array

        A position vector of the observed object

        uvw: numpy array

        An instantaneous velocity vector of the observed object
        """
        t = Time(t, format='jd', scale='utc')
        hor = Horizons(id=target, location=obs, epochs=t.tt.value, id_type='majorbody')
        vectors = hor.vectors(aberrations='astrometric', refplane = 'earth')
        xyz = np.array([float(vectors['x']),float(vectors['y']),float(vectors['z'])])
        vxvyvz = np.array([float(vectors['vx']),float(vectors['vy']),float(vectors['vz'])])
        radec = CordConv(xyz)
        return xyz, vxvyvz, radec
    return astroquery_get_target
