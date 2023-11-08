import math
from astral import sun

import pandas as pd
import numpy as np
from astropy.coordinates import Angle
from astropy import units as u
from astropy.io.votable import parse
from astropy.io import ascii
from astropy.table import Table
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from astral import LocationInfo
import jupyter
from datetime import datetime, timedelta, timezone
from astropy.time import Time
import pytz


# globals
horizonBox = {  # {tuple(decWindow):tuple(minAlt,maxAlt)}
    (-35, -34): (-35, 42.6104),
    (-34, -32): (-35, 45.9539),
    (-32, -30): (-35, 48.9586),
    (-30, -28): (-35, 51.6945),
    (-28, -26): (-35, 54.2121),
    (-26, -24): (-35, 56.5487),
    (-24, -22): (-35, 58.7332),
    (-22, 0): (-35, 60),
    (0, 46): (-52.5, 60),
    (46, 56): (-37.5, 60),
    (56, 65): (-30, 60)
}

# dec_vertices = [item for key in horizonBox.keys() for item in key]  # this is just a list of integers, each being one member of one
dec_vertices = list(set([item for key in horizonBox.keys() for item in key]))  # this is just a list of integers, each being one member of one
# of the dec tuples that are the keys to the horizonBox dictionary
dec_vertices.sort()
locationInfo = LocationInfo(name="TMO", region="CA, USA",
                                        timezone="UTC",
                                        latitude=34.36,
                                        longitude=-117.63)

sidereal_rate = 360 / (23  * 3600 + 56 * 60 + 4.091)  # deg/second


def get_curent_sidereal_time():
    return Time(datetime.utcnow()).sidereal_time('mean', longitude=locationInfo.longitude)

# tmo observability functions (from maestro)
def siderealToDate(siderealAngle: Angle, current_sidereal_time: Angle):
    """!
    Convert an angle representing a sidereal time to UTC by relating it to local sidereal time
    @param siderealAngle: astropy Angle
    @param current_sidereal_time: the current sidereal time, also an astropy angle
    @return: datetime object, utc
    """
    # ---convert from sidereal to UTC---
    # find the difference between the sidereal observability start time and the sidereal start time of the program
    siderealFromStart = siderealAngle - current_sidereal_time
    # add that offset to the utc start time of the program (we know siderealStart is local sidereal time at startTime, so we use it as our reference)
    timeUTC = datetime.utcnow() + timedelta(
        hours=siderealFromStart.hour / 1.0027)  # one solar hour is 1.0027 sidereal hours

    return timeUTC

def dateToSidereal(dt: datetime, current_sidereal_time):
    timeDiff = dt - datetime.utcnow()
    sidereal_factor = 1.0027
    return current_sidereal_time + Angle(str(timeDiff.total_seconds()*sidereal_factor / 3600) + "h")

def toDecimal(angle: Angle):
    """!
    Return the decimal degree representation of an astropy Angle, as a float
    @return: Decimal degree representation, float
    """
    return round(float(angle.degree), 6)  # ew

def ensureFloat(angle):
    """!
    Return angle as a float, converting if necessary
    @rtype angle: float, Angle
    @return: decimal angle, as a float
    """
    try:
        if isinstance(angle, str) or isinstance(angle, tuple):
            angle = Angle(angle)
            return ensureFloat(angle)  # lol
    except:
        pass
    if isinstance(angle, float):
        return angle
    if isinstance(angle, Angle):
        return toDecimal(angle)
    else:
        return float(angle)

def getHourAngleLimits(dec):
    """
    Get the hour angle limits of the target's observability window based on its dec.
    @param dec: float, int, or astropy Angle
    @return: A tuple of Angle objects representing the upper and lower hour angle limits
    """
    dec = ensureFloat(dec)
    for decRange in horizonBox:
        if decRange[0] < dec <= decRange[1]:  # man this is miserable
            finalDecRange = horizonBox[decRange]
            return tuple([Angle(finalDecRange[0], unit=u.deg), Angle(finalDecRange[1], unit=u.deg)])
    return None


def ensureAngle(angle):
    """!
    Return angle as an astropy Angle, converting if necessary
    @param angle: float, int, hms Sexagesimal string, hms tuple, or astropy Angle
    @return: angle, as an astropy Angle
    """
    if not isinstance(angle, Angle):
        try:
            if isinstance(angle, str) or isinstance(angle, tuple):
                angle = Angle(angle)
            else:
                angle = Angle(angle, unit=u.deg)
        except Exception as err:
            print("Error converting", angle, "to angle")
            raise err
    return angle

def angleToTimedelta(angle: Angle):  # low precision
    """!
    Convert an astropy Angle to an timedelta whose duration matches the hourangle of the angle
    @rtype: timedelta
    """
    angleTime = angle.to(u.hourangle)
    angleHours, angleMinutes, angleSeconds = angleTime.hms
    return timedelta(hours=angleHours, minutes=angleMinutes, seconds=0)

def findTransitTime(rightAscension: Angle, location, target_dt=None, current_sidereal_time=None):
    """!Calculate the transit time of an object at the given location.

    @param rightAscension: The right ascension of the object as an astropy Angle
    @type rightAscension: Angle
    @param location: The observatory location.
    @type location: astral.LocationInfo
    @param target_dt: find the next transit after this time. if None, uses currentTime
    @param current_sidereal_time: the current sidereal time, as an astropy angle. will be calculated (slow) if not provided
    @return: The transit time of the object as a datetime object.
    @rtype: datetime.datetime
    """
    currentTime = datetime.utcnow().replace(second=0, microsecond=0)
    if current_sidereal_time is None:
        lst = Time(currentTime).sidereal_time('mean', longitude=location.longitude)
    else:
        lst = current_sidereal_time
    target_time = target_dt.replace(second=0, microsecond=0) or currentTime
    target_sidereal_time = dateToSidereal(target_time,lst)
    ha = Angle(wrap_around((rightAscension - target_sidereal_time).deg),unit=u.deg)
    transitTime = target_time + angleToTimedelta(ha)
    return transitTime

def staticObservabilityWindow(RA: Angle, Dec: Angle, locationInfo: LocationInfo, target_dt=None, current_sidereal_time=None):
    """!
    Generate the TMO observability window for a static target based on RA, dec, and location
    @param RA: right ascension
    @param Dec: declination
    @param locationInfo: astral LocationInfo object for the observatory site
    @param target_dt: find the next transit after this time. if None, uses currentTime
    @param current_sidereal_time: the current sidereal time. will add sidereal days to this if necessary
    @return: [startTime, endTime]
    @rtype: list(datetime)
    """
    if current_sidereal_time is None:
        currentTime = datetime.utcnow().replace(second=0, microsecond=0)
        lst = Time(datetime.utcnow()).sidereal_time('mean', longitude=locationInfo.longitude)
    target_dt = target_dt or datetime.utcnow()
    t = findTransitTime(ensureAngle(float(RA)), locationInfo, current_sidereal_time=current_sidereal_time,target_dt=target_dt)
    timeWindow = (angleToTimedelta(a) for a in getHourAngleLimits(Dec))
    return [t + a for a in timeWindow]
    # HA = ST - RA -> ST = HA + RA


def wrap_around(value):
    a = -180
    b = 180
    return (value - a) % (b-a) + a

# get hour angle, in
def get_hour_angle(ra,dt,current_sidereal_time):
    sidereal = dateToSidereal(dt,current_sidereal_time)
    return Angle(wrap_around((sidereal-ra).deg),unit=u.deg)

def getSunriseSunset(dt=datetime.utcnow(), jd=False):
    """!
    get sunrise and sunset for TMO
    @return: sunriseUTC, sunsetUTC
    @rtype: datetime.datetime
    """
    dt = pytz.UTC.localize(dt)
    s = sun.sun(locationInfo.observer, date=dt, tzinfo=timezone.utc)
    sunriseUTC = s["sunrise"]
    sunsetUTC = sun.time_at_elevation(locationInfo.observer, -10, direction=sun.SunDirection.SETTING,date=dt)

    # TODO: make this less questionable - it probably doesn't do exactly what i want it to when run at certain times of the day:
    if sunriseUTC < dt:  # if the sunrise we found is earlier than the current time, add one day to it (approximation ofc)
        sunriseUTC = sunriseUTC + timedelta(days=1)

    if sunsetUTC > sunriseUTC:
        sunsetUTC = sunsetUTC - timedelta(days=1)

    if jd:
        sunriseUTC, sunsetUTC = datetimeToJulian(sunriseUTC), datetimeToJulian(sunsetUTC)
    return sunriseUTC, sunsetUTC


def get_RA_window(current_sidereal_time, target_dt, dec, ra=None):
    # get the bounding RA coordinates of the TMO observability window for time target_dt for targets at declination dec. Optionally, input an RA to also get out that RA, adjusted for box-shifting

    adjusted_ra = ra.copy() if ra is not None else None
    hourAngleWindow = getHourAngleLimits(dec)
    if not hourAngleWindow: return False
    raWindow = [dateToSidereal(target_dt, current_sidereal_time) - hourAngleWindow[1],
                (dateToSidereal(target_dt, current_sidereal_time) - hourAngleWindow[0]) % Angle(360, unit=u.deg)]

    # we want something like (23h to 17h) to look like [(23h to 24h) or (0h to 17h)] so we move the whole window to start at 0 instead
    if raWindow[0] > raWindow[1]:
        diff = Angle(24, unit=u.hour) - raWindow[0]
        raWindow[1] += diff
        if adjusted_ra is not None:
            adjusted_ra = (adjusted_ra + diff) % Angle(360, unit=u.deg)
        raWindow[0] = Angle(0, unit=u.deg)
    return raWindow, adjusted_ra


def observationViable(dt: datetime, ra: Angle, dec: Angle, current_sidereal_time=None,locationInfo=None):
    """
    Can a target with RA ra and Dec dec be observed at time dt? Checks hour angle limits based on TMO bounding box.
    @return: bool
    """
    if current_sidereal_time is None:
        current_sidereal_time = Time(datetime.utcnow()).sidereal_time('mean', longitude=locationInfo.longitude)
    HA_window = getHourAngleLimits(dec)
    HA = get_hour_angle(ra,dt,current_sidereal_time)
    # NOTE THE ORDER:
    return HA.is_within_bounds(HA_window[0], HA_window[1])


# def observationViable(dt: datetime, ra: Angle, dec: Angle, current_sidereal_time=None,locationInfo=None):
#     """
#     Can a target with RA ra and Dec dec be observed at time dt? Checks hour angle limits based on TMO bounding box.
#     @return: bool
#     """
#     if current_sidereal_time is None:
#         current_sidereal_time = Time(datetime.utcnow()).sidereal_time('mean', longitude=locationInfo.longitude)
#     raWindow, rac = get_RA_window(current_sidereal_time,dt,dec,ra=ra)
#     # NOTE THE ORDER:
#     return rac.is_within_bounds(raWindow[0], raWindow[1])