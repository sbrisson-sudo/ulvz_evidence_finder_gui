# ++ compute azimuth  = angle(station,source,ulvz) ++
def get_angle2(A,B,C):
    """Compute the angle ABC between 3 points on a sphere,
    points coordinates given in (lat,lon), in degrees
    
    A = ULVZ
    B = event
    C = station
    
    """
    
    print(A,B,C)
        
    # compute distances between event - pivot - station
    a = locations2degrees(*B, *C)*pi/180
    b = locations2degrees(*A, *C)*pi/180
    c = locations2degrees(*A, *B)*pi/180
    
    print(list(map(lambda x : x*180./pi, (a,b,c))))
    
    print((cos(b) - cos(a)*cos(c))/(sin(a)*sin(b)))
    
    ABC = arccos((cos(b) - cos(a)*cos(c))/(sin(a)*sin(b))) * 180./pi
    
    print(ABC)

    return ABC

# ++ compute azimuth  = angle(station,source,ulvz) ++
def get_angle(source, ulvz, station):
    """Compute the angle between the ulvz and the station with respect to the source.

    Args:
        source  : (lat,lon) source, in degrees
        ulvz    : (lat,lon) source, in degrees
        station : (lat,lon) source, in degrees
    """
        
    # compute distances between event - pivot - station
    Δsource_station = locations2degrees(*source, *station) # a
    Δsource_ulvz = locations2degrees(*source, *ulvz) # c
    Δulvz_station = locations2degrees(*ulvz, *station) # b
    
    # compute azimuth
    az = arccos((cos(Δulvz_station*pi/180.) - cos(Δsource_ulvz*pi/180.)*cos(Δsource_station*pi/180.)) / (sin(Δsource_ulvz*pi/180.)*sin(Δsource_station*pi/180.)))*180./pi
    
    if np.isnan(az):
        return 0.
            
    return az