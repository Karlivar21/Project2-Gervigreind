import math

from traffic.core.geodesy import bearing, distance, destination
from traffic.data import samples
from numpy.random import default_rng
import pyproj

# needed for set_lat_lon_from_x_y below
# is set by get_ground_truth_data()
projection_for_flight = {}


def get_ground_truth_data():
    """ returns a dict of flight_id -> flight with flight data containing the original GPS data
     resampled to 10s (with one position every 10s)
    """
    names = ['liguria', 'pixair_toulouse', 'indiana', 'texas', 'georeal_fyn_island', 'ign_mercantour',
             'ign_fontainebleau', 'mecsek_mountains', 'ign_lot_et_garonne', 'inflight_refuelling', 'aircraft_carrier',
             'luberon', 'alto_adige', 'franconia', 'danube_valley', 'cevennes', 'oxford_cambridge', 'alpi_italiane',
             'rega_zh', 'samu31', 'rega_sg', 'monastir', 'guatemala', 'london_heathrow', 'cardiff', 'sydney',
             'brussels_ils', 'ajaccio', 'toulouse', 'noumea', 'london_gatwick', 'perth', 'kota_kinabalu', 'montreal',
             'funchal', 'nice', 'munich', 'vancouver', 'lisbon', 'liege_sprimont', 'kiruna', 'bornholm', 'kingston',
             'brussels_vor', 'vienna', 'border_control', 'dreamliner_boeing', 'texas_longhorn', 'zero_gravity',
             'qantas747', 'turkish_flag', 'airbus_tree', 'easter_rabbit', 'belevingsvlucht', 'anzac_day', 'thankyou',
             'vasaloppet']

    flights = {}
    i = 0
    for x in names:
        flight = samples.__getattr__(x)
        if not flight:
            print("name %s does not work" % x)
        flight = flight.assign(flight_id=f"{flight.callsign}_{i:03}")
        print("reading data of flight: %s" % flight.flight_id)
        projection = pyproj.Proj(proj="lcc", ellps="WGS84",
                                 lat_1=flight.data.latitude.min(), lat_2=flight.data.latitude.max(),
                                 lon_1=flight.data.longitude.min(), lon_2=flight.data.longitude.max(),
                                 lat_0=flight.data.latitude.mean(), lon_0=flight.data.longitude.mean())
        if flight.flight_id in projection_for_flight:
            print("ERROR: duplicate flight ids: %s" % flight.flight_id)
        flights[flight.flight_id] = flight
        projection_for_flight[flight.flight_id] = projection
        print("reading data of flight: %s" % flight.flight_id)
        i += 1
    return flights


# a dict mapping flight_id -> (lat, lon), where lat, lon is the position of the (fake) radar that was used to create
#  the radar data for the respective flight
radar_position = {}


def get_radar_data_for_flight(flight):
    """
    get_radar_data_for_flight(flight) returns a copy of the given flight, but with the position data (lat, lon,
    altitude, x, y) modified as if it was a reading from a radar, i.e., the data is less accurate and with fewer points
    than the one from get_ground_truth_data().
    The resulting flight will have x, y coordinates set to suitable 2d projections of the lat/lon positions.
    You can access these coordinates in the flight.data attribute, which is a Pandas DataFrame.
    In the resulting flight there is one position every 10 seconds.
    """

    #  1. set parameters: angular error (probably around 1 degree), distance error (probably between 50 - 300m)
    #     see: https://aviation.stackexchange.com/questions/115/what-is-the-range-and-accuracy-of-atc-radar-systems
    #  2. put a radar in position for the flight (e.g., in the middle)
    #  3. for each point in the flight:
    #     - compute distance and bearing to the radar
    #     - add a random angle to the bearing and a random distance to the distance
    #     - compute new position based on new azimuth and distance
    #     - add a random difference to the altitude

    azimuth_error = 1  # in degrees
    distance_error = 100  # in meters
    altitude_angle_error = 1 * math.pi / 180  # 1 degree in radians
    rng = default_rng()
    # place the radar in a random point somewhere in area of the flight
    radar_lat = rng.uniform(flight.data.latitude.min(), flight.data.latitude.max())
    radar_lon = rng.uniform(flight.data.longitude.min(), flight.data.longitude.max())
    # TODO: store radar position for the flight and put it into the plot for reference
    print("flight: %s" % flight.flight_id)
    flight_radar = flight.resample("10s")
    radar_position[flight.flight_id] = (radar_lat, radar_lon)
    # flight_radar.__setattr__("radar_position", (radar_lat, radar_lon))
    for i in range(len(flight_radar.data)):
        lat = flight_radar.data.at[i, "latitude"]
        lon = flight_radar.data.at[i, "longitude"]
        bearing_to_point = bearing(radar_lat, radar_lon, lat, lon)
        dist_to_point = distance(radar_lat, radar_lon, lat, lon)
        bearing_to_point += rng.normal() * azimuth_error
        dist_to_point += rng.normal() * distance_error
        (flight_radar.data.at[i, "latitude"], flight_radar.data.at[i, "longitude"], _) = \
            destination(lat=radar_lat, lon=radar_lon, bearing=bearing_to_point, distance=dist_to_point)
        height_angle = math.atan(flight_radar.data.at[i, "altitude"] / dist_to_point)  # in radians
        height_angle += rng.normal() * altitude_angle_error
        flight_radar.data.at[i, "altitude"] = math.tan(height_angle) * dist_to_point
    projection = projection_for_flight[flight.flight_id]
    flight_radar = flight_radar.compute_xy(projection)
    return flight_radar


def get_radar_data(ground_truth_flights):
    """ get_radar_data(ground_truth_flights) returns dict mapping of flight_id -> flight with flight data containing the
    (fake) radar data computed for each of the flights
    """
    radar_data = {flight_id: get_radar_data_for_flight(flight) for flight_id, flight in ground_truth_flights.items()}
    return radar_data


def set_lat_lon_from_x_y(flight):
    """
    set_lat_lon_from_x_y(flight) updates the given flight's latitudes and longitudes to reflect its x, y positions
    in the data.
    The intended use of this function is to:
      1. make a (deep) copy of a flight that you got from get_radar_data
      2. use a kalman filter on that flight and set the x, y columns of the data to the filtered positions
      3. call set_lat_lon_from_x_y() on that flight to set its latitude and longitude columns according to the
        filtered x,y positions
    Step 3 is necessary, if you want to plot the data, because plotting is based on the lat/lon coordinates and not
    based on x,y.

    :param flight:
    :return: the same flight
    """
    projection = projection_for_flight[flight.flight_id]
    if projection is None:
        print("No projection found for flight %s. You probably did not get this flight from get_radar_data()." % (
            flight.flight_id))

    lons, lats = projection(flight.data["x"], flight.data["y"], inverse=True)
    flight.data["longitude"] = lons
    flight.data["latitude"] = lats
    return flight

