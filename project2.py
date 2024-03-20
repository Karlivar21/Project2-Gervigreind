#!/usr/bin/env python

from project2_base import *
from filterpy.kalman import KalmanFilter
import numpy as np
from cartes.crs import Mercator
import matplotlib.pyplot as plt
from copy import deepcopy
from geopy.distance import geodesic

#############################

def show_plot(flight, title, sub_title=None):
    """ Displays the plot of the provided flight data """

    with plt.style.context('traffic'):
        fig = plt.figure()
        ax = plt.axes(projection=Mercator())
        fig.suptitle(title)
        flight.plot(ax, color='green')
        if sub_title:
            ax.set_title(sub_title)
        plt.show()


def kalman_filter(radar_data):
    # Initialize Kalman filter
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # Define state transition matrix (constant velocity model)
    dt = 1.0  # assuming 1 second time step
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    # Define observation matrix
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

    # Define process noise covariance matrix (Q)
    q_var = 0.01
    kf.Q = np.array([[q_var, 0, 0, 0],
                     [0, q_var, 0, 0],
                     [0, 0, q_var, 0],
                     [0, 0, 0, q_var]])

    # Define measurement noise covariance matrix (R)
    r_var = 0.1
    kf.R = np.array([[r_var, 0],
                     [0, r_var]])

    # Set initial state
    initial_state = np.array([radar_data.data.x[0],
                               radar_data.data.y[0],
                               0, 0])  # Assuming initial velocity is zero
    kf.x = initial_state

    # Set initial covariance matrix
    kf.P = np.eye(4) * 1000  # Assuming high uncertainty initially

    # Iterate through radar measurements to update the filter and obtain filtered position estimates
    filtered_positions = []
    for i in range(len(radar_data.data)):
        measurement = np.array([[radar_data.data.x[i]],
                                [radar_data.data.y[i]]])
        kf.predict()
        kf.update(measurement)
        filtered_positions.append((kf.x[0], kf.x[1]))  # Append filtered position estimates

    return filtered_positions


def compute_distance_error(original_positions, filtered_positions, ellipsoid_model='WGS-84'):
    errors = []

    for orig_pos, filt_pos in zip(original_positions, filtered_positions):
        orig_lat, orig_lon = orig_pos
        filt_lat_lon = filt_pos  # Updated to handle filtered positions directly as (lat, lon) tuples
        orig_point = (orig_lat, orig_lon)
        dist = geodesic(orig_point, filt_lat_lon, ellipsoid=ellipsoid_model).kilometers
        errors.append(dist)

    mean_error = np.mean(errors)
    max_error = np.max(errors)
    return mean_error, max_error



def main():
    flights = get_ground_truth_data()
    flight_name = 'DMUPY_052' # change this to get another flight
    flight = flights[flight_name]

    unfiltered_radar_data = get_radar_data_for_flight(flight)
    # show_plot(flight, flight_name)
    # show_plot(unfiltered_radar_data, flight_name)

    filtered_positions = kalman_filter(unfiltered_radar_data)
    filtered_radar_data = deepcopy(unfiltered_radar_data)
    filtered_radar_data.data.x = [pos[0] for pos in filtered_positions]
    filtered_radar_data.data.y = [pos[1] for pos in filtered_positions]
    filtered_radar_data = set_lat_lon_from_x_y(filtered_radar_data)
    # show_plot(filtered_radar_data, flight_name, sub_title='Kalman Filtered Radar Data')

    # Convert filtered Cartesian coordinates to latitude/longitude
    filtered_lat_lon = list(zip(filtered_radar_data.data.latitude, filtered_radar_data.data.longitude))
    original_lat_lon = list(zip(flight.data.latitude, flight.data.longitude))

    # Compute distance errors
    mean_error, max_error = compute_distance_error(original_lat_lon, filtered_lat_lon)
    print(f"Mean Error: {mean_error} km")
    print(f"Max Error: {max_error} km")

    # Plot original and filtered tracks
    fig, ax = plt.subplots()
    flight.plot(ax, color='green', label='Original Track')
    filtered_radar_data.plot(ax, color='blue', label='Filtered Track')
    ax.legend()
    ax.set_title(f"{flight_name} - Original vs Filtered Track")
    plt.show()

#############################

if __name__ == "__main__":
    main()
