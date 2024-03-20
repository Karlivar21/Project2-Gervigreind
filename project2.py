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



def main():
    flights = get_ground_truth_data()
    flight_name = 'DMUPY_052' # change this to get another flight
    flight = flights[flight_name]

    unfiltered_radar_data = get_radar_data_for_flight(flight)
    # show_plot(flight, flight_name)
    show_plot(unfiltered_radar_data, flight_name)

    filtered_positions = kalman_filter(unfiltered_radar_data)
    filtered_radar_data = deepcopy(unfiltered_radar_data)
    filtered_radar_data.data.x = [pos[0] for pos in filtered_positions]
    filtered_radar_data.data.y = [pos[1] for pos in filtered_positions]
    show_plot(filtered_radar_data, flight_name, sub_title='Kalman Filtered Radar Data')

#############################

if __name__ == "__main__":
    main()
