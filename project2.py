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

def main():
    flights = get_ground_truth_data()
    flight_name = 'DMUPY_052' # change this to get another flight
    flight = flights[flight_name]

    unfiltered_radar_data = get_radar_data_for_flight(flight)
    show_plot(flight, flight_name)

#############################

if __name__ == "__main__":
    main()
