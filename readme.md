***INPUT DATA*** : obspy serialized stream object (`pickle` format), the `stats` field of the traces must contains metadata on the associated event (cf my data acquisition routine at [???])


1. Go to your working directory and execute the `main-app.py` script
2. Inside the GUI :
    1. Load the data with the `open pickle file` button
    
    The GUI options are divided in 3 parts:
    - The `Selection on stream` box : containing option to modify the data you want to look at (time, distance and azimuth bounds, component, frequency content, reference phase)
    - The `Plotting options` box : containing options to modify how is the data plotted (straight forward)
    - Miscellanous button allowing to :
        - open in another window the great circles paths between the stations and the source on top of the SEMUCB-WM1 model at the CMB
        - open a simplem map with the stations an source position along with a table containing the stations information
        - save the current stations which are used in a `receivers.dat` file

    After each `Selection on stream` option change, click on `Actualize stream` to update the data.
    After each `Plotting options` option change, click on `Actualize plot` to update the plot. Don't forget to click `Reset time window` if triming the traces on time of changing the orientation of the plot.

This software is far from perfect, if it start to mess up just restart it.