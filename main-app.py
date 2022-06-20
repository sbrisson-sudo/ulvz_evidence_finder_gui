import sys
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5.QtCore import *
from PyQt5.QtGui import * 
from PyQt5.QtWidgets import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from matplotlib.transforms import blended_transform_factory

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import numpy as np

from numpy import cos, sin, arccos, pi

from obspy.core import read
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel
from obspy.taup.utils import get_phase_names


taup_model = TauPyModel(model="prem")

import os               
            
from spherical_angle import spherical_angle as get_angle

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        
class MplCanvasGeo(FigureCanvasQTAgg):
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        
        self.proj = ccrs.Robinson()
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111, projection=self.proj)
        super(MplCanvasGeo, self).__init__(fig)

class GraphWindow(QMainWindow):
    
    def __init__(self, mainWindow, *args, **kwargs):
    
        super(GraphWindow, self).__init__(*args, **kwargs)
        
        
        self.mainWindow = mainWindow
        
        self.setWindowTitle("Setup map")
        
        w = 600
        h = 400
        self.resize(w,h)
        
        self.canvas_geo = MplCanvasGeo(self, width=6, height=4, dpi=100)
        toolbar_geo = NavigationToolbar(self.canvas_geo, self)

        layout_main = QHBoxLayout()
        
        layout_plot = QVBoxLayout()
        layout_main.addLayout(layout_plot, 3)
        layout_plot.addWidget(toolbar_geo)
        layout_plot.addWidget(self.canvas_geo)
        
        self.table_stations = TableView(self.mainWindow, len(self.mainWindow.stations["code"]), 3)
        self.table_stations.show()
        layout_main.addWidget(self.table_stations)
        
        self.table_stations.cellClicked.connect(self.plot_station_clicked)

        self.initiate_map()
        try:
            self.plot_map()
        except AttributeError:
            # no data loaded yet
            pass
                
        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QWidget()
        widget.setLayout(layout_main)
        self.setCentralWidget(widget)
        
        cid = self.canvas_geo.mpl_connect('button_press_event', self.__onclick__)

    def initiate_map(self):
            
        ax = self.canvas_geo.axes
        
        ax.coastlines()
        ax.add_feature(cfeature.LAND)
        ax.gridlines(linestyle=":", color="k")
                
    def plot_map(self):
                
        ax = self.canvas_geo.axes
        
        ax.cla()
        
        self.initiate_map()
        
        # plotting source
        ax.scatter(self.mainWindow.lon_event, self.mainWindow.lat_event, marker="*", color="r", s = 100, transform = ccrs.PlateCarree(), label="event")
        
        # plotting stations
        ax.scatter(self.mainWindow.stations["lon"],self.mainWindow.stations["lat"], marker="^", color="g", s = 100, transform = ccrs.PlateCarree(), ec="k", label="stations")
        
        self.plot_ulvz()
        
        ax.set_global()
        ax.legend()
        
        self.canvas_geo.draw()
        
    def plot_ulvz(self):
        
        # if it's not the dafault position
        if self.mainWindow.lon_ulvz != 0. and self.mainWindow.lat_ulvz != 90.:
            self.canvas_geo.axes.scatter(
                self.mainWindow.lon_ulvz,
                self.mainWindow.lat_ulvz, 
                marker="o", color="orange", s = 100, transform = ccrs.PlateCarree(), label="ulvz")
        
    def __onclick__(self, click):

        ax = self.canvas_geo.axes
        xy_data = (click.xdata,click.ydata)

        # convert from data to cartesian coordinates
        lon,lat = ccrs.PlateCarree().transform_point(*xy_data, src_crs=self.canvas_geo.proj)
        
        self.mainWindow.lon_ulvz = lon 
        self.mainWindow.lat_ulvz = lat
        
        self.mainWindow.set_actualisation_needed()
        self.plot_map()
        
    def plot_station_clicked(self, row, column):
        
        self.plot_map()
        
        lat,lon = self.mainWindow.stations["lat"][row],self.mainWindow.stations["lon"][row]
        ax = self.canvas_geo.axes
        ax.scatter(lon,lat, marker="^", color="orange", s = 100, transform = ccrs.PlateCarree(), ec="k", label="stations")
        self.canvas_geo.draw()
        
        
class TableView(QTableWidget):
    
    def __init__(self, mainWindow, *args):
        
        QTableWidget.__init__(self, *args)
        self.mainWindow = mainWindow
        self.set_data_stations()
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
 
    def set_data_stations(self): 
        
        data = self.mainWindow.stations
        
        horHeaders = []
        
        for n,code in enumerate(data["code"]):
        
            horHeaders.append(code)
            
            # latitude longitude
            self.setItem(n, 0,  QTableWidgetItem(f"{data['lat'][n]:.1f}"))
            self.setItem(n, 1,  QTableWidgetItem(f"{data['lon'][n]:.1f}"))
            
            # checkbox
            self.setItem(n, 2,  QTableWidgetItem(f"✅")) # ❌

                
        self.setHorizontalHeaderLabels(["lat", "lon"])
        self.setVerticalHeaderLabels(data["code"])

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        
        super(MainWindow, self).__init__(*args, **kwargs)
        
        self.setWindowTitle("Traces plot assistant")
        
        w = 1100; h = 800
        self.resize(w,h)
        
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        
        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(self.canvas, self)

        layout_1 = QHBoxLayout()
        
        layout_plot = QVBoxLayout()
        layout_1.addLayout(layout_plot, 3)
        layout_plot.addWidget(toolbar)
        layout_plot.addWidget(self.canvas)
        
        layout_config = QVBoxLayout()
        layout_1.addLayout(layout_config, 1)

        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QWidget()
        widget.setLayout(layout_1)
        self.setCentralWidget(widget)
        
        # Data widgets
        
        groupbox_data = QGroupBox("Waveforms data")
        layout_config.addWidget(groupbox_data)
                
        layout_data = QVBoxLayout()
        groupbox_data.setLayout(layout_data)
        
        # Add a button to open file choose dialog
        button_open_file = QPushButton("Open pickle file")
        layout_data.addWidget(button_open_file)
        button_open_file.clicked.connect(self.openPickleFile)
        
        # Add label displaying total number of traces
        self.label_filename = QLabel()
        layout_data.addWidget(self.label_filename)
        
        self.label_data = QLabel()
        layout_data.addWidget(self.label_data)
        self.label_data.setText("No data loaded yet")
        
        
        # Selection over stream
        
        # to see if actualisation needed
        self.need_actualisation = False
        # Actualize stream button
        self.button_stream = QPushButton("Actualize stream")
        
        groupbox_stream = QGroupBox("Selection on stream")
        layout_config.addWidget(groupbox_stream)
                
        layout_stream = QVBoxLayout()
        groupbox_stream.setLayout(layout_stream)
        
        # Time and Distance bounds
        
        layout_bounds = QGridLayout()
        layout_stream.addLayout(layout_bounds)
        
        # # TIME
        # layout_bounds.addWidget(QLabel("tmin : "), 0, 0)
        # layout_bounds.addWidget(QLabel("tmax : "), 1, 0)
        
        # self.tmin_w = QSlider(Qt.Horizontal)
        # self.tmax_w = QSlider(Qt.Horizontal)
        
        # self.tmin_w.setMinimum(0)
        # self.tmin_w.setMaximum(1000)
        # self.tmin_w.setValue(0)
        # self.tmax_w.setMinimum(0)
        # self.tmax_w.setMaximum(1000)
        # self.tmax_w.setValue(1000)
        
        # layout_bounds.addWidget(self.tmin_w, 0, 1)
        # layout_bounds.addWidget(self.tmax_w, 1, 1)
        
        # tmin_label = QLabel()
        # tmax_label = QLabel()   
        
        # tmin_label.setText(str(self.tmin_w.value())+"s")
        # tmax_label.setText(str(self.tmax_w.value())+"s")
        
        # self.tmin_w.valueChanged.connect(lambda x:tmin_label.setText(str(x)+"s"))
        # self.tmax_w.valueChanged.connect(lambda x:tmax_label.setText(str(x)+"s"))
        
        # layout_bounds.addWidget(tmin_label, 0, 2)
        # layout_bounds.addWidget(tmax_label, 1, 2)
        
        # self.tmin_w.valueChanged.connect(self.set_tmin)
        # self.tmax_w.valueChanged.connect(self.set_tmax)
        
        # DISTANCE
        layout_bounds.addWidget(QLabel("dmin : "), 2, 0)
        layout_bounds.addWidget(QLabel("dmax : "), 3, 0)
        
        self.dmin_w = QSlider(Qt.Horizontal)
        self.dmax_w = QSlider(Qt.Horizontal)
        
        self.dist_slider2value = 0.1
        
        self.dmin_w.setMinimum(0)
        self.dmin_w.setMaximum(1800)
        self.dmin_w.setValue(0)
        self.dmax_w.setMinimum(0)
        self.dmax_w.setMaximum(1800)
        self.dmax_w.setValue(1800)
        
        layout_bounds.addWidget(self.dmin_w, 2, 1)
        layout_bounds.addWidget(self.dmax_w, 3, 1)
        
        dmin_label = QLabel()
        dmax_label = QLabel()
        
        dmin_label.setText(str(self.dmin_w.value()*self.dist_slider2value)+"°")
        dmax_label.setText(str(self.dmax_w.value()*self.dist_slider2value)+"°")
        
        layout_bounds.addWidget(dmin_label, 2, 2)
        layout_bounds.addWidget(dmax_label, 3, 2)
        
        self.dmin_w.valueChanged.connect(lambda x:dmin_label.setText(f"{x*self.dist_slider2value:.1f}°"))
        self.dmax_w.valueChanged.connect(lambda x:dmax_label.setText(f"{x*self.dist_slider2value:.1f}°"))
        
        self.dmin_w.valueChanged.connect(self.set_dmin)
        self.dmax_w.valueChanged.connect(self.set_dmax)
        
        # AZIMUTH
        
        # reference point to compute azimuth (default : north pole)
        self.lat_ulvz = 90.0
        self.lon_ulvz = 0.0
        
        layout_bounds.addWidget(QLabel("azmin : "), 4, 0)
        layout_bounds.addWidget(QLabel("azmax : "), 5, 0)
       
        self.azmin_w = QSlider(Qt.Horizontal)
        self.azmax_w = QSlider(Qt.Horizontal)
        
        self.azmin_w.setMinimum(-180)
        self.azmin_w.setMaximum(180)
        self.azmin_w.setValue(-180)
        self.azmax_w.setMinimum(-180)
        self.azmax_w.setMaximum(180)
        self.azmax_w.setValue(180)
        
        layout_bounds.addWidget(self.azmin_w, 4, 1)
        layout_bounds.addWidget(self.azmax_w, 5, 1)
        
        azmin_label = QLabel()
        azmax_label = QLabel()
        
        azmin_label.setText(str(self.azmin_w.value())+"°")
        azmax_label.setText(str(self.azmax_w.value())+"°")

        layout_bounds.addWidget(azmin_label, 4, 2)
        layout_bounds.addWidget(azmax_label, 5, 2)
        
        self.azmin_w.valueChanged.connect(lambda x:azmin_label.setText(str(x)+"°"))
        self.azmax_w.valueChanged.connect(lambda x:azmax_label.setText(str(x)+"°"))
        
        self.azmin_w.valueChanged.connect(self.set_azmin)
        self.azmax_w.valueChanged.connect(self.set_azmax)
        
        # Filtering 
        
        self.filter_w = QCheckBox()
        self.filter_w.setText("Apply frequency filter")
        layout_bounds.addWidget(self.filter_w, 6, 1)
        
        layout_bounds.addWidget(QLabel("fmin : "), 7, 0)
        layout_bounds.addWidget(QLabel("fmax : "), 8, 0)
       
        self.fmin_w = QSlider(Qt.Horizontal)
        self.fmax_w = QSlider(Qt.Horizontal)
        
        self.fmin_w.setDisabled(True)
        self.fmax_w.setDisabled(True)
                
        self.filter_w.toggled.connect(lambda x : self.fmin_w.setDisabled(not(x)))
        self.filter_w.toggled.connect(lambda x : self.fmax_w.setDisabled(not(x)))
        
        self.freq_slider2value = 1e-4
        
        self.fmin_w.setMinimum(0)
        self.fmin_w.setMaximum(100)
        self.fmin_w.setValue(0)
        self.fmax_w.setMinimum(0)
        self.fmax_w.setMaximum(100)
        self.fmax_w.setValue(100)
        
        layout_bounds.addWidget(self.fmin_w, 7, 1)
        layout_bounds.addWidget(self.fmax_w, 8, 1)
        
        fmin_label = QLabel()
        fmax_label = QLabel()
        
        fmin_label.setText(f"{self.fmin_w.value()*self.freq_slider2value*1000:.2f}mHz")
        fmax_label.setText(f"{self.fmax_w.value()*self.freq_slider2value*1000:.2f}mHz")

        layout_bounds.addWidget(fmin_label, 7, 2)
        layout_bounds.addWidget(fmax_label, 8, 2)

        self.fmin_w.valueChanged.connect(lambda x:fmin_label.setText(f"{x*self.freq_slider2value*1000:.2f}mHz"))
        self.fmax_w.valueChanged.connect(lambda x:fmax_label.setText(f"{x*self.freq_slider2value*1000:.2f}mHz"))
        
        self.fmin_w.valueChanged.connect(self.set_fmin)
        self.fmax_w.valueChanged.connect(self.set_fmax)
        
        self.set_fmin(self.fmin_w.value())
        self.set_fmax(self.fmax_w.value())
        
        # Component
        self.component_w = QComboBox()
        self.component_w.addItems(["Vertical (Z)", "Radial (R)", "Transverse (T)"])
        layout_stream.addWidget(self.component_w)
        self.component_w.currentIndexChanged.connect(self.set_component)
        
        # azimuth or distance
        self.az_or_dist_w = QComboBox()
        self.az_or_dist_w.addItems(["Plot along distance", "Plot along azimuth"])
        layout_stream.addWidget(self.az_or_dist_w)
        self.az_or_dist_w.currentIndexChanged.connect(self.set_actualisation_needed)

        
        # Reference phase 
        layout_phase_ref = QHBoxLayout()
        layout_stream.addLayout(layout_phase_ref)
        layout_phase_ref.addWidget(QLabel("Reference phase :"))
        
        self.phase_ref_w = QComboBox()
        self.phase_ref_w.addItems(["None", "S Sdiff", "P Pdiff"])
        layout_phase_ref.addWidget(self.phase_ref_w)
        self.phase_ref_w.currentIndexChanged.connect(self.set_actualisation_needed)

        

        layout_stream.addWidget(self.button_stream)
        self.button_stream.clicked.connect(self.update_stream)
        
        # Add label displaying total number of traces
        self.label_stream = QLabel()
        layout_stream.addWidget(self.label_stream)
        self.label_stream.setText("No data loaded yet")
                        
        # Plotting options
        groupbox_plot_options = QGroupBox("Plotting options")
        layout_config.addWidget(groupbox_plot_options)
                
        layout_plot_options = QVBoxLayout()
        groupbox_plot_options.setLayout(layout_plot_options)
        
        # Scale
        layout_scale = QHBoxLayout()
        layout_plot_options.addLayout(layout_scale)

        self.scale_w = QSlider(Qt.Horizontal)
        self.scale_w.setMinimum(0)
        self.scale_w.setMaximum(1000)
        self.scale_w.setValue(10)
        
        self.smax = 100
        
        scale_label = QLabel()
        scale_label.setText(f"scale : {np.exp(np.log(self.smax+1)/1000*self.scale_w.value())-1.:03.1f}")
        
        self.scale_w.valueChanged.connect(lambda x:scale_label.setText(f"scale : {np.exp(np.log(self.smax+1)/1000*self.scale_w.value())-1.:03.1f}"))
        
        layout_scale.addWidget(scale_label)
        layout_scale.addWidget(self.scale_w)
        
        # Fill
        self.fill_w = QCheckBox()
        self.fill_w.setText("Fill wiggles")
        layout_plot_options.addWidget(self.fill_w)
        
        # Phases 
        layout_phases = QHBoxLayout()
        layout_plot_options.addLayout(layout_phases)
        layout_phases.addWidget(QLabel("Phases :"))
        self.phase_list_w = QLineEdit()
        layout_phases.addWidget(self.phase_list_w)
        
        # Stations codes
        self.name_stations_w = QCheckBox()
        self.name_stations_w.setText("Stations codes")
        layout_plot_options.addWidget(self.name_stations_w)
        
        # azimuth or distance
        self.horizontal_vertical = QComboBox()
        self.horizontal_vertical.addItems([ "Time vertical" , "Time horizontal" ])
        layout_plot_options.addWidget(self.horizontal_vertical)
        
        # button for resetting time window
        button_reset_time_bounds = QPushButton("Reset time window")
        layout_plot_options.addWidget(button_reset_time_bounds)
        button_reset_time_bounds.clicked.connect(self.reset_time_bounds)
        
        # Plotting button
        button_plot = QPushButton("Actualize plot")
        layout_plot_options.addWidget(button_plot)
        button_plot.clicked.connect(self.plot_stream)
        button_plot.clicked.connect(self.plot_map)
        
        # plotting canvas for map
        self.canvas_geo = MplCanvasGeo(self, width=3, height=3, dpi=100)
        
        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        # toolbar_geo = NavigationToolbar(self.canvas_geo, self)
        
        layout_plot_geo = QVBoxLayout() 
        layout_config.addLayout(layout_plot_geo, 3)
        # layout_plot_geo.addWidget(toolbar_geo)
        layout_plot_geo.addWidget(self.canvas_geo)
        
        layout_config.addStretch()
        
        self.initiate_map()
        
        # Open map in new window button
        button_open_map_window = QPushButton("Open in separated window")
        layout_plot_geo.addWidget(button_open_map_window)
        button_open_map_window.clicked.connect(self.open_map_window)
        
        self.map_window = False
        
        # reset the color of the stream actualization button
        self.unset_actualisation_needed()
        
        self.show()
        
    def openPickleFile(self):
        """Open a dialog window to select the pickle file.
        """
        
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        
        if fileName:
            self.load_pickle(fileName)
            self.initialize_entries()
            self.unset_actualisation_needed()
            
    def load_pickle(self, filename):
        
        print("Loading pickle obspy stream file")
        
        self.label_filename.setText(os.path.basename(filename))
        
        self.stream_orig = read(filename)
        
        self.set_metadata(self.stream_orig)
        self.compute_azimuth(self.stream_orig)
        
        self.stream = self.stream_orig.copy()
        
        # set info in label
        dt = self.stream_orig[0].stats.sampling_rate
        self.label_data.setText(f"{self.stream_orig.count()} traces sampled at {dt:.1f}s ({os.path.getsize(filename)/(1024*1024):.1f}Mo)")
        self.label_stream.setText(f"{self.stream_orig.count()} traces")
                        
        self.get_stations_metadata()
        self.plot_map()
                
        
    def set_metadata(self, stream):
        
        trace = stream[0]
        self.lat_event = trace.stats.evla
        self.lon_event = trace.stats.evlo
        self.depth_event = trace.stats.evde
        self.origin_time_event = trace.stats.event_origin_time
        
    def compute_azimuth(self, stream):
                
        for tr in stream:
                        
            tr.stats.azimuth = get_angle(
                [self.lat_ulvz, self.lon_ulvz],
                [self.lat_event,self.lon_event],  
                [tr.stats.coordinates["latitude"],tr.stats.coordinates["longitude"]]
                )
            
            # print("[compute azimuth]", [tr.stats.coordinates["latitude"],tr.stats.coordinates["longitude"]], tr.stats.azimuth)
            
        self.set_azimuth_bounds_slider(stream)
            
    def set_azimuth_bounds_slider(self, stream):
        
        self.azmin = 180.
        self.azmax = -180.
        
        for tr in stream:
            if tr.stats.azimuth < self.azmin: self.azmin = tr.stats.azimuth
            if tr.stats.azimuth > self.azmax: self.azmax = tr.stats.azimuth

        self.azmin -= 1.
        self.azmax += 1.
        
        print("azimuths : ", self.azmin,self.azmax)
        
        self.azmin_w.setValue(  int(self.azmin))
        self.azmin_w.setMinimum(int(self.azmin))
        self.azmin_w.setMaximum(int(self.azmax))
        self.azmax_w.setValue(  int(self.azmax))
        self.azmax_w.setMinimum(int(self.azmin))
        self.azmax_w.setMaximum(int(self.azmax))
        
                        
            
    def switch_az_dist(self):
        """Switch azimuth and distance fields of each traces in a stream object
        Meant to be able to use the stream.plot(type='record') function along azimuth"""
                        
        for tr in self.stream:
            tr.stats.distance, tr.stats.azimuth = tr.stats.azimuth,tr.stats.distance

               
    def update_stream_az_or_dist(self):
        
        idx = self.az_or_dist_w.currentIndex()
        
        if idx == 1:
            self.switch_az_dist()
            
            
    def multiply_distance_by_1000(self):
        
        for tr in self.stream:
            tr.stats.distance *= 1000
                
            
    def plot_stream(self):
        
        try:
            self.stream
        except AttributeError:
            dlg = QDialog(self)
            dlg_layout = QVBoxLayout()
            message = QLabel("Error : no data loaded.")
            dlg_layout.addWidget(message)
            dlg.setLayout(dlg_layout)
            dlg.exec()
            return
        
        ax = self.canvas.axes
        
        # get previous time bounds
        
        self.t_lims = self.get_time_bounds_axis()
        
        ax.cla()  # Clear the canvas.

        self.stream.plot(
            type='section', 
            # dist_degree=True, 
            # ev_coord = (self.lat_event,self.lon_event),
            # norm_method = args.norm,
            scale = self.get_scale(),
            show=False, 
            reftime = self.origin_time_event,
            fig=self.canvas.figure,
            fillcolors = ("r","b") if self.get_fill() else (None,None),
            orientation = self.get_orientation()
        )
        
        if self.get_stations_codes():
            self.plot_stations_codes()
            
        if self.az_or_dist_w.currentIndex() == 1:
            if self.horizontal_vertical.currentIndex() == 1:
                ax.set_ylabel("Azimuth [°]")
            else:
                ax.set_xlabel("Azimuth [°]")
        else:
            if self.horizontal_vertical.currentIndex() == 1:
                ax.set_ylabel("Distance [°]")
            else:
                ax.set_xlabel("Distance [°]")
        
        # set previous time bounds
        self.set_time_bounds_axis()

        
        self.canvas.draw()
        
    def reset_time_bounds(self):
        
        tmin = self.stream[0].stats.starttime - self.origin_time_event
        tmax = self.stream[0].stats.endtime - self.origin_time_event
        
        self.t_lims = [tmin,tmax]
        
        if self.get_orientation() == "vertical":
            self.canvas.axes.set_ylim(self.t_lims)
        else:
            self.canvas.axes.set_xlim(self.t_lims)
                
        
    def get_time_bounds_axis(self):
        
        if self.get_orientation() == "vertical":
            return self.canvas.axes.get_ylim()
        return self.canvas.axes.get_xlim()
    
    def set_time_bounds_axis(self):
        
        print(self.t_lims)
        
        if self.t_lims != (0., 1.):
            if self.get_orientation() == "vertical":
                self.canvas.axes.set_ylim(self.t_lims)
            else:
                self.canvas.axes.set_xlim(self.t_lims)
            
        
        
    def initialize_entries(self):
        
        self.component = "Z"
        
        # # time bounds
        # self.tmin = self.stream[0].stats.starttime - self.origin_time_event
        # self.tmax = self.stream[0].stats.endtime - self.origin_time_event
        
        # self.tmin_w.setValue(int(self.tmin))
        # self.tmin_w.setMinimum(int(self.tmin))
        # self.tmin_w.setMaximum(int(self.tmax))
        # self.tmax_w.setValue(int(self.tmax))
        # self.tmax_w.setMinimum(int(self.tmin))
        # self.tmax_w.setMaximum(int(self.tmax))
        
        # distance bounds
        self.dmin = 180.0
        self.dmax = 0.0
        self.azmin = 180.
        self.azmax = -180.
        
        for tr in self.stream:
            if tr.stats.distance < self.dmin: self.dmin = tr.stats.distance
            if tr.stats.distance > self.dmax: self.dmax = tr.stats.distance
            if tr.stats.azimuth < self.azmin: self.azmin = tr.stats.azimuth
            if tr.stats.azimuth > self.azmax: self.azmax = tr.stats.azimuth
            
        self.dmin -= 1.
        self.dmax += 1.
            
        self.dmin_w.setValue(   int(self.dmin/self.dist_slider2value))
        self.dmin_w.setMinimum( int(self.dmin/self.dist_slider2value))
        self.dmin_w.setMaximum( int(self.dmax/self.dist_slider2value))
        self.dmax_w.setValue(   int(self.dmax/self.dist_slider2value))
        self.dmax_w.setMinimum( int(self.dmin/self.dist_slider2value))
        self.dmax_w.setMaximum( int(self.dmax/self.dist_slider2value))
        
        self.azmin -= 1.
        self.azmax += 1.
        
        self.azmin_w.setValue(  int(self.azmin))
        self.azmin_w.setMinimum(int(self.azmin))
        self.azmin_w.setMaximum(int(self.azmax))
        self.azmax_w.setValue(  int(self.azmax))
        self.azmax_w.setMinimum(int(self.azmin))
        self.azmax_w.setMaximum(int(self.azmax))
            
    def update_stream(self):
        
        try:
            self.stream
        except AttributeError:
            dlg = QDialog(self)
            dlg_layout = QVBoxLayout()
            message = QLabel("Error : no data loaded.")
            dlg_layout.addWidget(message)
            dlg.setLayout(dlg_layout)
            dlg.exec()
            return
        
        self.stream = self.stream_orig.copy()
        
        self.update_stream_component()
        
        self.update_stream_distance()
        
        self.align_on_phase()
        
        self.compute_azimuth(self.stream)
        
        self.update_stream_az_or_dist()
        
        self.multiply_distance_by_1000()
        
        # self.update_stream_time()
        
        self.label_stream.setText(f"{self.stream.count()} traces")
        
        self.get_stations_metadata()
        
        if self.filter_w.isChecked():
            self.filter_data()
            
        self.unset_actualisation_needed()
            
    def set_actualisation_needed(self):
        
        # print("set_actualisation called")
        
        if not(self.need_actualisation):
            self.need_actualisation = True
            self.button_stream.setStyleSheet("background-color : yellow")
            
    def unset_actualisation_needed(self):
        
        # print("unset_actualisation called")
            
        if self.need_actualisation:
            self.need_actualisation = False
            self.button_stream.setStyleSheet("background-color : white")
            
            
    def set_component(self, index):
        self.set_actualisation_needed()
        components = ["Z","R","T"]
        self.component = components[index]
        
                
    def update_stream_component(self):
        self.set_actualisation_needed()
        self.stream = self.stream.select(component = self.component)
        
    def set_dmin(self,value):
        self.set_actualisation_needed()
        try: 
            self.dmin = float(value)*self.dist_slider2value
        except ValueError:
            return
    def set_dmax(self,value):
        self.set_actualisation_needed()
        try: 
            self.dmax = float(value)*self.dist_slider2value
        except ValueError:
            return
        
    def set_azmin(self,value):
        self.set_actualisation_needed()
        try: 
            self.azmin = float(value)
        except ValueError:
            return
    def set_azmax(self,value):
        self.set_actualisation_needed()
        try: 
            self.azmax = float(value)
        except ValueError:
            return
        
        
    def set_fmin(self,value):
        self.set_actualisation_needed()
        try: 
            self.fmin = float(value)*self.freq_slider2value
        except ValueError:
            return
    def set_fmax(self,value):
        self.set_actualisation_needed()
        try: 
            self.fmax = float(value)*self.freq_slider2value
        except ValueError:
            return
        
    def update_stream_distance(self):
        to_remove = []
        for trace in self.stream.traces:
            if not(self.dmin <= trace.stats.distance and self.dmax >= trace.stats.distance):
                to_remove.append(trace)
        for trace in to_remove:
            self.stream.remove(trace)
            
    # def set_tmin(self,value):
    #     self.set_actualisation_needed()
    #     try:
    #         self.tmin = float(value)
    #     except ValueError:
    #         return
    # def set_tmax(self,value):
    #     self.set_actualisation_needed()
    #     try:
    #         self.tmax = float(value)
    #     except ValueError:
    #         return

    # def update_stream_time(self):
    #     self.stream.trim(self.origin_time_event + self.tmin, self.origin_time_event + self.tmax)
        
    def get_scale(self):
        return np.exp(np.log(self.smax+1)/1000*self.scale_w.value())-1.
    
    def get_fill(self):
        return self.fill_w.isChecked()
    
    def get_stations_codes(self):
        return self.name_stations_w.isChecked()
    
    def get_stations_metadata(self):
            
        print("Updating stations metadata")
        
        self.stations = {
            "lon" : [],
            "lat" : [],
            "code" : []
        }
        
        self.set_component(self.component_w.currentIndex())

        for tr in self.stream:

            if tr.stats.component == self.component:
            
                self.stations["lat"].append(tr.stats.coordinates["latitude"])
                self.stations["lon"].append(tr.stats.coordinates["longitude"])
                self.stations["code"].append(tr.stats.station)
                                
    
    def plot_stations_codes(self):
        
        print(">> printing stations codes")
        ax = self.canvas.axes
        for tr in self.stream:
            if self.get_orientation() == "vertical" :
                transform = blended_transform_factory(ax.transData, ax.transAxes)
                ax.text(tr.stats.distance/1000, 1.0, tr.stats.station, transform=transform, zorder=10, va="bottom", ha="center", fontfamily = "monospace", rotation = -45.)
            else :
                transform = blended_transform_factory(ax.transAxes, ax.transData)
                ax.text(1.0,tr.stats.distance/1000, tr.stats.station, transform=transform, zorder=10, va="bottom", ha="center", fontfamily = "monospace", rotation = -45.)


    def align_on_phase(self):
        
        idx = self.phase_ref_w.currentIndex()
        
        if idx == 0 : return
        
        phase_list = self.phase_ref_w.currentText().split()
        
        print(f"Aligning arrivals on {phase_list}")
        
        for tr in self.stream:
            t_arr = taup_model.get_travel_times(
                source_depth_in_km=self.depth_event,
                distance_in_degree=tr.stats.distance,
                phase_list = phase_list,
                )[0].time
            
            tr.stats.starttime -= t_arr
            
    #     self.change_time_bounds_relative()
            
    # def change_time_bounds_relative(self):
        
    #     tmax_rel = self.stream[0].stats.endtime - self.origin_time_event
        
    #     self.tmin = -20
    #     self.tmax = 100
        
    #     self.tmin_w.setValue(-20)
    #     self.tmin_w.setMinimum(-int(tmax_rel))
    #     self.tmin_w.setMaximum(int(tmax_rel))
    #     self.tmax_w.setValue(100)
    #     self.tmax_w.setMinimum(-int(tmax_rel))
    #     self.tmax_w.setMaximum(int(tmax_rel))
        

    def get_orientation(self):
        
        return "horizontal" if self.horizontal_vertical.currentText() == "Time horizontal" else "vertical"
    
    def get_phases(self):
        
        phase_list = self.phase_list_w.text
        
        print(phase_list.split())
        
        for p in phase_list:
            if p not in get_phase_names("ttall"):
                phase_list.remove(p)
                print(f"Warning : invalid phase name : {p}")
                
        return phase_list
    
    def initiate_map(self):
        
        ax = self.canvas_geo.axes
        
        ax.coastlines()
        ax.add_feature(cfeature.LAND)
        ax.gridlines(linestyle=":", color="k")
                
    def plot_map(self):
        
        print("Plotting map...")
        
        ax = self.canvas_geo.axes
        
        ax.cla()
        
        self.initiate_map()
        
        # plotting source
        ax.scatter(self.lon_event, self.lat_event, marker="*", color="r", s = 100, transform = ccrs.PlateCarree())
        
        # plotting stations
        ax.scatter(self.stations["lon"],self.stations["lat"], marker="^", color="g", s = 100, transform = ccrs.PlateCarree(), ec="k")
        
        self.plot_ulvz()
        
        ax.set_global()
        
        self.canvas_geo.draw()
        
        # also updating in separated window if exist
        if self.map_window:
            self.map_window.plot_map()
        
    def filter_data(self):

        if self.fmin >= self.fmax:
            print("Error : fmin >= fmax")
            return 
        
        if self.fmin == 0.:
            self.fmin = 1e-6
            
        print(f"Filtering data, fmin={self.fmin}Hz, fmax={self.fmax}Hz")
            
        self.stream.filter('bandpass', freqmin=self.fmin, freqmax=self.fmax)
        
    def open_map_window(self):
        
        self.map_window = GraphWindow(self)
        self.map_window.show()
        
    def closeEvent(self, event):
        
        if self.map_window:
            self.map_window.close()


    def plot_ulvz(self):
        
        # if it's not the dafault position
        if self.lon_ulvz != 0. and self.lat_ulvz != 90.:
            self.canvas_geo.axes.scatter(
                self.lon_ulvz,
                self.lat_ulvz, 
                marker="o", color="orange", s = 100, transform = ccrs.PlateCarree(), label="ulvz")
        

app = QApplication(sys.argv)
w = MainWindow()
app.exec_()
