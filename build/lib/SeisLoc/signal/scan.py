################################################################################################



# ---- Import Packages -----

import numpy as np
from SeisLoc.core.time import UTCDateTime
import SeisLoc.core.model as cmod
from datetime import datetime
from datetime import timedelta

from obspy import read,Stream,Trace
from obspy.core import UTCDateTime

from obspy.signal.trigger import classic_sta_lta
from scipy.signal import butter, lfilter
from scipy.optimize import curve_fit

import SeisLoc.core.SeisLoclib as ilib

import obspy
import re

import os
import os.path as path
import pickle

import pandas as pd

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import matplotlib.dates as mdates
import matplotlib.image as mpimg
import matplotlib.animation as animation



# ----- Timing functions -----

import time

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)



# ----- Useful Functions -----


def gaussian_func(x,a,b,c):
    '''


    '''
    f = a*np.exp(-1.*((x-b)**2)/(2*(c**2)))
    return f

def onset(sig, stw, ltw):
    '''


    '''
    # assert isinstance(snr, object)
    nchan, nsamp = sig.shape
    snr = np.copy(sig)
    snr_raw = np.copy(sig)
    for ch in range(0, nchan):
        if np.sum(sig[ch,:]) == 0.0:
            snr[ch, :] = 0.0
            snr_raw[ch, :] = snr[ch,:]
        else:
            snr[ch, :] = classic_sta_lta(sig[ch, :]+1.0, stw, ltw)
            snr_raw[ch, :] = snr[ch,:]
            np.clip(1+snr[ch,:],0.8,np.inf,snr[ch, :])
            np.log(snr[ch, :], snr[ch, :])

    return snr_raw,snr


def filter(sig,srate,lc,hc,order=3):
    '''


    '''
    b1, a1 = butter(order, [2.0*lc/srate, 2.0*hc/srate], btype='band')
    nchan, nsamp = sig.shape
    fsig = np.copy(sig)
    #sig = detrend(sig)
    for ch in range(0, nchan):
        fsig[ch,:] = fsig[ch,:] - fsig[ch,0]
        fsig[ch,:] = lfilter(b1, a1, fsig[ch,::-1])[::-1]
        fsig[ch,:] = lfilter(b1, a1, fsig[ch,:])
    return fsig



def _find(obj, name, default=None):
    if isinstance(name, str):
        if name in obj:
            return obj[name]
        else:
            return default
    elif name[0] in obj:
        if len(name) == 1:
            return obj[name[0]]
        else:
            return _find(obj[name[0]], name[1:], default)
    else:
        return default


def _read_scan(fname):
    CoaVal = pd.read_csv(fname,names=['DT','COA','X','Y','Z'])
    CoaVal['DT'] = pd.to_datetime(CoaVal['DT'])
    return CoaVal





class SeisOutFile:
    '''
        Definition of manipulation types for the Seismic scan files.

    '''

    def __init__(self, path = '', name = None):
        self.open(path, name)
        self.FileSampleRate = None #Sample rate in miliseconds

    def open(self, path = '', name = None):
        self.path = path
        if name is None:
            name = datetime.now().strftime('RUN_%Y%m%d_%H%M%S')
        self.name = name
        print('Path = ' + repr(self.path) + ', Name = ' + repr(self.name))


    def read_scan(self):
        fname = path.join(self.path,self.name + '.scn')
        print(fname)
        DATA = _read_scan(fname)
        return DATA

    def read_coal4D(fname):
        map = np.load(fname)
        return map


    def read_decscan(self):
        fname = path.join(self.path,self.name + '.scnmseed')
        COA = obspy.read(fname)

        sampling_rate = COA[0].stats.sampling_rate

        DATA = pd.DataFrame()

        DATA['COA'] = COA[0].data
        timeline=np.arange(COA[0].stats.starttime,COA[0].stats.endtime,timedelta(seconds=1/COA[0].stats.sampling_rate))



    def write_log(self, message):
        fname = path.join(self.path,self.name + '.log')
        with open(fname, "a") as fp:
            fp.write(message + '\n')




    def cut_mseed(self,DATA,EventName):
        fname = path.join(self.path,self.name + '_{}.mseed'.format(EventName))
        
        st = DATA.st
        st.write(fname, format='MSEED')



    def del_scan(self):
        fname = path.join(self.path,self.name + '.scn')
        if path.exists(fname):
           print('Filename {} already exists. Deleting !'.format(fname))
           os.system('rm {}'.format(fname))


    def write_scan(self,daten,dsnr,dloc):
        # Defining the ouput filename
        fname = path.join(self.path,self.name + '.scn')

        # Defining the array to save
        ARRAY = np.array((daten,dsnr,dloc[:,0],dloc[:,1],dloc[:,2]))
        # # if 
        if self.FileSampleRate == None:
            DF        = pd.DataFrame(columns=['DT','COA','X','Y','Z'])
            DF['DT']  = daten
            DF['DT']  = pd.to_datetime(DF['DT'])
            DF['DT']  = DF['DT'].astype(str)
            DF['COA'] = dsnr
            DF['X']   = dloc[:,0]
            DF['Y']   = dloc[:,1]
            DF['Z']   = dloc[:,2]

        else:
            # Resampling the data on save
            DF = pd.DataFrame(columns=['DT','COA','X','Y','Z'])
            DF['DT']  = daten
            DF['COA'] = dsnr
            DF['X']   = dloc[:,0]
            DF['Y']   = dloc[:,1]
            DF['Z']   = dloc[:,2]
            DF['DT'] = pd.to_datetime(DF['DT'])
            #DF = DF.set_index(pd.DatetimeIndex(DF['DT']))
            DF = DF.set_index(DF['DT'])
            DF = DF.resample('{}L'.format(self.FileSampleRate)).mean()
            DF = DF.reset_index()
            DF = DF.rename(columns={"index":"DT"})
            DF['DT'] = DF['DT'].astype(str)

        if path.exists(fname):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not

        ARRAY = np.array(DF)

        with open(fname, append_write) as fp:
            for ii in range(ARRAY.shape[0]):
                fp.write('{},{},{},{},{}\n'.format(ARRAY[ii,0],ARRAY[ii,1],ARRAY[ii,2],ARRAY[ii,3],ARRAY[ii,4]))


    def write_decscan(self,sampling_rate):
        scn_fname = path.join(self.path,self.name +'.scn')
        mseed_fname = path.join(self.path,self.name +'.scnmseed')

        DATA = self.read_scan()

        stats_COA = {'network': 'NW', 'station':'COA','npts':len(DATA),'sampling_rate':sampling_rate}
        stats_X = {'network': 'NW', 'station':'COA_X','npts':len(DATA),'sampling_rate':sampling_rate}
        stats_Y = {'network': 'NW', 'station':'COA_Y','npts':len(DATA),'sampling_rate':sampling_rate}
        stats_Z = {'network': 'NW', 'station':'Coa_Z','npts':len(DATA),'sampling_rate':sampling_rate}

        stats_COA['starttime'] = DATA.iloc[0][0]
        stats_X['starttime'] = DATA.iloc[0][0]
        stats_Y['starttime'] = DATA.iloc[0][0]
        stats_Z['starttime'] = DATA.iloc[0][0]

        ST = Stream(Trace(data=np.array(DATA['COA']),header= stats_COA))
        ST = ST + Stream(Trace(data=np.array(DATA['X']),header= stats_X))
        ST = ST + Stream(Trace(data=np.array(DATA['Y']),header= stats_Y))
        ST = ST + Stream(Trace(data=np.array(DATA['Z']),header= stats_Z))

        ST.write(mseed_fname,format='MSEED')

        #os.system('rm {}'.format(scn_fname))



    def write_coal4D(self,map4D,EVENT,stT,enT):
        cstart = datetime.strftime(stT,'%Y-%m-%dT%H:%M:%S.%f')
        cend   = datetime.strftime(enT,'%Y-%m-%dT%H:%M:%S.%f')
        fname = path.join(self.path,self.name + '{}_{}_{}.coal4D'.format(EVENT,cstart,cend))
        
        # This file size is massive ! write as binary and include load function

        # Define the X0,Y0,Z0,T0,Xsiz,Ysiz,Zsiz,Tsiz,Xnum,Ynum,Znum,Tnum
        np.save(fname,map4D)


    def write_coalVideo(self,MAP,lookup_table,DATA,EventCoaVal,EventName,AdditionalOptions=None):
        '''
            Writing the coalescence video to file for each event
        '''

        filename = path.join(self.path,self.name)

        SeisPLT = SeisPlot(MAP,lookup_table,DATA,EventCoaVal)
    
        SeisPLT.CoalescenceVideo(SaveFilename='{}_{}'.format(filename,EventName))
        SeisPLT.CoalescenceMarginal(SaveFilename='{}_{}'.format(filename,EventName))

    def write_stationsfile(self,STATION_pickS,EventName):
        '''
            Writing the stations file
        '''
        fname = path.join(self.path,self.name + '_{}.stn'.format(EventName))
        STATION_pickS.to_csv(fname,index=False)

    def write_event(self,EVENT,EventName):
        '''
            Writing the stations file
        '''
        fname = path.join(self.path,self.name + '_{}.event'.format(EventName))
        EVENT.to_csv(fname,index=False)


class SeisPlot:
    '''
         Seismic plotting for SeisLoc ouptuts. Functions include:
            CoalescenceVideo - A script used to generate a coalescence 
            video over the period of earthquake location

            CoalescenceLocation - Location plot 

            CoalescenceMarginalizeLocation - 

    '''
    def __init__(self,lut,MAP,CoaMAP,DATA,EVENT,StationPick,PlotOptions=None):
        '''
            This is the initial variatiables
        '''
        self.LUT         = lut
        self.DATA        = DATA
        self.EVENT       = EVENT
        self.MAP         = MAP
        self.CoaMAP      = CoaMAP
        self.StationPick = StationPick
        self.RangeOrder  = False


        if PlotOptions == None:
            self.TraceScaling     = 1
            self.CMAP             = 'magma'
            self.LineStationColor = 'white'
            self.Plot_Stations    = True
            self.FilteredSignal   = True
            self.XYFiles          = None
        else:
            try:
                self.TraceScaling     = PlotOptions.TraceScaling
                self.CMAP             = PlotOptions.MAPColor
                self.LineStationColor = PlotOptions.LineStationColor
                self.Plot_Stations    = PlotOptions.Plot_Stations
                self.FilteredSignal   = PlotOptions.FilteredSignal
                self.XYFiles          = PlotOptions.XYFiles

            except:
                print('Error - Please define all plot option, see function ... for details.')



        self.times = np.arange(self.DATA.startTime,self.DATA.endTime,timedelta(seconds=1/self.DATA.sampling_rate))
        self.EVENT = self.EVENT[(self.EVENT['DT'] > self.times[0]) & (self.EVENT['DT'] < self.times[-1])]



        self.logoPath = '{}/SeisLoc.png'.format('/'.join(ilib.__file__.split('/')[:-2]))

        self.MAPmax   = np.max(MAP)


        self.CoaTraceVLINE  = None
        self.CoaValVLINE    = None
        
        self.CoaXYPlt       = None
        self.CoaYZPlt       = None
        self.CoaXZPlt       = None
        self.CoaXYPlt_VLINE = None
        self.CoaXYPlt_HLINE = None
        self.CoaYZPlt_VLINE = None
        self.CoaYZPlt_HLINE = None
        self.CoaXZPlt_VLINE = None
        self.CoaXZPlt_HLINE = None
        self.CoaArriavalTP  = None
        self.CoaArriavalTS  = None

    def CoalescenceImage(self,TimeSliceIndex):
        '''
            Takes the outputted coalescence values to plot a video over time
        '''



        TimeSlice = self.times[TimeSliceIndex]
        index = np.where(self.EVENT['DT'] == TimeSlice)[0][0]
        indexVal = self.LUT.coord2loc(np.array([[self.EVENT['X'].iloc[index],self.EVENT['Y'].iloc[index],self.EVENT['Z'].iloc[index]]])).astype(int)[0]
        indexCoord = np.array([[self.EVENT['X'].iloc[index],self.EVENT['Y'].iloc[index],self.EVENT['Z'].iloc[index]]])[0,:]

        print(indexVal)

        # ----------------  Defining the Plot Area -------------------
        fig = plt.figure(figsize=(30,15))
        fig.patch.set_facecolor('white')
        Coa_XYSlice  =  plt.subplot2grid((3, 5), (0, 0), colspan=2,rowspan=2)
        Coa_YZSlice  =  plt.subplot2grid((3, 5), (2, 0), colspan=2)
        Coa_XZSlice  =  plt.subplot2grid((3, 5), (0, 2), rowspan=2)
        Coa_Trace    =  plt.subplot2grid((3, 5), (0, 3), colspan=2,rowspan=2)
        Coa_Logo     =  plt.subplot2grid((3, 5), (2, 2))
        Coa_CoaVal   =  plt.subplot2grid((3, 5), (2, 3), colspan=2)


        # ---------------- Plotting the Traces -----------
        STIn = np.where(self.times == self.EVENT['DT'].iloc[0])[0][0]
        ENIn = np.where(self.times == self.EVENT['DT'].iloc[-1])[0][0]

        # ------------ Defining the stations in alphabetical order --------
        if self.RangeOrder == True: 
            ttp = self.LUT.get_value_at('TIME_P',np.array([indexVal[0],indexVal[1],indexVal[2]]))[0][::-1]
            StaInd = np.argsort(ttp)
        else:
            StaInd = np.argsort(self.DATA.StationInformation['Name'])[::-1]


        # ------------ Defining the stations in alphabetical order --------


        for ii in range(self.DATA.signal.shape[1]): 
           if self.FilteredSignal == False:
                   Coa_Trace.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.signal[0,ii,:]/np.max(abs(self.DATA.signal[0,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'r',linewidth=0.5)
                   Coa_Trace.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.signal[1,ii,:]/np.max(abs(self.DATA.signal[1,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'b',linewidth=0.5)
                   Coa_Trace.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.signal[2,ii,:]/np.max(abs(self.DATA.signal[2,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'g',linewidth=0.5)
           else:
                   Coa_Trace.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.FilteredSignal[0,ii,:]/np.max(abs(self.DATA.FilteredSignal[0,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'r',linewidth=0.5)
                   Coa_Trace.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.FilteredSignal[1,ii,:]/np.max(abs(self.DATA.FilteredSignal[1,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'b',linewidth=0.5)
                   Coa_Trace.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.FilteredSignal[2,ii,:]/np.max(abs(self.DATA.FilteredSignal[2,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'g',linewidth=0.5)

        # ---------------- Plotting the Station Travel Times -----------
        for i in range(self.LUT.get_value_at('TIME_P',np.array([indexVal[0],indexVal[1],indexVal[2]]))[0].shape[0]):
           tp = self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_P',np.array([indexVal[0],indexVal[1],indexVal[2]]))[0][i])
           ts = self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal[0],indexVal[1],indexVal[2]]))[0][i])

           if i == 0:
               TP = tp
               TS = ts
           else:
               TP = np.append(TP,tp)
               TS = np.append(TS,ts)


        self.CoaArriavalTP = Coa_Trace.scatter(TP,(StaInd+1),40,'pink',marker='v')
        self.CoaArriavalTS = Coa_Trace.scatter(TS,(StaInd+1),40,'purple',marker='v')

#        Coa_Trace.set_ylim([0,ii+2])
        Coa_Trace.set_xlim([self.DATA.startTime+timedelta(seconds=1.6),np.max(TS)])
        #Coa_Trace.get_xaxis().set_ticks([])
        Coa_Trace.yaxis.tick_right()
        Coa_Trace.yaxis.set_ticks(StaInd+1)
        Coa_Trace.yaxis.set_ticklabels(self.DATA.StationInformation['Name'])
        self.CoaTraceVLINE = Coa_Trace.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])],0,1000,linestyle='--',linewidth=2,color='r')

        # ------------- Plotting the Coalescence Function ----------- 
        Coa_CoaVal.plot(self.EVENT['DT'],self.EVENT['COA'])
        Coa_CoaVal.set_ylabel('Coalescence Value')
        Coa_CoaVal.set_xlabel('Date-Time')
        Coa_CoaVal.yaxis.tick_right()
        Coa_CoaVal.yaxis.set_label_position("right")
        Coa_CoaVal.set_xlim([self.EVENT['DT'].iloc[0],self.EVENT['DT'].iloc[-1]])
        Coa_CoaVal.format_xdate = mdates.DateFormatter('%Y-%m-%d') #FIX - Not working
        for tick in Coa_CoaVal.get_xticklabels():
                tick.set_rotation(45)
        self.CoaValVLINE   = Coa_CoaVal.axvline(TimeSlice,0,1000,linestyle='--',linewidth=2,color='r')



        #  ------------- Plotting the Coalescence Value Slices -----------
        gridX,gridY = np.mgrid[min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]))/self.LUT.cell_count[0], min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]))/self.LUT.cell_count[1]]
        self.CoaXYPlt = Coa_XYSlice.pcolormesh(gridX,gridY,self.MAP[:,:,int(indexVal[2]),int(TimeSliceIndex-STIn)]/self.MAPmax,vmin=0,vmax=1,cmap=self.CMAP)
        Coa_XYSlice.set_xlim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0])])
        Coa_XYSlice.set_ylim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1])])
        self.CoaXYPlt_VLINE = Coa_XYSlice.axvline(x=indexCoord[0],linestyle='--',linewidth=2,color='k')
        self.CoaXYPlt_HLINE = Coa_XYSlice.axhline(y=indexCoord[1],linestyle='--',linewidth=2,color='k')


        gridX,gridY = np.mgrid[min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]))/self.LUT.cell_count[0], min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]))/self.LUT.cell_count[2]]
        self.CoaYZPlt = Coa_YZSlice.pcolormesh(gridX,gridY,self.MAP[:,int(indexVal[1]),:,int(TimeSliceIndex-STIn)]/self.MAPmax,vmin=0,vmax=1,cmap=self.CMAP)
        Coa_YZSlice.set_xlim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0])])
        Coa_YZSlice.set_ylim([max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]),min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2])])
        self.CoaYZPlt_VLINE = Coa_YZSlice.axvline(x=indexCoord[0],linestyle='--',linewidth=2,color='k')
        self.CoaYZPlt_HLINE = Coa_YZSlice.axhline(y=indexCoord[2],linestyle='--',linewidth=2,color='k')
        Coa_YZSlice.invert_yaxis()


        gridX,gridY = np.mgrid[min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]))/self.LUT.cell_count[2], min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]))/self.LUT.cell_count[1]]
        self.CoaXZPlt = Coa_XZSlice.pcolormesh(gridX,gridY,np.transpose(self.MAP[int(indexVal[0]),:,:,int(TimeSliceIndex-STIn)])/self.MAPmax,vmin=0,vmax=1,cmap=self.CMAP)
        Coa_XZSlice.set_xlim([max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]),min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2])])
        Coa_XZSlice.set_ylim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1])])
        self.CoaXZPlt_VLINE = Coa_XZSlice.axvline(x=indexCoord[2],linestyle='--',linewidth=2,color='k')
        self.CoaXZPlt_HLINE = Coa_XZSlice.axhline(y=indexCoord[1],linestyle='--',linewidth=2,color='k')



        #  ------------- Plotting the station Locations -----------
        Coa_XYSlice.scatter(self.LUT.station_data['Longitude'],self.LUT.station_data['Latitude'],15,'k',marker='^')
        Coa_YZSlice.scatter(self.LUT.station_data['Longitude'],self.LUT.station_data['Elevation'],15,'k',marker='^')
        Coa_XZSlice.scatter(self.LUT.station_data['Elevation'],self.LUT.station_data['Latitude'],15,'k',marker='<')
        for i,txt in enumerate(self.LUT.station_data['Name']):
            Coa_XYSlice.annotate(txt,[self.LUT.station_data['Longitude'][i],self.LUT.station_data['Latitude'][i]])


    #  ------------- Plotting the XYFiles -----------
        if self.XYFiles != None:
           XYFiles = pd.read_csv(self.XYFiles,names=['File','Color','Linewidth','Linestyle'])
           c=0
           for ff in XYFiles['File']:
                XYF = pd.read_csv(ff,names=['X','Y'])       
                Coa_XYSlice.plot(XYF['X'],XYF['Y'],linestyle=XYFiles['Linestyle'].iloc[c],linewidth=XYFiles['Linewidth'].iloc[c],color=XYFiles['Color'].iloc[c])
                c+=1



        #  ------------- Plotting the station Locations -----------
        try:
            Coa_Logo.axis('off')
            im = mpimg.imread(self.logoPath)
            Coa_Logo.imshow(im)
            Coa_Logo.text(150, 200, r'CoalescenceVideo', fontsize=14,style='italic')
        except:
            'Logo not plotting'

        return fig


    def _CoalescenceVideo_update(self,frame):
        frame = int(frame)
        STIn = np.where(self.times == self.EVENT['DT'].iloc[0])[0][0]
        TimeSlice  = self.times[int(frame)]
        index      = np.where(self.EVENT['DT'] == TimeSlice)[0][0]
        indexVal = self.LUT.coord2loc(np.array([[self.EVENT['X'].iloc[index],self.EVENT['Y'].iloc[index],self.EVENT['Z'].iloc[index]]])).astype(int)[0]
        indexCoord = np.array([[self.EVENT['X'].iloc[index],self.EVENT['Y'].iloc[index],self.EVENT['Z'].iloc[index]]])[0,:]

        # Updating the Coalescence Value and Trace Lines
        self.CoaTraceVLINE.set_xdata(TimeSlice)
        self.CoaValVLINE.set_xdata(TimeSlice)

        # Updating the Coalescence Maps
        self.CoaXYPlt.set_array((self.MAP[:,:,indexVal[2],int(STIn-frame)]/self.MAPmax)[:-1,:-1].ravel())
        self.CoaYZPlt.set_array((self.MAP[:,indexVal[1],:,int(STIn-frame)]/self.MAPmax)[:-1,:-1].ravel())
        self.CoaXZPlt.set_array((np.transpose(self.MAP[indexVal[0],:,:,int(STIn-frame)])/self.MAPmax)[:-1,:-1].ravel())

        # Updating the Coalescence Lines
        self.CoaXYPlt_VLINE.set_xdata(indexCoord[0])
        self.CoaXYPlt_HLINE.set_ydata(indexCoord[1])
        self.CoaYZPlt_VLINE.set_xdata(indexCoord[0])
        self.CoaYZPlt_HLINE.set_ydata(indexCoord[2])
        self.CoaXZPlt_VLINE.set_xdata(indexCoord[2])
        self.CoaXZPlt_HLINE.set_ydata(indexCoord[1])


        # Updating the station travel-times
        for i in range(self.LUT.get_value_at('TIME_P',np.array([indexVal]))[0].shape[0]):
            try:
                tp = np.argmin(abs((self.times.astype(datetime) - (TimeSlice.astype(datetime) + timedelta(seconds=self.LUT.get_value_at('TIME_P',np.array([indexVal]))[0][i])))/timedelta(seconds=1)))
                ts = np.argmin(abs((self.times.astype(datetime) - (TimeSlice.astype(datetime) + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal]))[0][i])))/timedelta(seconds=1)))
            except:
                tp=0
                ts=0



            if i == 0:
                TP = tp
                TS = ts
            else:
                TP = np.append(TP,tp)
                TS = np.append(TS,ts)

        self.CoaArriavalTP.set_offsets(np.c_[TP,(np.arange(self.DATA.signal.shape[1])+1)])
        self.CoaArriavalTS.set_offsets(np.c_[TS,(np.arange(self.DATA.signal.shape[1])+1)])


        # # Updating the station travel-times
        # self.CoaArriavalTP.remove()
        # self.CoaArriavalTS.remove()
        # self.CoaArriavalTS = None
        # self.CoaArriavalTP = None
        # for i in range(self.LUT.get_value_at('TIME_P',np.array([indexVal]))[0].shape[0]):
        #     if i == 0:
        #         TP = TimeSlice + timedelta(seconds=self.LUT.get_value_at('TIME_P',np.array([indexVal]))[0][i])
        #         TS = TimeSlice + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal]))[0][i])
        #     else:
        #         TP = np.append(TP,(TimeSlice + timedelta(seconds=self.LUT.get_value_at('TIME_P',np.array([indexVal]))[0][i])))
        #         TS = np.append(TS,(TimeSlice + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal]))[0][i])))

        # self.CoaArriavalTP = Coa_Trace.scatter(TP,(np.arange(self.DATA.signal.shape[1])+1),15,'r',marker='v')
        # self.CoaArriavalTS = Coa_Trace.scatter(TS,(np.arange(self.DATA.signal.shape[1])+1),15,'b',marker='v')


    def CoalescenceTrace(self,SaveFilename=None):

        # Determining the maginal window value from the coalescence function
        mMAP = self.CoaMAP
        # mMAP = np.log(np.sum(np.exp(mMAP),axis=-1))
        # mMAP = mMAP/np.max(mMAP)
        # mMAP_Cutoff = np.percentile(mMAP,95)
        # mMAP[mMAP < mMAP_Cutoff] = mMAP_Cutoff 
        # mMAP = mMAP - mMAP_Cutoff 
        # mMAP = mMAP/np.max(mMAP)
        indexVal = np.where(mMAP == np.max(mMAP))
        indexCoord = self.LUT.xyz2coord(self.LUT.loc2xyz(np.array([[indexVal[0][0],indexVal[1][0],indexVal[2][0]]])))




        # Looping through all stations
        ii=0
        while ii < self.DATA.signal.shape[1]: 
                fig = plt.figure(figsize=(30,15))

                # Defining the plot
                fig.patch.set_facecolor('white')
                XTrace_Seis  =  plt.subplot(322)
                YTrace_Seis  =  plt.subplot(324)
                ZTrace_Seis  =  plt.subplot(321)
                P_Onset      =  plt.subplot(323)
                S_Onset      =  plt.subplot(326)



                # --- If trace is blank then remove and don't plot ---



                # Plotting the X-trace
                XTrace_Seis.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.FilteredSignal[0,ii,:]/np.max(abs(self.DATA.FilteredSignal[0,ii,:])))*self.TraceScaling,'r',linewidth=0.5)
                YTrace_Seis.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.FilteredSignal[1,ii,:]/np.max(abs(self.DATA.FilteredSignal[1,ii,:])))*self.TraceScaling,'b',linewidth=0.5)
                ZTrace_Seis.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.FilteredSignal[2,ii,:]/np.max(abs(self.DATA.FilteredSignal[2,ii,:])))*self.TraceScaling,'g',linewidth=0.5)
                P_Onset.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),self.DATA.SNR_P[ii,:],'r',linewidth=0.5)
                S_Onset.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),self.DATA.SNR_S[ii,:],'b',linewidth=0.5)


                # Defining Pick and Error
                PICKS_df = self.StationPick['Pick']
                STATION_pick = PICKS_df[PICKS_df['Name'] == self.LUT.station_data['Name'][ii]].reset_index(drop=True)
                if len(STATION_pick) > 0:
                    STATION_pick = STATION_pick.replace('-1.0',np.nan)


                    for jj in range(len(STATION_pick)):
                        if np.isnan(STATION_pick['PickError'].iloc[jj]):
                            continue 

                        if STATION_pick['Phase'].iloc[jj] == 'P':
                            ZTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=-STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            ZTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=+STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            ZTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj]))

                            # S_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=-STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            # S_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=+STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            # S_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj]))

                            yy = gaussian_func(self.StationPick['GAU_P'][ii]['xdata'],self.StationPick['GAU_P'][ii]['popt'][0],self.StationPick['GAU_P'][ii]['popt'][1],self.StationPick['GAU_P'][ii]['popt'][2])
                            P_Onset.plot(self.StationPick['GAU_P'][ii]['xdata_dt'],yy)
                            P_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=-STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            P_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=+STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            P_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj]))




                        else:
                            YTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=-STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            YTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=+STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            YTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj]))

                            XTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=-STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            XTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=+STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            XTrace_Seis.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj]))

                            yy = gaussian_func(self.StationPick['GAU_S'][ii]['xdata'],self.StationPick['GAU_S'][ii]['popt'][0],self.StationPick['GAU_S'][ii]['popt'][1],self.StationPick['GAU_S'][ii]['popt'][2])
                            S_Onset.plot(self.StationPick['GAU_S'][ii]['xdata_dt'],yy)

                            S_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=-STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            S_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj])+timedelta(seconds=+STATION_pick['PickError'].iloc[jj]/2),linestyle='--')
                            S_Onset.axvline(pd.to_datetime(STATION_pick['PickTime'].iloc[jj]))

                ZTrace_Seis.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii]),color='red')
                P_Onset.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii]),color='red')
                YTrace_Seis.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii]),color='red')
                XTrace_Seis.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii]),color='red')
                S_Onset.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii]),color='red')

                        
                # Refining the window as around the pick time
                MINT = pd.to_datetime(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=0.5*self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii]))
                MAXT = pd.to_datetime(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=1.5*self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][ii]))

                XTrace_Seis.set_xlim([MINT,MAXT])
                YTrace_Seis.set_xlim([MINT,MAXT])
                ZTrace_Seis.set_xlim([MINT,MAXT])
                P_Onset.set_xlim([MINT,MAXT])
                S_Onset.set_xlim([MINT,MAXT])


                fig.suptitle('Trace for Station {} - PPick = {}, SPick = {}'.format(self.LUT.station_data['Name'][ii],self.StationPick['GAU_P'][ii]['PickValue'],self.StationPick['GAU_S'][ii]['PickValue']))


                
                if SaveFilename == None:
                   plt.show()
                else:
                   plt.savefig('{}_CoalescenceTrace_{}.pdf'.format(SaveFilename,self.LUT.station_data['Name'][ii]))
                   plt.close("all")


                ii+=1
            

    def CoalescenceVideo(self,SaveFilename=None):
        STIn = np.where(self.times == self.EVENT['DT'].iloc[0])[0][0]
        ENIn = np.where(self.times == self.EVENT['DT'].iloc[-1])[0][0]

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=4, metadata=dict(artist='Ulvetanna'), bitrate=1800)


        FIG = self.CoalescenceImage(STIn)
        ani = animation.FuncAnimation(FIG, self._CoalescenceVideo_update, frames=np.linspace(STIn,ENIn-1,200),blit=False,repeat=False) 

        if SaveFilename == None:
            plt.show()
        else:
            ani.save('{}_CoalescenceVideo.mp4'.format(SaveFilename),writer=writer)

    def CoalescenceMarginal(self,SaveFilename=None):
        '''
            Generates a Marginal window about the event to determine the error.

            # Redefine the marginal as instead of the whole coalescence period, gaussian fit to the coalescence value 
            then take the 1st std to define the time window and use this

        '''

        TimeSliceIndex = np.where(self.times == self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])])[0][0]
        TimeSlice = self.times[TimeSliceIndex]
        index = np.where(self.EVENT['DT'] == TimeSlice)[0][0]
        indexVal1 = self.LUT.coord2loc(np.array([[self.EVENT['X'].iloc[index],self.EVENT['Y'].iloc[index],self.EVENT['Z'].iloc[index]]])).astype(int)[0]
        indexCoord = np.array([[self.EVENT['X'].iloc[index],self.EVENT['Y'].iloc[index],self.EVENT['Z'].iloc[index]]])[0,:]

        # Determining the maginal window value from the coalescence function
        mMAP = self.CoaMAP
        # mMAP = np.log(np.sum(np.exp(mMAP),axis=-1))
        # mMAP = mMAP/np.max(mMAP)
        # mMAP_Cutoff = np.percentile(mMAP,95)
        # mMAP[mMAP < mMAP_Cutoff] = mMAP_Cutoff 
        # mMAP = mMAP - mMAP_Cutoff 
        # mMAP = mMAP/np.max(mMAP)
        indexVal = np.where(mMAP == np.max(mMAP))
        indexCoord = self.LUT.xyz2coord(self.LUT.loc2xyz(np.array([[indexVal[0][0],indexVal[1][0],indexVal[2][0]]])))


        # Defining the plots to be represented
        fig = plt.figure(figsize=(30,15))
        fig.patch.set_facecolor('white')
        Coa_XYSlice  =  plt.subplot2grid((3, 5), (0, 0), colspan=2,rowspan=2)
        Coa_YZSlice  =  plt.subplot2grid((3, 5), (2, 0), colspan=2)
        Coa_XZSlice  =  plt.subplot2grid((3, 5), (0, 2), rowspan=2)
        Coa_Trace    =  plt.subplot2grid((3, 5), (0, 3), colspan=2,rowspan=2)
        Coa_Logo     =  plt.subplot2grid((3, 5), (2, 2))
        Coa_CoaVal   =  plt.subplot2grid((3, 5), (2, 3), colspan=2)



        # ---------------- Plotting the Traces -----------
        STIn = np.where(self.times == self.EVENT['DT'].iloc[0])[0][0]
        ENIn = np.where(self.times == self.EVENT['DT'].iloc[-1])[0][0]
        #print(STIn,ENIn)



        # --------------- Ordering by distance to event --------------
        if self.RangeOrder == True: 
            ttp = self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][::-1]
            StaInd = np.argsort(ttp)
        else:
            StaInd = np.argsort(self.DATA.StationInformation['Name'])[::-1]


        for ii in range(self.DATA.signal.shape[1]): 
           if self.FilteredSignal == False:
                   Coa_Trace.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.signal[0,ii,:]/np.max(abs(self.DATA.signal[0,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'r',linewidth=0.5)
                   Coa_Trace.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.signal[1,ii,:]/np.max(abs(self.DATA.signal[1,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'b',linewidth=0.5)
                   Coa_Trace.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.signal[2,ii,:]/np.max(abs(self.DATA.signal[2,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'g',linewidth=0.5)
           else:
                   Coa_Trace.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.FilteredSignal[0,ii,:]/np.max(abs(self.DATA.FilteredSignal[0,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'r',linewidth=0.5)
                   Coa_Trace.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.FilteredSignal[1,ii,:]/np.max(abs(self.DATA.FilteredSignal[1,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'b',linewidth=0.5)
                   Coa_Trace.plot(np.arange(self.DATA.startTime,self.DATA.endTime+timedelta(seconds=1/self.DATA.sampling_rate),timedelta(seconds=1/self.DATA.sampling_rate)),(self.DATA.FilteredSignal[2,ii,:]/np.max(abs(self.DATA.FilteredSignal[2,ii,:])))*self.TraceScaling+(StaInd[ii]+1),'g',linewidth=0.5)

        # ---------------- Plotting the Station Travel Times -----------
        for i in range(self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0].shape[0]):
           tp = self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_P',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][i])
           ts = self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])] + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal[0][0],indexVal[1][0],indexVal[2][0]]))[0][i])

           if i == 0:
               TP = tp
               TS = ts
           else:
               TP = np.append(TP,tp)
               TS = np.append(TS,ts)


        self.CoaArriavalTP = Coa_Trace.scatter(TP,(StaInd+1),40,'pink',marker='v')
        self.CoaArriavalTS = Coa_Trace.scatter(TS,(StaInd+1),40,'purple',marker='v')

#        Coa_Trace.set_ylim([0,ii+2])
        Coa_Trace.set_xlim([self.DATA.startTime+timedelta(seconds=1.6),np.max(TS)])
        #Coa_Trace.get_xaxis().set_ticks([])
        Coa_Trace.yaxis.tick_right()
        Coa_Trace.yaxis.set_ticks(StaInd+1)
        Coa_Trace.yaxis.set_ticklabels(self.DATA.StationInformation['Name'])
        self.CoaTraceVLINE = Coa_Trace.axvline(self.EVENT['DT'].iloc[np.argmax(self.EVENT['COA'])],0,1000,linestyle='--',linewidth=2,color='r')

        # ------------- Plotting the Coalescence Function ----------- 
        Coa_CoaVal.plot(self.EVENT['DT'],self.EVENT['COA'])
        Coa_CoaVal.set_ylabel('Coalescence Value')
        Coa_CoaVal.set_xlabel('Date-Time')
        Coa_CoaVal.yaxis.tick_right()
        Coa_CoaVal.yaxis.set_label_position("right")
        Coa_CoaVal.set_xlim([self.EVENT['DT'].iloc[0],self.EVENT['DT'].iloc[-1]])
        Coa_CoaVal.format_xdate = mdates.DateFormatter('%Y-%m-%d') #FIX - Not working
        for tick in Coa_CoaVal.get_xticklabels():
                tick.set_rotation(45)
        self.CoaValVLINE   = Coa_CoaVal.axvline(TimeSlice,0,1000,linestyle='--',linewidth=2,color='r')



        # ------------- Spatial Function  ----------- 

        # Plotting the marginal window
        gridX,gridY = np.mgrid[min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]))/self.LUT.cell_count[0], min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]))/self.LUT.cell_count[1]]
        rect = Rectangle((np.min(gridX), np.min(gridY)), np.max(gridX)-np.min(gridX),np.max(gridY)-np.min(gridY))
        pc = PatchCollection([rect], facecolor='k')
        Coa_XYSlice.add_collection(pc)
        Coa_XYSlice.pcolormesh(gridX,gridY,mMAP[:,:,int(indexVal[2][0])],cmap=self.CMAP,edgecolors='face')
        CS = Coa_XYSlice.contour(gridX,gridY,mMAP[:,:,int(indexVal[2][0])],levels=[0.65,0.75,0.95],colors=('g','m','k'))
        Coa_XYSlice.clabel(CS, inline=1, fontsize=10)
        Coa_XYSlice.set_xlim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0])])
        Coa_XYSlice.set_ylim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1])])
        Coa_XYSlice.axvline(x=indexCoord[0][0],linestyle='--',linewidth=2,color=self.LineStationColor)
        Coa_XYSlice.axhline(y=indexCoord[0][1],linestyle='--',linewidth=2,color=self.LineStationColor)

        gridX,gridY = np.mgrid[min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]))/self.LUT.cell_count[0], min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]))/self.LUT.cell_count[2]]
        rect = Rectangle((np.min(gridX), np.min(gridY)), np.max(gridX)-np.min(gridX),np.max(gridY)-np.min(gridY))
        pc = PatchCollection([rect], facecolor='k')
        Coa_YZSlice.add_collection(pc)
        Coa_YZSlice.pcolormesh(gridX,gridY,mMAP[:,int(indexVal[1][0]),:],cmap=self.CMAP,edgecolors='face')
        CS = Coa_YZSlice.contour(gridX,gridY,mMAP[:,int(indexVal[1][0]),:], levels=[0.65,0.75,0.95],colors=('g','m','k'))
        Coa_YZSlice.clabel(CS, inline=1, fontsize=10)
        Coa_YZSlice.set_xlim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0])])
        Coa_YZSlice.set_ylim([max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]),min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2])])
        Coa_YZSlice.axvline(x=indexCoord[0][0],linestyle='--',linewidth=2,color=self.LineStationColor)
        Coa_YZSlice.axhline(y=indexCoord[0][2],linestyle='--',linewidth=2,color=self.LineStationColor)
        Coa_YZSlice.invert_yaxis()


        gridX,gridY = np.mgrid[min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]))/self.LUT.cell_count[2], min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]))/self.LUT.cell_count[1]]
        rect = Rectangle((np.min(gridX), np.min(gridY)), np.max(gridX)-np.min(gridX),np.max(gridY)-np.min(gridY))
        pc = PatchCollection([rect], facecolor='k')
        Coa_XZSlice.add_collection(pc)
        Coa_XZSlice.pcolormesh(gridX,gridY,mMAP[int(indexVal[0][0]),:,:].transpose(),cmap=self.CMAP,edgecolors='face')
        CS = Coa_XZSlice.contour(gridX,gridY,mMAP[int(indexVal[0][0]),:,:].transpose(),levels =[0.65,0.75,0.95],colors=('g','m','k'))
        Coa_XZSlice.set_xlim([max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]),min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2])])
        Coa_XZSlice.set_ylim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1])])
        Coa_XZSlice.axvline(x=indexCoord[0][2],linestyle='--',linewidth=2,color=self.LineStationColor)
        Coa_XZSlice.axhline(y=indexCoord[0][1],linestyle='--',linewidth=2,color=self.LineStationColor)

        # Plotting the station locations
        Coa_XYSlice.scatter(self.LUT.station_data['Longitude'],self.LUT.station_data['Latitude'],15,marker='^',color=self.LineStationColor)
        Coa_YZSlice.scatter(self.LUT.station_data['Longitude'],self.LUT.station_data['Elevation'],15,marker='^',color=self.LineStationColor)
        Coa_XZSlice.scatter(self.LUT.station_data['Elevation'],self.LUT.station_data['Latitude'],15,marker='<',color=self.LineStationColor)
        for i,txt in enumerate(self.LUT.station_data['Name']):
            Coa_XYSlice.annotate(txt,[self.LUT.station_data['Longitude'][i],self.LUT.station_data['Latitude'][i]],color=self.LineStationColor)

        # Plotting the XYFiles
        if self.XYFiles != None:
           XYFiles = pd.read_csv(self.XYFiles,names=['File','Color','Linewidth','Linestyle'])
           c=0
           for ff in XYFiles['File']:
                XYF = pd.read_csv(ff,names=['X','Y'])       
                Coa_XYSlice.plot(XYF['X'],XYF['Y'],linestyle=XYFiles['Linestyle'].iloc[c],linewidth=XYFiles['Linewidth'].iloc[c],color=XYFiles['Color'].iloc[c])
                c+=1

        # Plotting the logo
        try:
            Coa_Logo.axis('off')
            im = mpimg.imread(self.logoPath)
            Coa_Logo.imshow(im)
            Coa_Logo.text(150, 200, r'Earthquake Location Error', fontsize=10,style='italic')
        except:
            'Logo not plotting'

        if SaveFilename == None:
            plt.show()

        else:
            plt.savefig('{}_EventLocationError.pdf'.format(SaveFilename),dpi=400)
            plt.close('all')




class SeisScanParam:
    '''
       Class that reads in a user defined parameter file for all the required
    scanning Information. Currently takes the 

      _set_param - Definition of the path for the Parameter file to be read

    '''

    def __init__(self, param = None):
        self.lookup_table = None
        self.seis_reader = None
        self.bp_filter_p1 = [2.0, 16.0, 3]
        self.bp_filter_s1 = [2.0, 12.0, 3]
        self.onset_win_p1 = [0.2, 1.0]
        self.onset_win_s1 = [0.2, 1.0]
        self.station_p1 = None
        self.station_s1 = None
        self.detection_threshold = 3.0
        self.detection_downsample = 5
        self.detection_window = 3.0
        self.minimum_velocity = 3000.0
        self.marginal_window = [0.5, 3000.0]
        self.location_method = "Mean"
        self.time_step = 10
        self.StartDateTime=None
        self.EndDateTime=None
        self.Decimate=[1,1,1]

        if param:
            self.load(param)

    def _set_param(self, param):

        # Defining the Model Types to load LUT from
        type = _find(param,("MODEL","Type"))
        
        if (type == "MATLAB"):
            path = _find(param,("MODEL","Path"))
            if path:
                decimate = _find(param,("MODEL","Decimate"))
                self.lookup_table = mload.lut02(path, decimate)

        if (type == "SeisLoc"):
            path = _find(param,("MODEL","Path"))
            if path:
                decimate = _find(param,("MODEL","Decimate"))
                self.lookup_table = mload.lut02(path, decimate)

        if (type == "NLLoc"):
            path = _find(param,("MODEL","Path"))
            if path:
                decimate = _find(param,("MODEL","Decimate"))
                self.lookup_table = mload.lut02(path, decimate)




        # Defining the Seimsic Data to load the information from
        type = _find(param,("SEISMIC","Type"))
        if (type == 'MSEED'):
            path = _find(param,("SEISMIC","Path"))
            if path:
                self.seis_reader = mload.mseed_reader(path)

        # ~ Other possible types to add DAT,SEGY,RAW


        # Defining the Time-Period to scan across 
        scn = _find(param,"SCAN")
        if scn:
            self.StartDateTime = _find(scn,"Start_DateTime",self.StartDateTime)
            self.EndDateTime   = _find(scn,"End_DateTime",self.EndDateTime)
            self.StartDateTime = datetime.strptime(self.StartDateTime,'%Y-%m-%dT%H:%M:%S.%f')
            self.EndDateTime   = datetime.strptime(self.EndDateTime,'%Y-%m-%dT%H:%M:%S.%f')

        # Defining the Parameters for the Coalescence
        scn = _find(param,("PARAM"))
        if scn:
            self.time_step            = _find(scn,"TimeStep",self.time_step)
            self.station_p1           = _find(scn,"StationSelectP",self.station_p1)
            self.station_s1           = _find(scn,"StationSelectS",self.station_s1)
            self.bp_filter_p1         = _find(scn,"SigFiltP1Hz",self.bp_filter_p1)
            self.bp_filter_s1         = _find(scn,"SigFiltS1Hz",self.bp_filter_s1)
            self.onset_win_p1         = _find(scn,"OnsetWinP1Sec",self.onset_win_p1)
            self.onset_win_s1         = _find(scn,"OnsetWinS1Sec",self.onset_win_s1)
            self.detection_downsample = _find(scn,"DetectionDownsample",self.detection_downsample)
            self.detection_window     = _find(scn,"DetectionWindow",self.detection_window)
            self.minimum_velocity     = _find(scn,"MinimumVelocity",self.minimum_velocity)
            self.marginal_window      = _find(scn,"MarginalWindow",self.marginal_window)
            self.location_method      = _find(scn,"LocationMethod",self.location_method)

    def _load_json(self, json_file):
        param = None
        with open(json_file,'r') as fp:
            param = json.load(fp)
        return param

    def load(self, file):
        param = self._load_json(file)
        self._set_param(param)


class SeisScan:

    def __init__(self, DATA, LUT, reader=None, param=None, output_path=None, output_name=None):
        
        lut = cmod.LUT()
        lut.load(LUT)
        self.sample_rate = 1000.0
        self.seis_reader = None
        self.lookup_table = lut
        self.DATA = DATA 

        if param is None:
            param = SeisScanParam()


        self.keep_map = False

        ttmax = np.max(lut.fetch_map('TIME_S'))
        self.pre_pad   = None
        self.post_pad  = round(ttmax)
        self.time_step = 10.0

        self.daten = None
        self.dsnr  = None
        self.dloc  = None

        self.PickThreshold = 1.0
        

        self.bp_filter_p1 = param.bp_filter_p1
        self.bp_filter_s1 = param.bp_filter_s1
        self.onset_win_p1 = param.onset_win_p1
        self.onset_win_s1 = param.onset_win_s1
        self.boxcar_p1 = 0.050
        self.boxcar_s1 = 0.100
        self.station_p1 = param.station_p1
        self.station_s1 = param.station_s1
        self.detection_threshold = param.detection_threshold

        if output_path is not None:
            self.output = SeisOutFile(output_path, output_name)

        else:
            self.outputps = None

        self.raw_data = dict()
        self.filt_data = dict()
        self.onset_data = dict()

        self._initialized = False
        self._station_name = None
        self._station_p1_flg = None
        self._station_s1_flg = None
        self._station_file = None
        self._map = None

        self.snr = None 
        self._data = None


        self.NumberOfCores = 1

        self.DetectionThreshold = 1
        self.MarginalWindow     = 30
        self.CoalescenceGrid    = False
        self.CoalescenceVideo   = False
        self.CoalescencePicture = False
        self.CoalescenceTrace   = False
        self.CutMSEED           = False
        self.PickingType        = 'Gaussian'
        self.LocationError      = 0.95

        self.Output_SampleRate = None 


        #self.plot = SeisPlot(lut)

        self.MAP = None
        self.CoaMAP = None
        self.EVENT = None
        self.XYFiles = None


    def _pre_proc_p1(self, sig_z, srate):
        lc, hc, ord = self.bp_filter_p1           # Apply - Bandpass Filter  - information defined in ParameterFile/Inputs
        sig_z = filter(sig_z, srate, lc, hc, ord) # Apply - Butter filter
        self.DATA.FilteredSignal[2,:,:] = sig_z
        return sig_z

    def _pre_proc_s1(self, sig_e, sig_n, srate):
        lc, hc, ord = self.bp_filter_s1               # Apply - Bandpass Filter  - information defined in ParameterFile/Inputs
        sig_e = filter(sig_e, srate, lc, hc, ord) # Apply - Butter filter on E
        sig_n = filter(sig_n, srate, lc, hc, ord) # Apply - Butter filter on N
        self.DATA.FilteredSignal[0,:,:] = sig_n
        self.DATA.FilteredSignal[1,:,:] = sig_e
        return sig_e, sig_n

    def _compute_onset_p1(self, sig_z, srate):
        stw, ltw = self.onset_win_p1             # Define the STW and LTW for the onset function
        stw = int(stw * srate) + 1               # Changes the onset window to actual samples
        ltw = int(ltw * srate) + 1               # Changes the onset window to actual samples
        sig_z = self._pre_proc_p1(sig_z, srate)  # Apply the pre-processing defintion
        self.filt_data['sigz'] = sig_z           # defining the data to pass 
        sig_z_raw,sig_z = onset(sig_z, stw, ltw)           # Determine the onset function using definition
        self.onset_data['sigz'] = sig_z          # Define the onset function from the data 
        return sig_z_raw,sig_z

    def _compute_onset_s1(self, sig_e, sig_n, srate):
        stw, ltw = self.onset_win_s1                                            # Define the STW and LTW for the onset function
        stw = int(stw * srate) + 1                                              # Changes the onset window to actual samples
        ltw = int(ltw * srate) + 1                                              # Changes the onset window to actual samples
        sig_e, sig_n = self._pre_proc_s1(sig_e, sig_n, srate)                   # Apply the pre-processing defintion
        self.filt_data['sige'] = sig_e                                          # Defining filtered signal to pass
        self.filt_data['sign'] = sig_n                                          # Defining filtered signal to pass
        sig_e_raw,sig_e = onset(sig_e, stw, ltw)                                          # Determine the onset function from the filtered signal
        sig_n_raw,sig_n = onset(sig_n, stw, ltw)                                          # Determine the onset function from the filtered signal
        self.onset_data['sige'] = sig_e                                         # Define the onset function from the data
        self.onset_data['sign'] = sig_n                                         # Define the onset function from the data                
        snr = np.sqrt(sig_e * sig_e + sig_n * sig_n)
        snr_raw = np.sqrt(sig_e_raw * sig_e_raw + sig_n_raw * sig_n_raw)                            # Define the combined onset function from E & N
        self.onset_data['sigs'] = snr
        return snr_raw,snr


    def _compute(self, cstart,cend, samples,station_avaliability):

        srate = self.sample_rate


        avaInd = np.where(station_avaliability == 1)[0]
        sige = samples[0]
        sign = samples[1]
        sigz = samples[2]

        # Demeaning the data 
        #sige = sige - np.mean(sige,axis=1)
        #sign = sign - np.mean(sign,axis=1)
        #sigz = sigz - np.mean(sigz,axis=1)

        snr_p1_raw,snr_p1 = self._compute_onset_p1(sigz, srate)
        snr_s1_raw,snr_s1 = self._compute_onset_s1(sige, sign, srate)
        self.DATA.SNR_P = snr_p1
        self.DATA.SNR_S = snr_s1
        self.DATA.SNR_P_raw = snr_p1_raw
        self.DATA.SNR_S_raw = snr_s1_raw

        #self._Gaussian_Coalescence()


        snr = np.concatenate((self.DATA.SNR_P, self.DATA.SNR_S))
        snr[np.isnan(snr)] = 0
        
        
        ttp = self.lookup_table.fetch_index('TIME_P', srate)
        tts = self.lookup_table.fetch_index('TIME_S', srate)
        tt = np.c_[ttp, tts]

        nchan, tsamp = snr.shape

        pre_smp = int(self.pre_pad * int(srate))
        pos_smp = int(self.post_pad * int(srate))
        nsamp = tsamp - pre_smp - pos_smp
        daten = 0.0 - pre_smp / srate

        ncell = tuple(self.lookup_table.cell_count)

        if self._map is None:
            #print('  Allocating memory: {}'.format(ncell + (tsamp,)))
            self._map = np.zeros(ncell + (nsamp,), dtype=np.float64)

        dind = np.zeros(nsamp, np.int64)
        dsnr = np.zeros(nsamp, np.double)

        # ilib.scan(snr, tt, 0, pre_smp + nsamp +pos_smp, self._map, self.NumberOfCores)
        # ilib.detect(self._map, dsnr, dind, 0, pre_smp + nsamp +pos_smp, self.NumberOfCores)
        # daten = np.arange((cstart+timedelta(seconds=self.pre_pad)), (cend + timedelta(seconds=-self.post_pad) + timedelta(seconds=1/srate)),timedelta(seconds=1/srate)) 
        # dsnr = np.exp((dsnr / nchan) - 1.0)
        # #dsnr = classic_sta_lta(np.exp((dsnr / nchan) - 1.0),self.onset_win_p1[0]*self.sample_rate*0.5,self.onset_win_p1[1]*self.sample_rate*0.5)
        # dsnr = dsnr[pre_smp:pre_smp + nsamp]
        # dloc  = self.lookup_table.index2xyz(dind[pre_smp:pre_smp + nsamp])
        # MAP   = self._map[:,:,:,(pre_smp+1):pre_smp + nsamp]

        ilib.scan(snr, tt, pre_smp, pos_smp, nsamp, self._map, self.NumberOfCores)
        ilib.detect(self._map, dsnr, dind, 0,nsamp, self.NumberOfCores)
        daten = np.arange((cstart+timedelta(seconds=self.pre_pad)), (cend + timedelta(seconds=-self.post_pad) + timedelta(seconds=1/srate)),timedelta(seconds=1/srate)) 
        dsnr = np.exp((dsnr / nchan) - 1.0)
        #dsnr = classic_sta_lta(np.exp((dsnr / nchan) - 1.0),self.onset_win_p1[0]*self.sample_rate*0.5,self.onset_win_p1[1]*self.sample_rate*0.5)
        dsnr = dsnr
        dloc  = self.lookup_table.index2xyz(dind)
        MAP   = self._map


        self._map = None
        return daten, dsnr, dloc, MAP


    def _continious_compute(self,starttime,endtime):
        ''' 
            Continious seismic compute from 

        '''

        # 1. variables check
        # 2. Defining the pre- and post- padding
        # 3.  

        CoaV = 1.0

        self.StartDateTime = datetime.strptime(starttime,'%Y-%m-%dT%H:%M:%S.%f')
        self.EndDateTime   = datetime.strptime(endtime,'%Y-%m-%dT%H:%M:%S.%f')

        # Deleting the scan if it exists alreadys
        self.output.del_scan()

        
        # ------- Continious Seismic Detection ------
        print('==============================================================================================================================')
        print('   SeisLoc - Coalescence Scanning : PATH:{} - NAME:{}'.format(self.output.path, self.output.name))
        print('======================================================================')
        print('   Continious Seismic Processing for {} to {}'.format(datetime.strftime(self.StartDateTime,'%Y-%m-%dT%H:%M:%S.%f'),datetime.strftime(self.EndDateTime,'%Y-%m-%dT%H:%M:%S.%f')))
        print('==============================================================================================================================')

        i = 0 
        while self.EndDateTime >= (self.StartDateTime + timedelta(seconds=self.time_step*(i+1))):
            cstart =  self.StartDateTime + timedelta(seconds=self.time_step*i) - timedelta(seconds=self.pre_pad)
            cend   =  self.StartDateTime + timedelta(seconds=self.time_step*(i+1)) + timedelta(seconds=self.post_pad)

            print('~~~~~~~~~~~~~ Processing - {} to {} ~~~~~~~~~~~~~'.format(datetime.strftime(cstart,'%Y-%m-%dT%H:%M:%S.%f'),datetime.strftime(cend,'%Y-%m-%dT%H:%M:%S.%f'))) 

            self.DATA.read_mseed(datetime.strftime(cstart,'%Y-%m-%dT%H:%M:%S.%f'),datetime.strftime(cend,'%Y-%m-%dT%H:%M:%S.%f'),self.sample_rate)
            #daten, dsnr, dloc = self._compute_s1(0.0, DATA.signal)
            daten, dsnr, dloc, map = self._compute(cstart,cend, self.DATA.signal,self.DATA.station_avaliability)

            dcoord = self.lookup_table.xyz2coord(dloc)
            self.output.FileSampleRate = self.Output_SampleRate

#            self.output.write_scan(daten[:-1],CoaVp[:-1],dcoord[:-1,:])

            if i == 0:
                CoaVp = dsnr + (CoaV-dsnr[0])
                self.output.write_scan(daten[:-1],CoaVp[:-1],dcoord[:-1,:])
                CoaV=CoaVp[-1]
            else:
                CoaVp = dsnr + (CoaV-dsnr[0])
                self.output.write_scan(daten[:-1],CoaVp[:-1],dcoord[:-1,:])
                CoaV=CoaVp[-1]

            i += 1



        # Changing format of SCN file to reduce filesize
        #self.output.write_decscan(self.DATA.sampling_rate)
        del daten, dsnr, dloc, map



    def _Trigger_scn(self,CoaVal,starttime,endtime):


        # Defining when exceeded threshold
        CoaVal = CoaVal[CoaVal['COA'] > self.DetectionThreshold] 
        CoaVal = CoaVal[(CoaVal['DT'] >= datetime.strptime(starttime,'%Y-%m-%dT%H:%M:%S.%f')) & (CoaVal['DT'] <= datetime.strptime(endtime,'%Y-%m-%dT%H:%M:%S.%f'))]
        CoaVal = CoaVal.reset_index(drop=True)

        # ----------- Determining the initial triggered events, not inspecting overlaps ------
        c = 0
        e = 1
        while c < len(CoaVal):

            # Determining the index when above the level and maximum value
            d=c
            while CoaVal['DT'].iloc[d] + timedelta(seconds=1/self.sample_rate) == CoaVal['DT'].iloc[d+1]:
                d+=1
                if d+1 >= len(CoaVal)-1:
                    d=len(CoaVal)-1
                    break



            indmin = c
            indmax = d    
            indVal = np.argmax(CoaVal['COA'].iloc[np.arange(c,d+1)])

            # Determining the times for min,max and max coalescence value
            TimeMin = CoaVal['DT'].iloc[indmin]
            TimeMax = CoaVal['DT'].iloc[indmax]
            TimeVal = CoaVal['DT'].iloc[indVal]

            COA_V = CoaVal['COA'].iloc[indVal]
            COA_X = CoaVal['X'].iloc[indVal]
            COA_Y = CoaVal['Y'].iloc[indVal]
            COA_Z = CoaVal['Z'].iloc[indVal]



            if (TimeVal-TimeMin) < timedelta(seconds=0.5*self.MarginalWindow):
                TimeMin = CoaVal['DT'].iloc[indmin] + timedelta(seconds=-0.5*self.MarginalWindow)
            if (TimeMax - TimeVal) < timedelta(seconds=0.5*self.MarginalWindow):
                TimeMax = CoaVal['DT'].iloc[indmax] + timedelta(seconds=0.5*self.MarginalWindow)

            
            # Appending these triggers to array
            if 'IntEvents' not in vars():
                IntEvents = pd.DataFrame([[e,TimeVal,COA_V,COA_X,COA_Y,COA_Z,TimeMin,TimeMax]],columns=['EventNum','CoaTime','COA_V','COA_X','COA_Y','COA_Z','MinTime','MaxTime'])
            else:
                dat       = pd.DataFrame([[e,TimeVal,COA_V,COA_X,COA_Y,COA_Z,TimeMin,TimeMax]],columns=['EventNum','CoaTime','COA_V','COA_X','COA_Y','COA_Z','MinTime','MaxTime'])
                IntEvents = IntEvents.append(dat,ignore_index=True)
                


            c=d+1
            e+=1


        # ----------- Determining the initial triggered events, not inspecting overlaps ------
        EventNum = np.ones((len(IntEvents)),dtype=int)
        d=1
        for ee in range(len(IntEvents)):

            if (ee+1 < len(IntEvents)) and ((IntEvents['MaxTime'].iloc[ee] - IntEvents['MinTime'].iloc[ee+1]).total_seconds() < 0):
                EventNum[ee] = d
                d+=1
            else:
                EventNum[ee] = d
        IntEvents['EventNum'] = EventNum




        d=0
        for ee in range(1,np.max(IntEvents['EventNum'])+1):
            tmp = IntEvents[IntEvents['EventNum'] == ee].reset_index(drop=True)
            if d==0:
                EVENTS = pd.DataFrame([[ee, tmp['CoaTime'].iloc[np.argmax(tmp['COA_V'])], np.max(tmp['COA_V']), tmp['COA_X'].iloc[np.argmax(tmp['COA_V'])], tmp['COA_X'].iloc[np.argmax(tmp['COA_V'])], tmp['COA_X'].iloc[np.argmax(tmp['COA_V'])],tmp['CoaTime'].iloc[np.argmax(tmp['COA_V'])] + timedelta(seconds=-self.MarginalWindow),tmp['CoaTime'].iloc[np.argmax(tmp['COA_V'])] + timedelta(seconds=self.MarginalWindow)]],columns=['EventNum','CoaTime','COA_V','COA_X','COA_Y','COA_Z','MinTime','MaxTime'])
                d+=1
            else:
                EVENTS = EVENTS.append(pd.DataFrame([[ee, tmp['CoaTime'].iloc[np.argmax(tmp['COA_V'])], np.max(tmp['COA_V']), tmp['COA_X'].iloc[np.argmax(tmp['COA_V'])], tmp['COA_X'].iloc[np.argmax(tmp['COA_V'])], tmp['COA_X'].iloc[np.argmax(tmp['COA_V'])],tmp['CoaTime'].iloc[np.argmax(tmp['COA_V'])] + timedelta(seconds=-self.MarginalWindow),tmp['CoaTime'].iloc[np.argmax(tmp['COA_V'])] + timedelta(seconds=self.MarginalWindow)]],columns=['EventNum','CoaTime','COA_V','COA_X','COA_Y','COA_Z','MinTime','MaxTime']),ignore_index=True)





        # Defining an event id based on maximum coalescence
        EVID = np.chararray(len(EVENTS),17)
        for e in range(len(EVENTS)):
            EVID[e] = str(re.sub(r"\D", "",EVENTS['CoaTime'].astype(str).iloc[e]))
        EVENTS['EventID'] = EVID




        return EVENTS


    def plot_scn(self,starttime,endtime,stations=None,savefile=None):
        '''


        '''

        # Defining the filename of the trace
        fname = path.join(self.output.path,self.output.name + '.scn')



        if path.exists(fname):

            # Loading the .scn file
            DATA = pd.read_csv(fname,names=['DT','COA','X','Y','Z'])
            DATA['DT'] = pd.to_datetime(DATA['DT'])

            if stations == None:
                # Plotting the .scn file

                fig = plt.figure(figsize=(30,15))
                fig.patch.set_facecolor('white')
                plt.plot(DATA['DT'],DATA['COA'],color='blue')
                plt.xlabel('Datetime')
                plt.ylabel('Maximum Coalescence')
                plt.axhline(self.DetectionThreshold,color='green')

                EVENTS = self._Trigger_scn(DATA,DATA['DT'].iloc[0].strftime('%Y-%m-%dT%H:%M:%S.%f'),DATA['DT'].iloc[-1].strftime('%Y-%m-%dT%H:%M:%S.%f'))

                for ee in range(len(EVENTS['MinTime'])):
                    plt.axvline(x=pd.to_datetime(EVENTS['MinTime'].iloc[ee]),linestyle='--',color='red')
                    plt.axvline(x=pd.to_datetime(EVENTS['MaxTime'].iloc[ee]),linestyle='--',color='red')
                    plt.axvline(x=pd.to_datetime(EVENTS['CoaTime'].iloc[ee]),color='red')

                plt.xlim([pd.to_datetime(starttime),pd.to_datetime(endtime)])


            else:
                # Plotting the .scn file with the addition of station avaliability
                fig = plt.figure(figsize=(30,15))
                fig.patch.set_facecolor('white')
                plt.plot(DATA['DT'],DATA['COA'],color='blue')
                plt.xlabel('Datetime')
                plt.ylabel('Maximum Coalescence')
                plt.axhline(self.DetectionThreshold,color='green')

                EVENTS = self._Trigger_scn(DATA,DATA['DT'].iloc[0].strftime('%Y-%m-%dT%H:%M:%S.%f'),DATA['DT'].iloc[-1].strftime('%Y-%m-%dT%H:%M:%S.%f'))

                for ee in range(len(EVENTS['MinTime'])):
                    plt.axvline(x=pd.to_datetime(EVENTS['MinTime'].iloc[ee]),linestyle='--',color='red')
                    plt.axvline(x=pd.to_datetime(EVENTS['MaxTime'].iloc[ee]),linestyle='--',color='red')
                    plt.axvline(x=pd.to_datetime(EVENTS['CoaTime'].iloc[ee]),color='red')

                plt.xlim([pd.to_datetime(starttime),pd.to_datetime(endtime)])





            # Saving figure if defined
            if savefile == None:
                plt.show()
            else:
                plt.savefig(savefile)

        else:
            print('Please run scn.Detect to generate a .scn file !')




    def Detect(self,starttime,endtime):
        ''' 
           Function 

           Detection of the  

        '''
        # Conduct the continious compute on the decimated grid
        self.lookup_table = self.lookup_table.decimate(self.Decimate)
        
        # Define pre-pad as a function of the onset windows
        if self.pre_pad is None:
            self.pre_pad = max(self.onset_win_p1[1],self.onset_win_s1[1]) + 3*max(self.onset_win_p1[0],self.onset_win_s1[0])
        
        # Dectect the possible events from the decimated grid
        self._continious_compute(starttime,endtime)


    def _Gaussian_Coalescence(self):
        '''
            Function to fit a gaussian for the coalescence function.
        '''


        SNR_P = self.DATA.SNR_P
        SNR_S = self.DATA.SNR_S
        X     = np.arange(SNR_P.shape[1])

        GAU_THRESHOLD = 1.4

        #---- Selecting only the data above a predefined threshold ----
        # Setting values below threshold to nan
        SNR_P[np.where(SNR_P < GAU_THRESHOLD)] = np.nan
        SNR_S[np.where(SNR_S < GAU_THRESHOLD)] = np.nan




        # Defining two blank arrays that gaussian periods should be defined for
        SNR_P_GauNum = np.zeros(SNR_P.shape)*np.nan
        SNR_S_GauNum = np.zeros(SNR_S.shape)*np.nan


        # --- Determing the indexs to fit gaussians about ---

        for s in range(len(SNR_P)):
            c=0
            e=1

            ValInd = np.where(~np.isnan(SNR_P[s,:]))[0]
            while c < len(ValInd):

                # Determining the index when above the level and maximum value
                d=c
                while ValInd[d]+1 == ValInd[d+1]:
                    d+=1
                    if d+1 >= len(ValInd)-1:
                        d=len(ValInd)-1
                        break


                indmin = c
                indmax = d  

                SNR_P_GauNum[s,ValInd[c]:ValInd[d]] = e  


                c=d+1
                e+=1

        self.DATA.SNR_P_GauNum = SNR_P_GauNum

        for s in range(len(SNR_S)):
            c=0
            e=1
            ValInd = np.where(~np.isnan(SNR_S[s,:]))[0]
            while c < len(ValInd):

                # Determining the index when above the level and maximum value
                d=c
                while ValInd[d]+1 == ValInd[d+1]:
                    d+=1
                    if d+1 >= len(ValInd)-1:
                        d=len(ValInd)-1
                        break


                indmin = c
                indmax = d  

                SNR_S_GauNum[s,ValInd[c]:ValInd[d]] = e  


                c=d+1
                e+=1



        self.DATA.SNR_S_GauNum = SNR_S_GauNum

        # --- Determing the indexs to fit gaussians about ---
        
        SNR_PGAU = np.zeros(SNR_P.shape)
        for s in range(SNR_P.shape[0]): 
            if ~np.isnan(np.nanmax(SNR_P_GauNum[s,:])):
                c=0
                for ee in range(1,int(round(np.nanmax(SNR_P_GauNum[s,:])))):


                    XSig = X[np.where((SNR_P_GauNum[s,:] == ee))[0]] 
                    YSig = SNR_P[s,np.where((SNR_P_GauNum[s,:] == ee))[0]]

                    #print(' LEN = {} and CUT_LEN = {}'.format(len(YSig),round(float(self.bp_filter_p1[0])*self.sample_rate)/10))

                    if len(YSig) > 8:

                        #self.DATA.SNR_P =  YSig

                        try:
                            lowfreq=float(self.bp_filter_p1[0])
                            p0 = [np.max(YSig), np.argmax(YSig) + np.min(XSig), 1./(lowfreq/4.)]

                            # Fitting the gaussian to the function

                            #print(XSig)
                            #print(YSig)
                            #print(p0)
                            popt, pcov = curve_fit(gaussian_func, XSig, YSig, p0) # Fit gaussian to data
                            tmp_PGau = gaussian_func(XSig.astype(float),float(popt[0]),float(popt[1]),float(popt[2]))
                            #print(tmp_PGau)

                            if c == 0:
                                SNR_P_GAU = np.zeros(X.shape)
                                SNR_P_GAU[np.where((SNR_P_GauNum[s,:] == ee))[0]] = tmp_PGau
                                c+=1
                            else:
                                SNR_P_GAU[np.where((SNR_P_GauNum[s,:] == ee))[0]] = tmp_PGau
                        except:
                            print('Error with {}'.format(ee))

                    else:
                        continue

                SNR_PGAU[s,:] =  SNR_P_GAU

        self.DATA.SNR_P = SNR_PGAU



        # --- Determing the indexs to fit gaussians about ---
        
        SNR_SGAU = np.zeros(SNR_S.shape)
        for s in range(SNR_S.shape[0]): 
            if ~np.isnan(np.nanmax(SNR_S_GauNum[s,:])):
                c=0
                for ee in range(1,int(round(np.nanmax(SNR_S_GauNum[s,:])))):


                    XSig = X[np.where((SNR_S_GauNum[s,:] == ee))[0]] 
                    YSig = SNR_S[s,np.where((SNR_S_GauNum[s,:] == ee))[0]]

                    print(' LEN = {} and CUT_LEN = {}'.format(len(YSig),round(float(self.bp_filter_p1[0])*self.sample_rate)/10))

                    if len(YSig) > 8:

                        #self.DATA.SNR_P =  YSig

                        try:
                            lowfreq=float(self.bp_filter_p1[0])
                            p0 = [np.max(YSig), np.argmax(YSig) + np.min(XSig), 1./(lowfreq/4.)]

                            # Fitting the gaussian to the function

                            print(XSig)
                            print(YSig)
                            print(p0)
                            popt, pcov = curve_fit(gaussian_func, XSig, YSig, p0) # Fit gaussian to data
                            tmp_SGau = gaussian_func(XSig.astype(float),float(popt[0]),float(popt[1]),float(popt[2]))
                            print(tmp_SGau)

                            if c == 0:
                                SNR_S_GAU = np.zeros(X.shape)
                                SNR_S_GAU[np.where((SNR_S_GauNum[s,:] == ee))[0]] = tmp_SGau
                                c+=1
                            else:
                                SNR_S_GAU[np.where((SNR_S_GauNum[s,:] == ee))[0]] = tmp_SGau
                        except:
                            print('Error with {}'.format(ee))

                    else:
                        continue

                SNR_SGAU[s,:] =  SNR_S_GAU

        self.DATA.SNR_S = SNR_PGAU


    def _GaussianTrigger(self,SNR,PHASE,cstart,eventTP,eventTS,Name):
        '''
            Function to fit gaussian to onset function, based on knowledge of approximate trigger index, 
            lowest freq within signal and signal sampling rate. Will fit gaussian and return standard 
            deviation of gaussian, representative of timing error.
    
        '''

        #print('Fitting Gaussian for {} -  {} -  {}'.format(PHASE,cstart,eventT))

        sampling_rate = self.sample_rate

        # Determining the triggering X location based on the SNR value
        trig_idx_P = int(((eventTP-cstart).seconds + (eventTP-cstart).microseconds/10.**6) *sampling_rate)
        trig_idx_S = int(((eventTS-cstart).seconds + (eventTS-cstart).microseconds/10.**6) *sampling_rate)



        P_idxmin = int(trig_idx_P - (trig_idx_S-trig_idx_P)/2)
        P_idxmax = int(trig_idx_P + (trig_idx_S-trig_idx_P)/2)
        S_idxmin = int(trig_idx_S - (trig_idx_S-trig_idx_P)/2)
        S_idxmax = int(trig_idx_S + (trig_idx_S-trig_idx_P)/2)
        for ii in [P_idxmin,P_idxmax,S_idxmin,S_idxmax]:
            if ii < 0:
                ii = 0
            if ii > len(SNR):
                ii = len(SNR)


        #print(' Pmin = {} , Pmax = {}'.format(P_idxmin,P_idxmax))
        #print(' Smin = {} , Smax = {}'.format(S_idxmin,S_idxmax))
    
        Pidx = np.argmax(SNR[P_idxmin:P_idxmax]) + P_idxmin
        Sidx = np.argmax(SNR[S_idxmin:S_idxmax]) + S_idxmin
        #print(Pidx,Sidx)



        if PHASE == 'P':
            lowfreq = self.bp_filter_p1[0]
            trig_idx = Pidx
        if PHASE == 'S':
            lowfreq = self.bp_filter_s1[0]
            trig_idx = Sidx

        data_half_range = int(1.25*sampling_rate/(lowfreq)) # half range number of indices to fit guassian over (for 1 wavelengths of lowest frequency component)
        x_data = np.arange(trig_idx-data_half_range, trig_idx+data_half_range,dtype=float)/sampling_rate # x data, in seconds

        y_data = SNR[int(trig_idx-data_half_range):int(trig_idx+data_half_range)] # +/- one wavelength of lowest frequency around trigger
        p0 = [np.amax(SNR), float(trig_idx)/sampling_rate, 1./(lowfreq/4.)] # Initial guess (should work for any sampling rate and frequency)

        d = 0
        for jj in range(len(x_data)):
            if d == 0:
                XDATA = cstart + timedelta(seconds=x_data[jj])
                d+=1
            else:
                XDATA = np.hstack((XDATA,(cstart + timedelta(seconds=x_data[jj]))))











        
        try:
            popt, pcov = curve_fit(gaussian_func, x_data, y_data, p0) # Fit gaussian to data
            sigma = np.absolute(popt[2]) # Get standard deviation from gaussian fit

            # Mean is popt[1]. x_data[0] + popt[1] (In seconds)
            mean = cstart + timedelta(seconds=float(popt[1]))

            maxSNR = popt[0]


            # Determining if the pick is above 
            n, bins = np.histogram(SNR,bins=np.arange(0,np.max(SNR),7/5000))
            mids = 0.5*(bins[1:] + bins[:-1])
            ncum = np.cumsum(n)/np.sum(n)
            #print(ncum)

            #print(np.where((mids-popt[0]) >= 0)[0])

            if (len(np.where((mids-popt[0]) >= 0)[0]) == 0):
                #print('Picked {} for {}'.format(PHASE,Name))
                GAU_FITS = {}
                GAU_FITS['popt'] = popt
                GAU_FITS['xdata'] = x_data
                GAU_FITS['xdata_dt'] = XDATA
                GAU_FITS['PickValue'] = 1.0 

            else:
                if (np.min(ncum[np.where((mids-popt[0]) >= 0)[0]]) >= self.PickThreshold):
                    #print('Picked 2 {} for {} - {}'.format(PHASE,Name,np.min(ncum[np.where((mids-popt[0]) >= 0)[0]])))
                    GAU_FITS = {}
                    GAU_FITS['popt'] = popt
                    GAU_FITS['xdata'] = x_data
                    GAU_FITS['xdata_dt'] = XDATA
                    GAU_FITS['PickValue'] = np.min(ncum[np.where((mids-popt[0]) >= 0)[0]])
                else:
                    #print('Didnt Pick {} for {} - {}'.format(PHASE,Name,np.min(ncum[np.where((mids-popt[0]) >= 0)[0]])))
                    GAU_FITS = {}
                    GAU_FITS['popt'] = np.zeros((3))
                    GAU_FITS['xdata'] = np.zeros(x_data.shape)
                    GAU_FITS['xdata_dt'] = np.zeros(XDATA.shape)
                    GAU_FITS['PickValue'] = np.min(ncum[np.where((mids-popt[0]) >= 0)[0]])
                    sigma = -1
                    mean  = -1
                    maxSNR = -1 


            #print(mean)
        except:
            GAU_FITS = {}
            GAU_FITS['popt'] = np.zeros((3))
            GAU_FITS['xdata'] = np.zeros(x_data.shape)
            GAU_FITS['xdata_dt'] = np.zeros(XDATA.shape)
            GAU_FITS['PickValue'] = -1


            sigma = -1
            mean  = -1
            maxSNR = -1

        return GAU_FITS,maxSNR,sigma,mean





    def _ArrivalTrigger(self,EVENT_MaxCoa,EventName):
        '''
            FUNCTION - _ArrivalTrigger - Used to determine earthquake station arrival time

        '''

        SNR_P = self.DATA.SNR_P
        SNR_S = self.DATA.SNR_S

        #print(EVENT_MaxCoa[['X','Y','Z']].iloc[0])

        ttp = self.lookup_table.value_at('TIME_P', np.array(self.lookup_table.coord2xyz(np.array([EVENT_MaxCoa[['X','Y','Z']].values]))).astype(int))[0]
        tts = self.lookup_table.value_at('TIME_S', np.array(self.lookup_table.coord2xyz(np.array([EVENT_MaxCoa[['X','Y','Z']].values]))).astype(int))[0]
        

        # Determining the stations that can be picked on and the phasese
        STATION_pickS=pd.DataFrame(columns=['Name','Phase','ModelledTime','PickTime','PickError'])
        c=0
        d=0
        for s in range(len(SNR_P)):
            stationEventPT = EVENT_MaxCoa['DT'] + timedelta(seconds=ttp[s])
            stationEventST = EVENT_MaxCoa['DT'] + timedelta(seconds=tts[s])

            if self.PickingType == 'Gaussian':
                GauInfoP,maxSNR_P,Err,Mn = self._GaussianTrigger(SNR_P[s],'P',self.DATA.startTime,stationEventPT.to_pydatetime(),stationEventST.to_pydatetime(),self.lookup_table.station_data['Name'][s])

            if c==0:
                GAUP = GauInfoP
                c+=1
            else:
                GAUP = np.hstack((GAUP,GauInfoP))
            
            tmpSTATION_pick = pd.DataFrame([[self.lookup_table.station_data['Name'][s],'P',stationEventPT,Mn,Err,maxSNR_P]],columns=['Name','Phase','ModelledTime','PickTime','PickError','PickSNR'])
            STATION_pickS = STATION_pickS.append(tmpSTATION_pick)


            if self.PickingType == 'Gaussian':
                GauInfoS,maxSNR_S,Err,Mn = self._GaussianTrigger(SNR_S[s],'S',self.DATA.startTime,stationEventPT.to_pydatetime(),stationEventST.to_pydatetime(),self.lookup_table.station_data['Name'][s])


            if d==0:
                GAUS = GauInfoS
                d+=1
            else:
                GAUS = np.hstack((GAUS,GauInfoS))

            tmpSTATION_pick = pd.DataFrame([[self.lookup_table.station_data['Name'][s],'S',stationEventST,Mn,Err,maxSNR_S]],columns=['Name','Phase','ModelledTime','PickTime','PickError','PickSNR'])
            STATION_pickS = STATION_pickS.append(tmpSTATION_pick)

        #print(STATION_pickS)
        # Saving the output from the triggered events
        STATION_pickS = STATION_pickS[['Name','Phase','ModelledTime','PickTime','PickError']]
        self.output.write_stationsfile(STATION_pickS,EventName)

        return STATION_pickS,GAUP,GAUS

    def _ErrorEllipse(self,COA3D):
        """
        Function to calculate covariance matrix and expectation hypocentre from coalescence array.
        Inputs: 
            coal_array - 3D array of coalescence values for a particular time (in x,y,z dimensions); 
            x,y,z_indices_m - 1D arrays containing labels of the indices of the coalescence grid in metres.
        Outputs are: expect_vector - x,y,z coordinates of expectation hypocentre in m; cov_matrix - Covariance matrix (of format: xx,xy,xz;yx,yy,yz;zx,zy,zz).


        """
        # samples_vectors = np.zeros((np.product(np.shape(coal_array)), 3), dtype=float)
        #samples_weights = np.zeros(np.product(np.shape(COA3D)), dtype=float)

        # Get point sample coords and weights:
        samples_weights = COA3D.flatten()

        lc = self.lookup_table.cell_count
        ly, lx, lz = np.meshgrid(np.arange(lc[1]), np.arange(lc[0]), np.arange(lc[2]))
        x_samples      = lx.flatten()*self.lookup_table.cell_size[0]
        y_samples      = ly.flatten()*self.lookup_table.cell_size[1]
        z_samples      = lz.flatten()*self.lookup_table.cell_size[2]

        SumSW = np.sum(samples_weights)

        # Calculate expectation values:
        x_expect = np.sum(samples_weights*x_samples)/SumSW
        y_expect = np.sum(samples_weights*y_samples)/SumSW
        z_expect = np.sum(samples_weights*z_samples)/SumSW



        # And calculate covariance matrix:
        cov_matrix = np.zeros((3,3))
        cov_matrix[0,0] = np.sum(samples_weights*(x_samples-x_expect)*(x_samples-x_expect))/SumSW
        cov_matrix[1,1] = np.sum(samples_weights*(y_samples-y_expect)*(y_samples-y_expect))/SumSW
        cov_matrix[2,2] = np.sum(samples_weights*(z_samples-z_expect)*(z_samples-z_expect))/SumSW
        cov_matrix[0,1] = np.sum(samples_weights*(x_samples-x_expect)*(y_samples-y_expect))/SumSW
        cov_matrix[1,0] = cov_matrix[0,1]
        cov_matrix[0,2] = np.sum(samples_weights*(x_samples-x_expect)*(z_samples-z_expect))/SumSW
        cov_matrix[2,0] = cov_matrix[0,2]
        cov_matrix[1,2] = np.sum(samples_weights*(y_samples-y_expect)*(z_samples-z_expect))/SumSW
        cov_matrix[2,1] = cov_matrix[1,2]



        # Determining the maximum location, and taking 2xgrid cells possitive and negative for location in each dimension
        CoaMap_max =  np.where(COA3D == np.max(COA3D))
        xmin = CoaMap_max[0][0]-2
        xmax = CoaMap_max[0][0]+2
        ymin = CoaMap_max[1][0]-2
        ymax = CoaMap_max[1][0]+2
        zmin = CoaMap_max[2][0]-2
        zmax = CoaMap_max[2][0]+2

        x_data = np.arange(1,6) - 3.0
        try:
            y_data = COA3D[xmin:xmax+1,ymin:ymax+1,zmin:zmax+1][:,2,2]
            p0 = [np.max(y_data), 0, 1] # Initial guess (should work for any sampling rate and frequency)
            popt, pcov = curve_fit(gaussian_func, x_data, y_data, p0) # Fit gaussian to data
            xloca = popt[1]
        except:
            xloca = 0


        try:
            y_data = COA3D[xmin:xmax+1,ymin:ymax+1,zmin:zmax+1][2,:,2]
            p0 = [np.max(y_data), 0, 1] # Initial guess (should work for any sampling rate and frequency)
            popt, pcov = curve_fit(gaussian_func, x_data, y_data, p0) # Fit gaussian to data
            yloca = popt[1]
        except:
            yloca = 0


        try:
            y_data = COA3D[xmin:xmax+1,ymin:ymax+1,zmin:zmax+1][2,2,:]
            p0 = [np.max(y_data), 0, 1] # Initial guess (should work for any sampling rate and frequency)
            popt, pcov = curve_fit(gaussian_func, x_data, y_data, p0) # Fit gaussian to data
            zloca = popt[1]
        except:
            zloca = 0



        # Converting the grid location to X,Y,Z
        expect_vector = self.lookup_table.xyz2coord(self.lookup_table.loc2xyz(np.array([[xloca + CoaMap_max[0][0],yloca + CoaMap_max[1][0],zloca + CoaMap_max[2][0]]])))[0]

        return expect_vector, cov_matrix

    def _LocationError(self,Map4D):

        '''
            Function

        '''

        # Determining the coalescence 3D map
        CoaMap = np.log(np.sum(np.exp(Map4D),axis=-1))
        CoaMap = CoaMap/np.max(CoaMap)

        CoaMap_Cutoff = 0.88
        CoaMap[CoaMap < CoaMap_Cutoff] = CoaMap_Cutoff 
        CoaMap = CoaMap - CoaMap_Cutoff 
        CoaMap = CoaMap/np.max(CoaMap)
        self.CoaMAP = CoaMap


        # Determining the location error as a error-ellipse
        LOC,ErrCOV = self._ErrorEllipse(CoaMap)
        LOC_ERR =  np.array([np.sqrt(ErrCOV[0,0]), np.sqrt(ErrCOV[1,1]), np.sqrt(ErrCOV[2,2])])



        # Determining maximum location and error about this point
        # ErrorVolume = np.zeros((CoaMap.shape))
        # ErrorVolume[np.where(CoaMap > self.LocationError)] = 1
        # MaxX = np.sum(np.max(np.sum(ErrorVolume,axis=0),axis=1),axis=0)*self.lookup_table.cell_size[0]
        # MaxY = np.sum(np.max(np.sum(ErrorVolume,axis=1),axis=1),axis=0)*self.lookup_table.cell_size[1]
        # MaxZ = np.sum(np.max(np.sum(ErrorVolume,axis=2),axis=1),axis=0)*self.lookup_table.cell_size[2]

        return LOC,LOC_ERR



    def Trigger(self,starttime,endtime):
        '''
        

        '''

        # Intial Detection of the events from .scn file
        CoaVal = self.output.read_scan()
        EVENTS = self._Trigger_scn(CoaVal,starttime,endtime)
        
        # Conduct the continious compute on the decimated grid
        self.lookup_table =  self.lookup_table.decimate(self.Decimate)

        # Define pre-pad as a function of the onset windows
        if self.pre_pad is None:
            self.pre_pad = max(self.onset_win_p1[1],self.onset_win_s1[1]) + 3*max(self.onset_win_p1[0],self.onset_win_s1[0])
        
        #
        Triggered = pd.DataFrame(columns=['DT','COA','X','Y','Z','ErrX','ErrY','ErrZ'])
        for e in range(len(EVENTS)):

            print('--Processing for Event {} of {} - {}'.format(e+1,len(EVENTS),(EVENTS['EventID'].iloc[e]).astype(str)))
            tic()
            # Determining the Seismic event location
            cstart = EVENTS['MinTime'].iloc[e] + timedelta(seconds=-self.pre_pad) 
            cend   = EVENTS['MaxTime'].iloc[e] + timedelta(seconds=self.post_pad)
            self.DATA.read_mseed(cstart.strftime('%Y-%m-%dT%H:%M:%S.%f'),cend.strftime('%Y-%m-%dT%H:%M:%S.%f'),self.sample_rate)

            daten, dsnr, dloc, self.MAP = self._compute(cstart,cend,self.DATA.signal,self.DATA.station_avaliability)

            dcoord = self.lookup_table.xyz2coord(np.array(dloc).astype(int))
            EventCoaVal = pd.DataFrame(np.array((daten,dsnr,dcoord[:,0],dcoord[:,1],dcoord[:,2])).transpose(),columns=['DT','COA','X','Y','Z'])
            EventCoaVal['DT'] = pd.to_datetime(EventCoaVal['DT'])
            self.EVENT = EventCoaVal
            self.EVENT_max = self.EVENT.iloc[EventCoaVal['COA'].astype('float').idxmax()]

            # Determining the hypocentral location from the maximum over the marginal window.
            Picks,GAUP,GAUS = self._ArrivalTrigger(self.EVENT_max,(EVENTS['EventID'].iloc[e].astype(str)))

            StationPick = {}
            StationPick['Pick'] = Picks
            StationPick['GAU_P'] = GAUP
            StationPick['GAU_S'] = GAUS
            toc()

            # Determining earthquake location error
            tic()
            LOC,LOC_ERR = self._LocationError(self.MAP)
            toc()


            EV = pd.DataFrame([np.append(self.EVENT_max.as_matrix(),[LOC[0],LOC[1],LOC[2],LOC_ERR[0],LOC_ERR[1],LOC_ERR[2]])],columns=['DT','COA','X','Y','Z','X_ErrE','Y_ErrE','Z_ErrE','ErrX','ErrY','ErrZ'])
            self.output.write_event(EV,str(EVENTS['EventID'].iloc[e].astype(str)))
            if self.CutMSEED == True:
                print('Creating cut Mini-SEED')
                tic()
                self.output.cut_mseed(self.DATA,str(EVENTS['EventID'].iloc[e].astype(str)))
                toc()

            # Outputting coalescence grids and triggered events
            if self.CoalescenceTrace == True:
                tic()
                print('Creating Station Traces')
                SeisPLT = SeisPlot(self.lookup_table,self.MAP,self.CoaMAP,self.DATA,self.EVENT,StationPick)
                SeisPLT.CoalescenceTrace(SaveFilename='{}_{}'.format(path.join(self.output.path, self.output.name),EVENTS['EventID'].iloc[e].astype(str)))
                toc()

            if self.CoalescenceGrid == True:
                tic()
                print('Creating 4D Coalescence Grids')
                self.output.write_coal4D(self.MAP,EVENTS['EventID'].iloc[e].astype(str),cstart,cend)
                toc()

            if self.CoalescenceVideo == True:
                tic()
                print('Creating Seismic Videos')
                SeisPLT = SeisPlot(self.lookup_table,self.MAP,self.CoaMAP,self.DATA,self.EVENT,StationPick)
                SeisPLT.CoalescenceVideo(SaveFilename='{}_{}'.format(path.join(self.output.path, self.output.name),EVENTS['EventID'].iloc[e].astype(str)))
                toc()

            if self.CoalescencePicture == True:
                tic()
                print('Creating Seismic Picture')
                SeisPLT = SeisPlot(self.lookup_table,self.MAP,self.CoaMAP,self.DATA,self.EVENT,StationPick)
                SeisPLT.CoalescenceMarginal(SaveFilename='{}_{}'.format(path.join(self.output.path, self.output.name),EVENTS['EventID'].iloc[e].astype(str)))
                toc()

            self.MAP    = None
            self.CoaMAP = None
            self.EVENT  = None
            self.cstart = None
            self.cend   = None

