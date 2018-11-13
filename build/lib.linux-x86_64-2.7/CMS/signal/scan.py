################################################################################################



# ---- Import Packages -----

import numpy as np
from CMS.core.time import UTCDateTime
from datetime import datetime
from datetime import timedelta

from obspy import read,Stream,Trace
from obspy.core import UTCDateTime

from obspy.signal.trigger import classic_sta_lta
from scipy.signal import butter, lfilter
from scipy.optimize import curve_fit

import CMS.core.cmslib as ilib

import obspy
import re

import os
import os.path as path
import pickle

import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import matplotlib.dates as mdates
import matplotlib.image as mpimg
import matplotlib.animation as animation


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
    for ch in range(0, nchan):
        snr[ch, :] = classic_sta_lta(sig[ch, :], stw, ltw)
        np.clip(1+snr[ch,:],0.8,np.inf,snr[ch, :])
        np.log(snr[ch, :], snr[ch, :])
    return snr


def filter(sig,srate,lc,hc,order=3):
    '''


    '''
    b1, a1 = butter(order, [2.0*lc/srate, 2.0*hc/srate], btype='band')
    nchan, nsamp = sig.shape
    fsig = np.copy(sig)
    #sig = detrend(sig)
    for ch in range(0, nchan):
        fsig[ch,:] = fsig[ch,:] - fsig[ch,0]
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


    def write_event(self, EVENT):
        fname = path.join(self.path,self.name + '_Event.txt')
        EVENT.to_csv(fname,index=False)

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


    def write_coalVideo(self,MAP,lookup_table,DATA,EventCoaVal,EventName):
        '''
            Writing the coalescence video to file for each event
        '''

        filename = path.join(self.path,self.name)

        SeisPLT = SeisPlot(MAP,lookup_table,DATA,EventCoaVal)
        SeisPLT.CoalescenceVideo(SaveFilename='{}_{}'.format(filename,EventName))
        SeisPLT.CoalescenceMarginal(SaveFilename='{}_{}'.format(filename,EventName))

    def write_stationsfile(self,STATIONS,EventName):
        '''
            Writing the stations file
        '''
        fname = path.join(self.path,self.name + '_{}.stn'.format(EventName))
        STATIONS.to_csv(fname,index=False)

    def write_event(self,EVENT,EventName):
        '''
            Writing the stations file
        '''
        fname = path.join(self.path,self.name + '_{}.event'.format(EventName))
        EVENT.to_csv(fname,index=False)


class SeisPlot:
    '''
         Seismic plotting for CMS ouptuts. Functions include:
            CoalescenceVideo - A script used to generate a coalescence 
            video over the period of earthquake location

            CoalescenceLocation - Location plot 

            CoalescenceMarginalizeLocation - 

    '''
    def __init__(self,lut,MAP,DATA,EVENT):
        '''
            This is the initial variatiables
        '''
        self.LUT   = lut
        self.DATA  = DATA
        self.EVENT = EVENT
        self.MAP   = MAP

        self.TraceScaling = 1
        self.CMAP  = 'jet'
        self.Plot_Stations=True
        self.FilteredSignal=True



        self.times = np.arange(self.DATA.startTime,self.DATA.endTime,timedelta(seconds=1/self.DATA.sampling_rate))
        self.EVENT = self.EVENT[(self.EVENT['DT'] > self.times[0]) & (self.EVENT['DT'] < self.times[-1])]



        self.logoPath = '{}/CMS.png'.format('/'.join(ilib.__file__.split('/')[:-2]))

        self.MAPmax   = np.max(MAP)


        self.CoaTraceVLINE = None
        self.CoaValVLINE   = None

        # Updating the Coalescence Maps
        self.CoaXYPlt = None
        self.CoaYZPlt = None
        self.CoaXZPlt = None
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
        indexVal = np.array(self.LUT.xyz2loc(self.LUT.coord2xyz(np.array([[self.EVENT['X'].iloc[index],self.EVENT['Y'].iloc[index],self.EVENT['Z'].iloc[index]]]))[0])).astype(int)
        indexCoord = np.array([[self.EVENT['X'].iloc[index],self.EVENT['Y'].iloc[index],self.EVENT['Z'].iloc[index]]])[0,:]


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

        for ii in range(self.DATA.signal.shape[1]): 
            if self.FilteredSignal == False:
                    Coa_Trace.plot(np.arange(STIn,ENIn),(self.DATA.signal[0,ii,STIn:ENIn]/np.max(abs(self.DATA.signal[0,ii,STIn:ENIn])))*self.TraceScaling+(ii+1),'r',linewidth=0.5)
                    Coa_Trace.plot(np.arange(STIn,ENIn),(self.DATA.signal[1,ii,STIn:ENIn]/np.max(abs(self.DATA.signal[1,ii,STIn:ENIn])))*self.TraceScaling+(ii+1),'b',linewidth=0.5)
                    Coa_Trace.plot(np.arange(STIn,ENIn),(self.DATA.signal[2,ii,STIn:ENIn]/np.max(abs(self.DATA.signal[2,ii,STIn:ENIn])))*self.TraceScaling+(ii+1),'g',linewidth=0.5)
            else:
                    Coa_Trace.plot(np.arange(STIn,ENIn),(self.DATA.FilteredSignal[0,ii,STIn:ENIn]/np.max(abs(self.DATA.FilteredSignal[0,ii,STIn:ENIn])))*self.TraceScaling+(ii+1),'r',linewidth=0.5)
                    Coa_Trace.plot(np.arange(STIn,ENIn),(self.DATA.FilteredSignal[1,ii,STIn:ENIn]/np.max(abs(self.DATA.FilteredSignal[1,ii,STIn:ENIn])))*self.TraceScaling+(ii+1),'b',linewidth=0.5)
                    Coa_Trace.plot(np.arange(STIn,ENIn),(self.DATA.FilteredSignal[2,ii,STIn:ENIn]/np.max(abs(self.DATA.FilteredSignal[2,ii,STIn:ENIn])))*self.TraceScaling+(ii+1),'g',linewidth=0.5)

        # ---------------- Plotting the Station Travel Times -----------
        for i in range(self.LUT.get_value_at('TIME_P',np.array([indexVal]))[0].shape[0]):
            tp = np.argmin(abs((self.times.astype(datetime) - (TimeSlice.astype(datetime) + timedelta(seconds=self.LUT.get_value_at('TIME_P',np.array([indexVal]))[0][i])))/timedelta(seconds=1)))
            ts = np.argmin(abs((self.times.astype(datetime) - (TimeSlice.astype(datetime) + timedelta(seconds=self.LUT.get_value_at('TIME_S',np.array([indexVal]))[0][i])))/timedelta(seconds=1)))

            if i == 0:
                TP = tp
                TS = ts
            else:
                TP = np.append(TP,tp)
                TS = np.append(TS,ts)


        self.CoaArriavalTP = Coa_Trace.scatter(TP,(np.arange(self.DATA.signal.shape[1])+1),15,'r',marker='v')
        self.CoaArriavalTS = Coa_Trace.scatter(TS,(np.arange(self.DATA.signal.shape[1])+1),15,'b',marker='v')

        Coa_Trace.set_ylim([0,ii+2])
        Coa_Trace.set_xlim([STIn,ENIn])
        Coa_Trace.get_xaxis().set_ticks([])
        Coa_Trace.yaxis.tick_right()
        Coa_Trace.yaxis.set_ticks(np.arange(self.DATA.signal.shape[1])+1)
        Coa_Trace.yaxis.set_ticklabels(self.DATA.StationInformation['Name'])
        self.CoaTraceVLINE = Coa_Trace.axvline(TimeSlice,0,1000,linestyle='--',linewidth=2,color='r')


        # ------------- Plotting the Coalescence Function ----------- 
        Coa_CoaVal.plot(self.EVENT['DT'],self.EVENT['COA'])
        Coa_CoaVal.set_ylabel('Coalescence Value')
        Coa_CoaVal.set_xlabel('Date-Time')
        Coa_CoaVal.yaxis.tick_right()
        Coa_CoaVal.yaxis.set_label_position("right")
        Coa_CoaVal.set_xlim([self.times[STIn],self.times[ENIn]])
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


        #  ------------- Plotting the station Locations -----------
        Coa_Logo.axis('off')
        im = mpimg.imread(self.logoPath)
        Coa_Logo.imshow(im)
        Coa_Logo.text(150, 200, r'CoalescenceVideo', fontsize=14,style='italic')

        return fig


    def _CoalescenceVideo_update(self,frame):
        STIn = np.where(self.times == self.EVENT['DT'].iloc[0])[0][0]
        TimeSlice  = self.times[int(frame)]
        index      = np.where(self.EVENT['DT'] == TimeSlice)[0][0]
        indexVal = np.array(self.LUT.xyz2loc(self.LUT.coord2xyz(np.array([[self.EVENT['X'].iloc[index],self.EVENT['Y'].iloc[index],self.EVENT['Z'].iloc[index]]]))[0])).astype(int)
        indexCoord = np.array([[self.EVENT['X'].iloc[index],self.EVENT['Y'].iloc[index],self.EVENT['Z'].iloc[index]]])[0,:]

        # Updating the Coalescence Value and Trace Lines
        self.CoaTraceVLINE.set_xdata(TimeSlice)
        self.CoaValVLINE.set_xdata(TimeSlice)

        # Updating the Coalescence Maps
        self.CoaXYPlt.set_array((self.MAP[:,:,int(indexVal[2]),int(frame-STIn)]/self.MAPmax)[:-1,:-1].ravel())
        self.CoaYZPlt.set_array((self.MAP[:,int(indexVal[1]),:,int(frame-STIn)]/self.MAPmax)[:-1,:-1].ravel())
        self.CoaXZPlt.set_array((np.transpose(self.MAP[int(indexVal[0]),:,:,int(frame-STIn)])/self.MAPmax)[:-1,:-1].ravel())

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


    def CoalescenceVideo(self,SaveFilename=None):
        STIn = np.where(self.times == self.EVENT['DT'].iloc[0])[0][0]
        ENIn = np.where(self.times == self.EVENT['DT'].iloc[-1])[0][0]

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=4, metadata=dict(artist='Ulvetanna'), bitrate=1800)



        FIG = self.CoalescenceImage(STIn)
        ani = animation.FuncAnimation(FIG, self._CoalescenceVideo_update, frames=np.arange(STIn,ENIn,int(self.DATA.sampling_rate/20)),blit=False,repeat=False) 

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

        # Gaussian fit about the time period 

        fig = plt.figure(figsize=(15,15))
        fig.patch.set_facecolor('white')
        Coa_XYSlice  =  plt.subplot2grid((3, 3), (0, 0), colspan=2,rowspan=2)
        Coa_YZSlice  =  plt.subplot2grid((3, 3), (2, 0), colspan=2)
        Coa_XZSlice  =  plt.subplot2grid((3, 3), (0, 2), rowspan=2)
        Coa_Logo     =  plt.subplot2grid((3, 3), (2, 2))


        # Determining the maginal window value from the coalescence function
        mMAP = self.MAP
        mMAP = np.log(np.sum(np.exp(mMAP),axis=-1))
        mMAP = mMAP/np.max(mMAP)
        mMAP_Cutoff = np.percentile(mMAP,95)
        mMAP[mMAP < mMAP_Cutoff] = mMAP_Cutoff 
        mMAP = mMAP - mMAP_Cutoff 
        mMAP = mMAP/np.max(mMAP)
        indexVal = np.where(mMAP == np.max(mMAP))
        indexCoord = self.LUT.xyz2coord(self.LUT.loc2xyz(np.array([[indexVal[0][0],indexVal[1][0],indexVal[2][0]]])))

        # # Determining the location optimal location and error ellipse
        # samples_weights = mMAP.flatten()
        # lc = self.LUT.cell_count
        # ly, lx, lz = np.meshgrid(np.arange(lc[1]), np.arange(lc[0]), np.arange(lc[2]))
        # x_samples      = lx.flatten()*self.LUT.cell_size[0]
        # y_samples      = ly.flatten()*self.LUT.cell_size[1]
        # z_samples      = lz.flatten()*self.LUT.cell_size[2]
        # SumSW = np.sum(samples_weights)
        # x_expect = np.sum(samples_weights*x_samples)/SumSW
        # y_expect = np.sum(samples_weights*y_samples)/SumSW
        # z_expect = np.sum(samples_weights*z_samples)/SumSW
        # expect_vector = np.array([x_expect/self.LUT.cell_size[0], y_expect/self.LUT.cell_size[1], z_expect/self.LUT.cell_size[2]], dtype=float)
        # cov_matrix = np.zeros((3,3))
        # cov_matrix[0,0] = np.sum(samples_weights*(x_samples-x_expect)*(x_samples-x_expect))/SumSW
        # cov_matrix[1,1] = np.sum(samples_weights*(y_samples-y_expect)*(y_samples-y_expect))/SumSW
        # cov_matrix[2,2] = np.sum(samples_weights*(z_samples-z_expect)*(z_samples-z_expect))/SumSW
        # cov_matrix[0,1] = np.sum(samples_weights*(x_samples-x_expect)*(y_samples-y_expect))/SumSW
        # cov_matrix[1,0] = cov_matrix[0,1]
        # cov_matrix[0,2] = np.sum(samples_weights*(x_samples-x_expect)*(z_samples-z_expect))/SumSW
        # cov_matrix[2,0] = cov_matrix[0,2]
        # cov_matrix[1,2] = np.sum(samples_weights*(y_samples-y_expect)*(z_samples-z_expect))/SumSW
        # cov_matrix[2,1] = cov_matrix[1,2]
        # expect_vector = self.lookup_table.xyz2coord(self.LUT.loc2xyz(np.array([[expect_vector[0],expect_vector[1],expect_vector[2]]])))[0]


        # lambda_, v = np.linalg.eig(cov_matrix)
        # lambda_ = np.sqrt(lambda_)

        # Plotting the marginal window
        gridX,gridY = np.mgrid[min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]))/self.LUT.cell_count[0], min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]))/self.LUT.cell_count[1]]
        Coa_XYSlice.pcolormesh(gridX,gridY,mMAP[:,:,int(indexVal[2][0])],cmap=self.CMAP,linewidth=0)
        CS = Coa_XYSlice.contour(gridX,gridY,mMAP[:,:,int(indexVal[2][0])],levels=[0.65,0.75,0.95],colors=('g','m','k'))
        Coa_XYSlice.clabel(CS, inline=1, fontsize=10)
        Coa_XYSlice.set_xlim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0])])
        Coa_XYSlice.set_ylim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1])])
        #Coa_XYSlice_ell = Ellipse(xy=(expect_vector[0], expect_vector[1]),width=lambda_[0]*2, height=lambda_[1]*2,angle=np.rad2deg(np.arccos(v[0, 0])))
        #Coa_XYSlice_ell.set_facecolor('none')
        #Coa_XYSlice_ell.set_edgecolor('r')
        #Coa_XYSlice.add_artist(Coa_XYSlice_ell)
        #Coa_XYSlice.scatter(expect_vector[0], expect_vector[1], c="r", marker="x")
        Coa_XYSlice.axvline(x=indexCoord[0][0],linestyle='--',linewidth=2,color='k')
        Coa_XYSlice.axhline(y=indexCoord[0][1],linestyle='--',linewidth=2,color='k')


        gridX,gridY = np.mgrid[min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]))/self.LUT.cell_count[0], min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]))/self.LUT.cell_count[2]]
        Coa_YZSlice.pcolormesh(gridX,gridY,mMAP[:,int(indexVal[1][0]),:],cmap=self.CMAP,linewidth=0)
        CS = Coa_YZSlice.contour(gridX,gridY,mMAP[:,int(indexVal[1][0]),:], levels=[0.65,0.75,0.95],colors=('g','m','k'))
        Coa_YZSlice.clabel(CS, inline=1, fontsize=10)
        #Coa_YZSlice_ell = Ellipse(xy=(expect_vector[1], expect_vector[2]),width=lambda_[1]*2, height=lambda_[2]*2,angle=np.rad2deg(np.arccos(v[1, 0])))
        #Coa_YZSlice_ell.set_facecolor('none')
        #Coa_YZSlice_ell.set_edgecolor('r')
        #Coa_YZSlice.add_artist(Coa_YZSlice_ell)
        #Coa_YZSlice.scatter(expect_vector[1], expect_vector[2], c="r", marker="x")
        Coa_YZSlice.set_xlim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,0])])
        Coa_YZSlice.set_ylim([max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]),min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2])])
        Coa_YZSlice.axvline(x=indexCoord[0][0],linestyle='--',linewidth=2,color='k')
        Coa_YZSlice.axhline(y=indexCoord[0][2],linestyle='--',linewidth=2,color='k')
        Coa_YZSlice.invert_yaxis()


        gridX,gridY = np.mgrid[min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]))/self.LUT.cell_count[2], min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]):(max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]) - min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]))/self.LUT.cell_count[1]]
        Coa_XZSlice.pcolormesh(gridX,gridY,mMAP[int(indexVal[0][0]),:,:].transpose(),cmap=self.CMAP,linewidth=0)
        CS = Coa_XZSlice.contour(gridX,gridY,mMAP[int(indexVal[0][0]),:,:].transpose(),levels =[0.65,0.75,0.95],colors=('g','m','k'))
        #Coa_XZSlice.clabel(CS, inline=1, fontsize=10)
        #Coa_XZSlice_ell = Ellipse(xy=(expect_vector[0], expect_vector[2]),width=lambda_[0]*2, height=lambda_[2]*2,angle=np.rad2deg(np.arccos(v[0, 0])))
        #Coa_XZSlice_ell.set_facecolor('none')
        #Coa_XZSlice_ell.set_edgecolor('r')
        #Coa_XZSlice.add_artist(Coa_XZSlice_ell)
        #Coa_XZSlice.scatter(expect_vector[0], expect_vector[2], c="r", marker="x")
        Coa_XZSlice.set_xlim([max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2]),min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,2])])
        Coa_XZSlice.set_ylim([min(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1]),max(self.LUT.xyz2coord(self.LUT.get_grid_xyz())[:,1])])
        Coa_XZSlice.axvline(x=indexCoord[0][2],linestyle='--',linewidth=2,color='k')
        Coa_XZSlice.axhline(y=indexCoord[0][1],linestyle='--',linewidth=2,color='k')


        Coa_XYSlice.scatter(self.LUT.station_data['Longitude'],self.LUT.station_data['Latitude'],15,'k',marker='^')
        Coa_YZSlice.scatter(self.LUT.station_data['Longitude'],self.LUT.station_data['Elevation'],15,'k',marker='^')
        Coa_XZSlice.scatter(self.LUT.station_data['Elevation'],self.LUT.station_data['Latitude'],15,'k',marker='<')
        for i,txt in enumerate(self.LUT.station_data['Name']):
            Coa_XYSlice.annotate(txt,[self.LUT.station_data['Longitude'][i],self.LUT.station_data['Latitude'][i]])


        # Plotting the logo
        Coa_Logo.axis('off')
        im = mpimg.imread(self.logoPath)
        Coa_Logo.imshow(im)
        Coa_Logo.text(150, 200, r'Earthquake Location Error', fontsize=10,style='italic')


        if SaveFilename == None:
            plt.show()

        else:
            plt.savefig('{}_EventLocationError.pdf'.format(SaveFilename),dpi=400)




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
        self.Decimate=10

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

        if (type == "CMS"):
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

    def __init__(self, DATA, lut, reader=None, param=None, output_path=None, output_name=None):
        self.sample_rate = 1000.0
        self.seis_reader = None
        self.lookup_table = lut
        self.DATA = DATA 

        if param is None:
            param = SeisScanParam()


        self.keep_map = False

        self.pre_pad   = 0.0
        self.post_pad  = 0.0
        self.time_step = 10.0

        self.daten = None
        self.dsnr  = None
        self.dloc  = None
        

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
        self.PickingType        = 'Gaussian'
        self.LocationError      = 0.95

        self.Output_SampleRate = None 


        #self.plot = SeisPlot(lut)

        self.MAP = None
        self.EVENT = None


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
        sig_z = onset(sig_z, stw, ltw)           # Determine the onset function using definition
        self.onset_data['sigz'] = sig_z          # Define the onset function from the data 
        return sig_z

    def _compute_onset_s1(self, sig_e, sig_n, srate):
        stw, ltw = self.onset_win_s1                                            # Define the STW and LTW for the onset function
        stw = int(stw * srate) + 1                                              # Changes the onset window to actual samples
        ltw = int(ltw * srate) + 1                                              # Changes the onset window to actual samples
        sig_e, sig_n = self._pre_proc_s1(sig_e, sig_n, srate)                   # Apply the pre-processing defintion
        self.filt_data['sige'] = sig_e                                          # Defining filtered signal to pass
        self.filt_data['sign'] = sig_n                                          # Defining filtered signal to pass
        sig_e = onset(sig_e, stw, ltw)                                          # Determine the onset function from the filtered signal
        sig_n = onset(sig_n, stw, ltw)                                          # Determine the onset function from the filtered signal
        self.onset_data['sige'] = sig_e                                         # Define the onset function from the data
        self.onset_data['sign'] = sig_n                                         # Define the onset function from the data                
        snr = np.sqrt(sig_e * sig_e + sig_n * sig_n)                            # Define the combined onset function from E & N
        self.onset_data['sigs'] = snr
        return snr


    def _compute(self, cstart,cend, samples,station_avaliability):

        srate = self.sample_rate


        avaInd = np.where(station_avaliability == 1)[0]
        sige = samples[0]
        sign = samples[1]
        sigz = samples[2]
        

        snr_p1 = self._compute_onset_p1(sigz, srate)
        snr_s1 = self._compute_onset_s1(sige, sign, srate)
        self.DATA.SNR_P = snr_p1
        self.DATA.SNR_S = snr_s1


        snr = np.concatenate((snr_p1, snr_s1))
        snr[np.isnan(snr)] = 0
        
        
        ttp = self.lookup_table.fetch_index('TIME_P', srate)
        tts = self.lookup_table.fetch_index('TIME_S', srate)
        tt = np.c_[ttp, tts]

        nchan, tsamp = snr.shape

        pre_smp = int(self.pre_pad * srate)
        pos_smp = int(self.post_pad * srate)
        nsamp = tsamp - pre_smp - pos_smp
        daten = 0.0 - pre_smp / srate

        ncell = tuple(self.lookup_table.cell_count)

        if self._map is None:
            #print('  Allocating memory: {}'.format(ncell + (tsamp,)))
            self._map = np.zeros(ncell + (tsamp,), dtype=np.float64)

        dind = np.zeros(tsamp, np.int64)
        dsnr = np.zeros(tsamp, np.double)

        ilib.scan(snr, tt, 0, pre_smp + nsamp +pos_smp, self._map, self.NumberOfCores)
        ilib.detect(self._map, dsnr, dind, 0, pre_smp + nsamp +pos_smp, self.NumberOfCores)

        daten = np.arange((cstart+timedelta(seconds=self.pre_pad)), (cend + timedelta(seconds=-self.post_pad) + timedelta(seconds=1/srate)),timedelta(seconds=1/srate)) 
        dsnr  = np.exp((dsnr[pre_smp:pre_smp + nsamp] / nchan) - 1.0)
        dloc  = self.lookup_table.index2xyz(dind[pre_smp:pre_smp + nsamp])

        MAP   = self._map[:,:,:,(pre_smp+1):pre_smp + nsamp]
        return daten, dsnr, dloc, MAP


    def _continious_compute(self,starttime,endtime):
        ''' 
            Continious seismic compute from 

        '''

        # 1. variables check
        # 2. Defining the pre- and post- padding
        # 3.  


        self.StartDateTime = datetime.strptime(starttime,'%Y-%m-%dT%H:%M:%S.%f')
        self.EndDateTime   = datetime.strptime(endtime,'%Y-%m-%dT%H:%M:%S.%f')



        if self.pre_pad == 0.0 and self.post_pad == 0.0:
            self.pre_pad = sum(np.max(np.array([self.onset_win_p1,self.onset_win_s1]),0))


        # ------- Continious Seismic Detection ------
        print('==============================================================================================================================')
        print('   Coalescence Microseismic Scanning : PATH:{} - NAME:{}'.format(self.output.path, self.output.name))
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
            self.output.write_scan(daten,dsnr,dcoord)

            i += 1



        # Changing format of SCN file to reduce filesize
        #self.output.write_decscan(self.DATA.sampling_rate)
        del daten, dsnr, dloc, map



    def _Trigger_scn(self,CoaVal,starttime,endtime):
        CoaVal = CoaVal[CoaVal['COA'] > self.DetectionThreshold]       

        CoaVal = CoaVal[(CoaVal['DT'] >= datetime.strptime(starttime,'%Y-%m-%dT%H:%M:%S.%f')) & (CoaVal['DT'] <= datetime.strptime(endtime,'%Y-%m-%dT%H:%M:%S.%f'))]
        CoaVal = CoaVal.sort_values(['COA'],ascending=False).reset_index(drop=True)
        CoaVal['EventID'] = 0
        c=1
        for i in range(len(CoaVal)):
            if CoaVal['EventID'].iloc[i] == 0:
                tmpDT_MIN = CoaVal['DT'].iloc[i] + timedelta(seconds=-self.MarginalWindow/2)
                tmpDT_MAX = CoaVal['DT'].iloc[i] + timedelta(seconds=self.MarginalWindow/2)


                tmpDATA = CoaVal[(CoaVal['DT'] >= tmpDT_MIN) & (CoaVal['DT'] <= tmpDT_MAX)]

                CoaVal['EventID'].iloc[tmpDATA.index] = c
                c+=1
            else:
                continue 

        EVENTS = pd.DataFrame(columns=['DT','COA','X','Y','Z'])
        if max(CoaVal['EventID']) > 0:
            for j in range(1,max(CoaVal['EventID'])+1):
                tmpDATA = CoaVal[CoaVal['EventID'] == j].sort_values(['COA'],ascending=False).reset_index(drop=True)

                EVENTS = EVENTS.append(tmpDATA.iloc[0])
        EVENTS.reset_index(drop=True)


        for e in range(len(EVENTS)):
            EVENTS['EventID'].iloc[e] = re.sub(r"\D", "",EVENTS['DT'].astype(str).iloc[e])

        return EVENTS





    def Detect(self,starttime,endtime):
        ''' 
           Function 

           Detection of the  

        '''
        # Conduct the continious compute on the decimated grid
        lut = self.lookup_table
        lut_decimate = lut.decimate([self.Decimate,self.Decimate,self.Decimate])
        self.lookup_table = lut_decimate

        # Dectect the possible events from the decimated grid
        self._continious_compute(starttime,endtime)


    def _GaussianTrigger(self,SNR,PHASE,cstart,eventT):
        '''
            Function to fit gaussian to onset function, based on knowledge of approximate trigger index, 
            lowest freq within signal and signal sampling rate. Will fit gaussian and return standard 
            deviation of gaussian, representative of timing error.
    
        '''

        sampling_rate = self.sample_rate
        trig_idx = int(((eventT-cstart).seconds + (eventT-cstart).microseconds/10.**6) *sampling_rate)

        if PHASE == 'P':
            lowfreq = self.bp_filter_p1[0]
        if PHASE == 'S':
            lowfreq = self.bp_filter_s1[0]

        data_half_range = int(1.25*sampling_rate/(lowfreq)) # half range number of indices to fit guassian over (for 1 wavelengths of lowest frequency component)
        x_data = np.arange(trig_idx-data_half_range, trig_idx+data_half_range,dtype=float)/sampling_rate # x data, in seconds
        y_data = SNR[int(trig_idx-data_half_range):int(trig_idx+data_half_range)] # +/- one wavelength of lowest frequency around trigger
        p0 = [np.amax(SNR), float(trig_idx)/sampling_rate, 1./(lowfreq/4.)] # Initial guess (should work for any sampling rate and frequency)
        
        try:
            popt, pcov = curve_fit(gaussian_func, x_data, y_data, p0) # Fit gaussian to data
            sigma = np.absolute(popt[2]) # Get standard deviation from gaussian fit
        except:
            sigma = -1

        return sigma



    def _ArrivalTrigger(self,EVENT_MaxCoa,EventName):
        '''
            FUNCTION - _ArrivalTrigger - Used to determine earthquake station arrival time

        '''

        SNR_P = self.DATA.SNR_P
        SNR_S = self.DATA.SNR_S

        ttp = self.lookup_table.value_at('TIME_P', np.array(self.lookup_table.coord2xyz(np.array([EVENT_MaxCoa[['X','Y','Z']].tolist()]))).astype(int))[0]
        tts = self.lookup_table.value_at('TIME_S', np.array(self.lookup_table.coord2xyz(np.array([EVENT_MaxCoa[['X','Y','Z']].tolist()]))).astype(int))[0]



        # Determining the stations that can be picked on and the phasese
        STATIONS=pd.DataFrame(columns=['Name','Phase','Pick','PickError'])
        for s in range(len(SNR_P)):
            if np.nansum(SNR_P[s]) !=  0:
                stationEventPT = EVENT_MaxCoa['DT'] + timedelta(seconds=ttp[s])

                if self.PickingType == 'Gaussian':
                    Err = self._GaussianTrigger(SNR_P[s],'P',self.DATA.startTime,stationEventPT.to_pydatetime())


                
                tmpSTATION = pd.DataFrame([[self.lookup_table.station_data['Name'][s],'P',stationEventPT,Err]],columns=['Name','Phase','Pick','PickError'])
                STATIONS = STATIONS.append(tmpSTATION)

            if np.nansum(SNR_S[s]) != 0:
                
                stationEventST = EVENT_MaxCoa['DT'] + timedelta(seconds=tts[s])

                if self.PickingType == 'Gaussian':
                    Err = self._GaussianTrigger(SNR_P[s],'S',self.DATA.startTime,stationEventPT.to_pydatetime())

                tmpSTATION = pd.DataFrame([[self.lookup_table.station_data['Name'][s],'S',stationEventST,Err]],columns=['Name','Phase','Pick','PickError'])
                STATIONS = STATIONS.append(tmpSTATION)

        #print(STATIONS)
        # Saving the output from the triggered events
        self.output.write_stationsfile(STATIONS,EventName)


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

        expect_vector = np.array([x_expect, y_expect, z_expect], dtype=float)

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


        # Converting the grid location to X,Y,Z

        expect_vector = self.lookup_table.xyz2coord(self.lookup_table.loc2xyz(np.array([[expect_vector[0]/self.lookup_table.cell_size[0],expect_vector[1]/self.lookup_table.cell_size[1],expect_vector[2]/self.lookup_table.cell_size[2]]])))[0]

        return expect_vector, cov_matrix

    def _LocationError(self,Map4D):

        '''
            Function

        '''

        # Determining the coalescence 3D map
        CoaMap = np.log(np.sum(np.exp(Map4D),axis=-1))
        CoaMap = CoaMap/np.max(CoaMap)

        CoaMap_Cutoff = np.percentile(CoaMap,95)
        CoaMap[CoaMap < CoaMap_Cutoff] = CoaMap_Cutoff 
        CoaMap = CoaMap - CoaMap_Cutoff 
        CoaMap = CoaMap/np.max(CoaMap)

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
        # 
        Triggered = pd.DataFrame(columns=['DT','COA','X','Y','Z','ErrX','ErrY','ErrZ'])
        for e in range(len(EVENTS)):

            print('--Processing for Event {} of {} - {}'.format(e+1,len(EVENTS),EVENTS['EventID'].iloc[e]))

            # Determining the Seismic event location
            cstart = EVENTS['DT'].iloc[e].to_pydatetime() + timedelta(seconds = -self.MarginalWindow/2) + timedelta(seconds = -self.pre_pad)
            cend   = EVENTS['DT'].iloc[e].to_pydatetime() + timedelta(seconds = self.MarginalWindow/2) + timedelta(seconds = self.post_pad)
            self.DATA.read_mseed(cstart.strftime('%Y-%m-%dT%H:%M:%S.%f'),cend.strftime('%Y-%m-%dT%H:%M:%S.%f'),self.sample_rate)

            self._map = None
            daten, dsnr, dloc, MAP = self._compute(cstart,cend,self.DATA.signal,self.DATA.station_avaliability)
            dcoord = self.lookup_table.xyz2coord(np.array(dloc).astype(int))
            EventCoaVal = pd.DataFrame(np.array((daten,dsnr,dcoord[:,0],dcoord[:,1],dcoord[:,2])).transpose(),columns=['DT','COA','X','Y','Z'])
            EventCoaVal['DT'] = pd.to_datetime(EventCoaVal['DT'])

            EVENT = EventCoaVal.sort_values(by=['COA'],ascending=False).iloc[0]

            pickle.dump(self.DATA,open("ONSET.p", "wb" ))

            self.MAP = MAP
            self.EVENT = EVENT
            self.cstart = cstart
            self.cend   = cend


            # Determining the hypocentral location from the maximum over the marginal window.
            self._ArrivalTrigger(EVENT,EVENTS['EventID'].iloc[e])


            
            # Determining earthquake location error
            LOC,LOC_ERR = self._LocationError(MAP)

            EVENT['X_ErrE'] = LOC[0]
            EVENT['Y_ErrE'] = LOC[1]
            EVENT['Z_ErrE'] = LOC[2]

            EVENT['ErrX'] = LOC_ERR[0]
            EVENT['ErrY'] = LOC_ERR[1]
            EVENT['ErrZ'] = LOC_ERR[2]

            
            self.output.write_event(EVENT,EVENTS['EventID'].iloc[e])

            # Outputting coalescence grids and triggered events
            if self.CoalescenceGrid == True:

                # CoaMap = {}
                # CoaMap['MAP'] = MAP
                # CoaMap['CoaDATA'] = tmpEvent
                self.output.write_coal4D(MAP,e,cstart,cend)


            if self.CoalescenceVideo == True:
                self.MAP = MAP
                self.EVENT = EventCoaVal
                self.output.write_coalVideo(self.lookup_table,MAP,self.DATA,EventCoaVal,EVENTS['EventID'].iloc[e])




            
            Triggered = Triggered.append(EVENT)



        return Triggered

