############################################################################
############## Scripts for Scanning and Coalescence of Data ################
############################################################################
# ---- Import Packages -----
import obspy
from obspy import UTCDateTime

from datetime import datetime
from datetime import timedelta
from glob import glob

import numpy as np


# ----- Useful Functions -----
def _downsample(st,sr):
    '''
        Downsampling the MSEED to the designated sampling rate
    '''
    for i in range(0,len(st)):
        st[i].decimate(factor=int(st[i].stats.sampling_rate/sr), strict_length=False)

    return st



class MSEED():

    def __init__(self,lut,HOST_PATH='/PATH/MSEED'):

        
        self.lookup_table        = lut

        self.startTime           = None
        self.endTime             = None
        self.sampling_rate       = None
        self.MSEED_path          = HOST_PATH

        self.Type                = None 
        self.FILES               = None 
        self.signal              = None
        self.FilteredSignal      = None
        self.StationAvaliability = None
        self.StationInformation  = lut.station_data


    def _stationAvaliability(self,st):
        ''' Reading the Avaliability of the stations between two times 

        '''
        stT  = self.startTime
        endT = self.endTime

        # Since the traces are the same sample-rates then the stations can be selected based
        #on the start and end time
        exSamples = (endT-stT).total_seconds()*self.sampling_rate + 1


        stationAva = np.zeros((len(self.lookup_table.station_data['Name']),1))
        signal     = np.zeros((3,len(self.lookup_table.station_data['Name']),int(exSamples)))

        for i in range(0,len(self.lookup_table.station_data['Name'])):

            tmp_st = st.select(station=self.lookup_table.station_data['Name'][i])
            if len(tmp_st) == 3:
                if tmp_st[0].stats.npts == exSamples and tmp_st[1].stats.npts == exSamples and tmp_st[2].stats.npts == exSamples:
                    # Defining the station as avaliable
                    stationAva[i] = 1
                    
                    for tr in tmp_st:
                        # Giving each component the correct signal
                        if tr.stats.channel[-1] == 'E' or tr.stats.channel[-1] == '2':
                            signal[1,i,:] = tr.data

                        if tr.stats.channel[-1] == 'N' or tr.stats.channel[-1] == '1':
                            signal[0,i,:] = tr.data

                        if tr.stats.channel[-1] == 'Z':
                            signal[2,i,:] = tr.data

            else:
                # Trace not completly active during this period
                continue 




        return signal,stationAva


    def path_structure(self,TYPE='YEAR/JD/STATION'):
        ''' Function to define the path structure of the mseed. 

            This is a complex problem and will depend entirely on how the data is structured.
            Since the reading of the headers is quick we only need to get to the write data.
        '''

        if TYPE == 'YEAR/JD/STATION':
            self.Type  = 'YEAR/JD/STATION'




    def _load_fromPath(self):
        '''
            Given the type of path structure load the data in the required format

        '''
        if self.Type == None:
            print('Please Specfiy the path_structure - DATA.path_structure')
            return

        if self.Type == 'YEAR/JD/STATION':
            dy = 0
            FILES = []
            while (self.endTime.timetuple().tm_yday) >= (self.startTime + timedelta(days=dy)).timetuple().tm_yday:
                # Determine current time
                ctime = self.startTime + timedelta(days=dy)

                for st in self.lookup_table.station_data['Name'].tolist():
                    FILES.extend(glob('{}/{}/{}/*{}*'.format(self.MSEED_path,ctime.year,ctime.timetuple().tm_yday,st)))

                dy += 1 

        self.FILES = FILES


    def read_mseed(self,starttime,endtime,sampling_rate):
        ''' 
            Reading the required mseed files for all stations between two times and return 
            station avaliability of the seperate stations during this period
        '''


        self.startTime = datetime.strptime(starttime,'%Y-%m-%dT%H:%M:%S.%f')
        self.endTime      = datetime.strptime(endtime,'%Y-%m-%dT%H:%M:%S.%f')

        self._load_fromPath()

        #print('Loading the MSEED')

        # Loading the required mseed data
        c=0
        for f in self.FILES:
          try:
             if c==0:
                st = obspy.read(f,starttime=UTCDateTime(self.startTime),endtime=UTCDateTime(self.endTime))
                c +=1
             else:
                st += obspy.read(f,starttime=UTCDateTime(self.startTime),endtime=UTCDateTime(self.endTime))
          except:
            continue
            #print('Station File not MSEED - {}'.format(f))

        # Removing all the stations with gaps
        if len(st.get_gaps()) > 0:
            stationRem = np.unique(np.array(st.get_gaps())[:,1]).tolist()
            for sa in stationRem:
                tr = st.select(station=sa)
                for tra in tr:

                    st.remove(tra) 


        # Combining the mseed and determining station avaliability
        #print('Detrending and Merging MSEED')
        #st.detrend()
        st.merge()
        


        
        # Downsample the mseed to the same level
        #print('Downsampling MSEED')
        st = _downsample(st,sampling_rate)
        self.sampling_rate = sampling_rate

        # Checking the station Avaliability for each of the stations across this time period
        #print('stationAvaliability MSEED')
        signal,stA = self._stationAvaliability(st)


        self.signal  = signal
        self.FilteredSignal = np.empty((self.signal.shape))
        self.FilteredSignal[:] = np.nan
        self.station_avaliability = stA


