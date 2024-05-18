import nitime.algorithms as tsa  # has the multitaper routine
import numpy as np
import scipy.signal as sig

def dB(x, out=None):
    if out is None:
        return 10 * np.log10(x)
    else:
        np.log10(x, out)
        np.multiply(out, 10, out)

def hz2bark(hz):
        '''
        Convert frequency in Hz to Bark using the Schroeder 1977 formula.

        Parameters
        ----------

        hz: scalar or array
        Frequency in Hz.

        Returns
        -------
        scalar or array
        Frequency in Bark, in an array the same size as `hz`, or a scalar if
        `hz` is a scalar.
        '''
        return 7 * np.arcsinh(hz/650)

def main_peak(x,mp,sf):
    ''' 
    Measure fricative acoustic values, from a 20 ms window centered on time <mp>
    
    Parameters
    ----------

    x -- a one-dimensional array of audio samples
    mp -- the midpoint (in seconds) of a fricative from which to take measurements
    sf -- the sampling frequency of x (ideally should be at least 22050)

    Returns
    --------
     
    Fm -- Frequency of the main peak in Hertz (F1)
    FmB -- Frequency of the main peak on the Bark scale
    Am -- Amplitude (in dB) at Fm
    Ll -- mean Level in the lowest band (550-3000 Hz)
    Lm -- mean Level in the mid band (3000-7000 Hz)
    Lh -- mean Level in the high band (7000-nyquist freq)
    COG -- center of gravity 
    SD -- standard deviation 
    Skew -- scaled third moment, skew
    Kurtosis -- scaled fourth moment, kurtosis
    spec -- the multi-taper power spectrum at the midpoint
    freq -- the frequency scale of the spectrum

    '''
    winsize = 0.02   # 20 ms window centered at midpoint (mp)
    
    i_center = int(mp * sf)   # index of midpoint time: seconds to samples
    i_start  = int(i_center - winsize/2*sf)  # back 10 ms
    i_end = int(i_center + winsize/2*sf)     # forward 10 ms
    
    f, psd, nu = tsa.multi_taper_psd(x[i_start:i_end])  # get multi-taper spectrum
    
    spec = dB(psd)           # work with magnitude spectrum
    freq = (f/(2*np.pi))*sf   # frequency axis for plotting spectra
    nyquist = sf/2
    fspace = nyquist/len(f)  # frequency spacing in spectrum - map from frequency to array index
    
    i1 = int(550/fspace)  # indeces of band edges
    i2 = int(3000/fspace)
    i3 = int(7000/fspace)
    i4 = int(nyquist/fspace)

    Ll = dB(np.sum(psd[i1:i2])/len(f[i1:i2]))  # level in low band
    Lm = dB(np.sum(psd[i2:i3])/len(f[i2:i3]))  # level in mid band
    Lh = dB(np.sum(psd[i3:i4])/len(f[i3:i4]))  # level in high band

    '''
    Find the "main peak" in the fricative spectrum.  For fricatives with a resonant cavity, 
    this is the lowest resonance of the cavity.  For those with no resonance this may be 
    more a property of the source function.
    
    The function 'scipy.signal.find_peaks()' reports peaks that fit the criteria in frequency 
    order, starting with the lowest peak in the spectrum that fits the criteria.
    
    The heuristic is to start by looking the in the spectrum above 500 Hz, for a peak that 
    is at least 80% of the amplitude range, separated by the nearest peack by 500Hz, and prominent
    by 6bB (see find_peaks() documentation.  If no peak is found, relax the amplitude contraint, 
    then the prominence constraint, and then both. 
    
    The variable 'mode' reports how the peak was found.
    '''
    bottom_freq = int(500/fspace)  # bottom of the search space
    top_freq = int(11000/fspace)  # set frequency range for analysis (11kHz is max
    if (nyquist < top_freq):      # but if sampling rate is less than 22kHz, the max
        top_freq = nyquist        # is reduced to the Nyquist freq (1/2 the sampling rate)
    min_dist = int(500/fspace)  # for peak picking - peaks must be at least this far apart

    mode = '1. 500/80%/6dB'
    min_height =  np.min(spec) + (np.max(spec)-np.min(spec))*0.8  # 80% of the range in this spectrum
    min_prom = 6  # dB
    (peaks,prop) = sig.find_peaks(spec[bottom_freq:top_freq], height=min_height,
                                  distance=min_dist, prominence=min_prom)
    if (len(peaks)<1):  # if we didn't find any peaks, relax the constraints
        mode = '2. 500/70%/6dB'
        min_height =  np.min(spec) + (np.max(spec)-np.min(spec))*0.7
        min_prom = 6  # dB
        (peaks,prop) = sig.find_peaks(spec[bottom_freq:top_freq],height=min_height,
                                      distance=min_dist, prominence=min_prom)
    if (len(peaks)<1):
        mode = '3. 500/65%/4dB'
        min_height =  np.min(spec) + (np.max(spec)-np.min(spec))*0.65
        min_prom = 4  # dB
        (peaks,prop) = sig.find_peaks(spec[bottom_freq:top_freq],height=min_height,
                                      distance=min_dist, prominence=min_prom)
    if (len(peaks)<1):
        mode = '5. 500/33%/3dB'
        min_height =  np.min(spec) + (np.max(spec)-np.min(spec))*0.33
        min_prom = 3  # dB
        (peaks,prop) = sig.find_peaks(spec[bottom_freq:top_freq],height=min_height,
                                      distance=min_dist, prominence=min_prom)
    if (len(peaks)<1):
        mode = '6. give up'
        peaks = np.append(peaks,[0])

    index_of_main_peak = bottom_freq + peaks[0]  # this is selecting the first peak in an array of peaks
    Fm = freq[index_of_main_peak]      # convert to Hz
    FmB = hz2bark(Fm)                   # convert to Bark
    Am = spec[index_of_main_peak]      # get amplitude

    # -------- moments analysis -----------------------
    bottom_freq = int(300/fspace)
    f = freq[bottom_freq:top_freq]  # taking frequencies from 300 to 11000 (or lower if sf < 22000)
    temp = spec[bottom_freq:top_freq] - np.min(spec[bottom_freq:top_freq]) # make sure none are negative
    Ps = temp/np.sum(temp)  # scale to sum to 1
    COG = np.sum(Ps*f)  # center 
    dev = f-COG  # deviations from COG
    Var = np.sum(Ps * dev**2)  # second moment
    SD = np.sqrt(Var)  # Standard deviation
    Skew = np.sum(Ps * dev**3)  # third moment 
    Kurtosis = np.sum(Ps * dev**4)  # fourth moment

    # scaling recommended by Forrest et al. 1990
    Skew = Skew/np.sqrt(Var**3)
    Kurtosis = Kurtosis/(Var**2) - 3
    
    return(Fm,FmB,Am,Ll,Lm,Lh,mode,COG,SD,Skew,Kurtosis,spec,freq)
