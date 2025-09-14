import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm
import pandas as pd
from datetime import datetime
import matplotlib.ticker as ticker
import matplotlib.dates as dates
import argparse
import json
import os
import csv
import sys
from dateutil.parser import parse
import warnings
import time
import scipy.stats
from scipy import optimize
from statsmodels.tsa.stattools import adfuller,coint
from scipy.stats import wasserstein_distance,ks_2samp
from scipy import signal
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.spatial.distance import euclidean,cosine,chebyshev,canberra
import calibrate

#from fastdtw import fastdtw



VERSION="3.0"


def simpson_integration(y, x=None):
    """
    Perform Simpson's rule integration compatible with all versions of SciPy.

    Parameters:
    - y: array-like, values of the function to integrate.
    - x: array-like, optional, the sample points corresponding to the `y` values.
         If not provided, it is assumed `y` is sampled at evenly spaced points.

    Returns:
    - float, the integral.
    """
    # Check if scipy.integrate.simpson is available
    if hasattr(scipy.integrate, 'simpson'):
        return scipy.integrate.simpson(y, x=x)
    else:
        # Fallback for older versions: Use scipy.integrate.simps
        if x is not None:
            return scipy.integrate.simps(y, x=x)
        else:
            return scipy.integrate.simps(y)

def trapezoid_integration(y, x=None):
    """
    Perform Trapezoid's rule integration compatible with all versions of SciPy.

    Parameters:
    - y: array-like, values of the function to integrate.
    - x: array-like, optional, the sample points corresponding to the `y` values.
         If not provided, it is assumed `y` is sampled at evenly spaced points.

    Returns:
    - float, the integral.
    """
    # Check if scipy.integrate.simpson is available
    if hasattr(scipy.integrate, 'trapezoid'):
        return scipy.integrate.trapezoid(y, x=x)
    else:
        # Fallback for older versions: Use scipy.integrate.simps
        if x is not None:
            return scipy.integrate.trapz(y, x=x)
        else:
            return scipy.integrate.trapz(y)




def dtw(s, t):
    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s[i-1] - t[j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix


def dtw_distance(ss,tt):
    #normalise
    ms=np.mean(ss)
    mt=np.mean(tt)
    stds=np.std(ss)
    stdt=np.std(tt)
    s=[(x-ms)/stds for x in ss]
    t=[(x-mt)/stdt for x in tt]
    '''
    distance,_ = fastdtw(s, t, dist=euclidean)
    return distance/len(s)
    '''
    dtw_mat=dtw(s,t)
    n=len(ss)
    return dtw_mat[n,n]/n

def calculate_mse(col1,col2):
    l=len(col1)
    sum=0
    for i in range(l):
        diff=col1.iloc[i]-col2.iloc[i]
        sum+=diff**2
    return np.sqrt(sum/l)


def match_dtw(s,t,maxlag=1):
    best_dtw=dtw_distance(s,t)
    best_lag=0
    for i in range(1,maxlag):
        d=dtw_distance(s[i:],t[:-i])
        if d<best_dtw:
            best_dtw=d
            best_lag=i

    return best_dtw,best_lag

def compute_part1(col1,col2):
    l=len(col1)
    sum1=0
    sum2=0
    for i in range(l):
        sum1+=col1.iloc[i]
        sum2+=col2.iloc[i]
    dv=0.0
    if sum2!=0.0:
        dv=sum2/sum1
    return (sum1,sum2,dv)

def pearsonr_2D(x, y):
    """computes pearson correlation coefficient
       where x is a 1D and y a 2D array"""

    upper = np.sum((x - np.mean(x)) * (y - np.mean(y, axis=1)[:,None]), axis=1)
    lower = np.sqrt(np.sum(np.power(x - np.mean(x), 2)) * np.sum(np.power(y - np.mean(y, axis=1)[:,None], 2), axis=1))

    rho = upper / lower

    return rho

def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def lowpass_filter(data, band_limit, sampling_rate):
     cutoff_index = int(band_limit * data.size / sampling_rate)
     F = np.fft.rfft(data)
     F[cutoff_index + 1:] = 0
     return np.fft.irfft(F, n=data.size).real

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def spectral_entropy(x, freq, nfft=None,do_detrend=False):    
    if do_detrend:
        x=signal.detrend(x)
    _, psd = signal.periodogram(x, freq, nfft = nfft)   
    # calculate shannon entropy of normalized psd
    psd_norm = psd / np.sum(psd)
    entropy = np.nansum(psd_norm * np.log2(psd_norm))
    return  -(entropy / np.log2(psd_norm.size))


def get_flat_groups(df,row,min_num_points,max_difference=None):
    # group by maximum difference
    if max_difference is None:
        stdevs=np.sqrt(df[row].var())
        max_difference=stdevs*0.2
    group_ids = (abs(df[row].diff(1)) > max_difference).cumsum()
    ret_groups=[]
    for group_idx, group_data in df.groupby(group_ids):
        # filter non-plateaus by min number of points
        if len(group_data) < min_num_points:
            continue
        ret_groups.append(group_data)
    return ret_groups

def merge_groups(grp1,grp2,min_num_points):
    res=[]
    for g1 in grp1:
        for g2 in grp2:
            #r=pd.merge(g1,g2,how="inner")
            r=g1[g1.isin(g2)].dropna()
            if len(r)>=min_num_points:
                res.append(r)
    return res

def check_match(x,y,freq,dtw=None,use_logistic=True):
    #coef1=-77.9667
    #coef2=112.4261
    #coef3=97.2695
    coef=[-3.7722,9.1555,-3.0262,12.0903]
    #coef=[-3.9791,10.3796,0.03903]
    sig1=signal.detrend(x)
    Y1=np.fft.fft(sig1)
    N=len(Y1)
    sig2=signal.detrend(y)
    Y2=np.fft.fft(sig2)
    my_rho = np.corrcoef(np.abs(Y1[1:N//2]), np.abs(Y2[1:N//2]))
    dt=dtw_distance(np.abs(Y1[1:N//2]), np.abs(Y2[1:N//2]))
    if not dtw:
        dtw=dtw_distance(x,y)

    dt,lfft=match_dtw(np.abs(Y1[1:N//2]), np.abs(Y2[1:N//2]),1)
    dtw,dl=match_dtw(x,y,1)
    se1=spectral_entropy(x,freq, nfft=None,do_detrend=True)
    se2=spectral_entropy(y,freq, nfft=None,do_detrend=True)
    r=1/(1+np.exp(coef[0]+coef[1]*dtw+coef[2]*dt+coef[3]*np.abs(se1-se2)))
    #r=1/(1+np.exp(coef[0]+coef[1]*dtw+coef[2]*np.abs(se1-se2)))

    print("DTW: ",round(dtw,4),"   FFTDTW: ",round(dt,4),"  SEDiff: ",np.abs(se2-se1),"   RHO: ",round(my_rho[0][1],5))
    prob=np.abs(my_rho[0][1])*r if use_logistic else 1-(dt+dtw)/2
    return prob
        


def plot1(df,args,index1,index2):
    vars=list(df.columns)
    fig, axs = plt.subplots(2, sharex=True)
    
    df["AccTrapezoid"].plot(x=df.index,ax=axs[0],legend=True, linewidth=0.75)
    df["AccSimpson"].plot(x=df.index,ax=axs[0],legend=True, linewidth=0.75)
    df[vars[index1]].plot(x=df.index,ax=axs[1],legend=True, linewidth=0.75)
    df[vars[index2]].plot(x=df.index,ax=axs[1],legend=True, linewidth=0.75)

    if args.add_grouping:
        s1=vars[index1]
        s2=vars[index2]
        groups1=get_flat_groups(df,s1,20)
        groups2=get_flat_groups(df,s2,20)
        grp=merge_groups(groups1,groups2,30)
        for g in grp:
            g[s2].plot(x=g.index, ax=axs[1], label="Plateau", marker='x', lw=1.5, ms=5.0,)

    if args.display_grid:
        axs[0].grid(True)
        axs[1].grid(True)
        plt.grid(linestyle = '--', linewidth = 0.5)
    plt.legend(fontsize="10")
    plt.tight_layout()


def plot2(df,args,index1,index2):
    vars=list(df.columns)
    fig, axs = plt.subplots(2, sharex=True)
    
    df["Diff"].plot(x=df.index,ax=axs[0],legend=True, linewidth=0.75)
    df[vars[index1]].plot(x=df.index,ax=axs[1],legend=True, linewidth=0.75)
    df[vars[index2]].plot(x=df.index,ax=axs[1],legend=True, linewidth=0.75)

    if args.display_grid:
        axs[0].grid(True)
        axs[1].grid(True)
        plt.grid(linestyle = '--', linewidth = 0.5)

    plt.legend(fontsize="10")
    plt.tight_layout()

def plot3(df,var,args,index1,index2):
    vars=list(df.columns)
    fig, axs = plt.subplots()
    
    if var&1 or index2 is None:
        df[vars[index1]].plot(x=df.index,ax=axs,legend=True, linewidth=0.75)
    if var&2 and not index2 is None:
        df[vars[index2]].plot(x=df.index,ax=axs,legend=True, linewidth=0.75)
    if args.display_grid:
        axs.grid(True)
        plt.grid(linestyle = '--', linewidth = 0.5)
    plt.legend(fontsize="10")
    plt.tight_layout()

def plot4(df,args,index1,index2):
    vars=list(df.columns)
    cnt=1
    if not index2 is None:
        cnt+=1
    fig, axs = plt.subplots(cnt, sharex=True)
    
    if cnt==1:
        axs=[axs]
    df[vars[index1]].hist(ax=axs[0],legend=True)
    if not index2 is None:
        df[vars[index2]].hist(ax=axs[0],legend=True)
    if not index2 is None:
        df['Diff'].hist(ax=axs[1],legend=True)
    plt.legend(fontsize="10")
    plt.tight_layout()

def DFT(x):
    """
    Function to calculate the
    discrete Fourier Transform
    of a 1D real-valued signal x
    """

    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)

    X = np.dot(e, x)

    return X

def plot5(df,args,index1,index2):
    vars=list(df.columns)
    fig, axs = plt.subplots(2, sharex=True)
    
    Y=np.fft.fft(df[vars[index1]])
    freq=np.fft.fftfreq(len(df),df['TimeDiff'][1])
    N=len(Y)
    axs[0].plot( freq[1:N//2], np.abs(Y[1:N//2]), label=vars[index1])
    if not index2 is None:
        Y1=np.fft.fft(df[vars[index2]])
        axs[0].plot( freq[1:N//2], np.abs(Y1[1:N//2]), label=vars[index2])
        Y2=np.fft.fft(df['Diff'])
        axs[1].plot( freq[1:N//2], np.abs(Y2[1:N//2]), label='Difference')
        #add correlation data
        my_rho = np.corrcoef(np.abs(Y[1:N//2]), np.abs(Y2[1:N//2]))
        text=''.join((r'$\rho=%.4f$' % (my_rho[0][1], )))
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axs[0].text(0.05, 0.95, text, transform=axs[0].transAxes, fontsize=14,verticalalignment='top', bbox=props)


    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()



def plot_cluster(df,args,index1,index2):
    vars=list(df.columns)
    fig, axs = plt.subplots(2, sharex=True)
    
    df[vars[index1]].plot(x=df.index,ax=axs[0],legend=True, linewidth=0.75,label=vars[index1])
    df[vars[index2]].plot(x=df.index,ax=axs[0],legend=True, linewidth=0.75,label=vars[index2])
    df["predicted_value"].plot(x=df.index,ax=axs[0],legend=True, linewidth=0.75,label="Predicted "+vars[index2])
    df["values_diff"].plot(x=df.index,ax=axs[1],legend=True, linewidth=0.75,label=f"{vars[index1]}-{vars[index2]}")
    df["predicted_diff"].plot(x=df.index,ax=axs[1],legend=True, linewidth=0.75,label=f"{vars[index1]}-predicted {vars[index2]}")

    if args.display_grid:
        axs[0].grid(True)
        axs[1].grid(True)
        plt.grid(linestyle = '--', linewidth = 0.5)
    plt.tight_layout()

def filter_last_n_hours(df, hours, time_column='TIME'):
    """
    Filters the DataFrame to include only rows from the last N hours,
    based on the range of time in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        hours (int): The number of hours to retain.
        time_column (str): The name of the datetime column if it's not the index.
    
    Returns:
        pd.DataFrame: A DataFrame containing rows from the last N hours.
    """
    # Ensure the column is a datetime type
    if time_column in df.columns:
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.set_index(time_column)
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame must have a datetime index or a column specified as 'time_column'.")
    
    # Calculate the cutoff time
    max_time = df.index.max()
    cutoff_time = max_time - pd.Timedelta(hours=hours)
    
    # Filter the DataFrame
    filtered_df = df[df.index < cutoff_time]
    
    return filtered_df


def find_lsq(df,val):
    # Convert datetime index to numeric (e.g., seconds since start)
    x = np.array((df.index - df.index[0]).total_seconds())
    y = df[val].to_numpy()

    # Calculate the mean of x and y
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate the slope (m) and intercept (b) for the least-squares line
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope,intercept


def integrate_diff(df,var):
    df["Trapez"+var] = [0.0] * len(df) #4
    for i in range(len(df)-1):
        #trapezoid calc
        df["Trapez"+var].iloc[i] = df["TimeDiff"].iloc[i]/3600*0.5*(df[var].iloc[i]+df[var].iloc[i+1])
    df["AccTrapezoid"+var] = df["Trapez"+var].cumsum()
    return df

def calibrate_and_chart(df,var1,var2,args):
    y=[]
    x=list(df.index)
    cs=calibrate.CalibrationSystem(args)
    firstIndex=None
    for i in range(len(df)):
        cs.add_sample_and_calibrate(x[i],df[var1].iloc[i],df[var2].iloc[i],0)
        if cs.firstUpdate and not firstIndex:
            firstIndex=pd.to_datetime(x[i])
        y.append(cs.InletFlowMultiplier)
    newdf=df[[var1,var2]]
    #x=[time.mktime(xx.timetuple()) for xx in list(df.index)]
    x=[pd.to_datetime(xx) for xx in list(df.index)]
    newdf["TIME"]=x
    newdf["Calibration"]=y
    newdf['TimeDiff'] = -newdf['TIME'].diff(-1).dt.total_seconds()
    newdf['TimeDiff'] = newdf['TimeDiff'].fillna(0)
    fig, axs = plt.subplots(3, sharex=True)
    if args.synchronize_axis:
        axs[1].sharey(axs[0])
    #Plot integration curve Integral of (Flow2 - Flow1) 
    newdf["FlowDiff"] = newdf[var2]-newdf[var1]
    #Plot modelling curve integral of (Flow1*MeterFactor - Flow2)
    newdf["ModelDiff"] = newdf[var2]-newdf[var1]*y

    if args.synchronize_index:
        newdf=newdf[newdf['TIME']>=firstIndex]
    newdf.set_index("TIME")
    newdf["FlowDiffSum"] = newdf["FlowDiff"].cumsum()
    newdf["ModelDiffSum"] = newdf["ModelDiff"].cumsum()
    newdf=integrate_diff(newdf,"FlowDiff")
    newdf=integrate_diff(newdf,"ModelDiff")
    if args.useopt:
        optimal=newdf["Calibration"].mean()
        optimal=round(optimal,4)
        newdf["ModelDiffOpt"] = newdf[var2]-newdf[var1]*optimal
        newdf=integrate_diff(newdf,"ModelDiffOpt")
    if args.uselsq:
        slope,intercept=find_lsq(newdf,"Calibration")
        x_1 = np.array((newdf.index - newdf.index[0]).total_seconds())
        fitted_line = slope * x_1 + intercept
        newdf["ModelDiffLsq"] = newdf[var2]-newdf[var1]*fitted_line
        newdf=integrate_diff(newdf,"ModelDiffLsq")
    
    if args.CalibrationOffsetSamplesFromCurrentTime:
        newdf=filter_last_n_hours(newdf,args.CalibrationOffsetSamplesFromCurrentTime)
        if args.uselsq:
            fitted_line=fitted_line[:len(newdf)]

    if args.calibration_output:
        out_cols=["AccTrapezoidFlowDiff"]
        newdf[out_cols].to_csv(args.calibration_output,float_format="%.4f")
        
    #Plot calibration curve
    for ax in axs:
        ax.ticklabel_format(useOffset=False, style='plain')
    newdf["Calibration"].plot(x=newdf.index,ax=axs[2], legend=True,label="Calibration")
    if args.useopt:
        axs[2].axhline(y=optimal, color='r', linestyle='--', label=f"Optimal: {optimal}")
        newdf["AccTrapezoidModelDiffOpt"].plot(x=newdf.index,ax=axs[1],legend=True, color='red', linewidth=0.75,label=f"Opt Model diff {var2} - {var1} * OptMFactor")
    newdf["AccTrapezoidFlowDiff"].plot(x=newdf.index,ax=axs[0],legend=True, linewidth=0.75,label=f"Integral of flow diff {var2} - {var1}" )
    newdf["AccTrapezoidModelDiff"].plot(x=newdf.index,ax=axs[1],legend=True, linewidth=0.75,label=f"Integral of Model diff {var2} - {var1} * MFactor")
    if args.uselsq:
        axs[2].plot(newdf.index, fitted_line, color='green', linestyle='--', label=f"Fitted Line: y = {slope:.8f}x + {intercept:.6f}")
        newdf["AccTrapezoidModelDiffLsq"].plot(x=newdf.index,ax=axs[1],legend=True, color='green', linewidth=0.75,label=f"Integral of LSQ Model diff {var2} - {var1} * LSQ")
    for ax in axs:
        ax.legend(fontsize="8",loc ="lower left")
    if args.display_grid:
        axs[0].grid(True)
        axs[1].grid(True)
        axs[2].grid(True)
        plt.grid(linestyle = '--', linewidth = 0.5)
    plt.tight_layout()

def calc_c(q1,q2):
    s1=0
    s2=0
    for i,j in list(zip(q1,q2)):
        s1+=i*j
        s2+=j**2
    return s1/s2


def decode_flow(flow,vars):
    if flow.isdigit():
        return int(flow)
    if flow in vars:
        return vars.index(flow)
    return -1

def PandasResample(df, length):
    td = (df.index[-1] - df.index[0]) / (length-1)
    return df.resample(td).mean().interpolate() # Handle NaNs when upsampling


def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = np.sin((2 * np.pi) * frequencies * 10)
    return x, y

def process_bias_node(node):
    b={
        "from":None,
        "to":None,
        "bias1":0,
        "bias2":0,
        "mult1":1.0,
        "mult2":1.0
    }
    if "from" in node:
        b["from"]=parse(node["from"],yearfirst=True)
    if "to" in node:
        b["to"]=parse(node["to"],yearfirst=True)
    for s in ["bias1","bias2","mult1","mult2"]:
        if s in node:
            b[s]=node[s]
    return b

def process_config(args):
    with open(args.config, 'r') as j:
        config = json.loads(j.read())  
    bias_list=[] 
    if "bias_list" in config:
        for b in config["bias_list"]:
            bias_list.append(process_bias_node(b))
    args.bias_list=bias_list

def main(args):
    filter_from=None
    filter_to=None

    warnings.filterwarnings("ignore")
    df=pd.read_csv(args.input)
    df["TIME"]=pd.to_datetime(df["TIME"],format="%Y/%m/%d %H:%M:%S.%f")


    if not args.frm is None:
        filter_from=parse(args.frm,yearfirst=True)
    if not args.to is None:
        filter_to=parse(args.to,yearfirst=True)

    if filter_from and filter_to:
        df=df[(df['TIME']>=filter_from) & (df['TIME']<=filter_to)] 
    elif filter_from:
        df=df[(df['TIME']>=filter_from)] 
    elif filter_to:
        df=df[(df['TIME']<=filter_to)] 

    if args.config:
        process_config(args)
    else:
        args.bias_list=None

    cols={}
    for c in df.columns:
        i=c.find(':VAL')
        if i!=-1:
            cols[c]=c[:i]
        else:
            cols[c]=c
        cols[c]=cols[c].strip()
    df.rename(columns=cols,inplace=True)

    if args.display_columns:
        for c in cols:
            print(c)

    if args.single_flow:
        single_flow(args,df)
    else:
        if not args.flows or (len(args.flows)==2 and not args.baseflow):
            multi_flow(args,df)
        elif args.flows:
            #more complicated case, multiple calls to multi_flow required
            base=None
            if args.baseflow:
                base=args.baseflow[0]
            else:
                base=args.flows[0]
            for i in args.flows:
                if i!=base:
                    multi_flow(args,df,(base,i))


def single_flow(args,df):
    index=0
    start_index=-1

    vars=list(df.columns[1:])

    start_index=len(vars)
    index=decode_flow(*args.single_flow,vars)
    
    now = datetime.now()

    if index==-1:
         raise Exception("Flow index(es) are incorrect")
    x=[time.mktime(xx.timetuple()) for xx in list(df["TIME"])]

    xdif=[]
    for i in range(len(x)-1):
        xdif.append(x[i+1]-x[i])
    xdif.append(0.0)
    if len(x)==0:
        print("Empty data feed detected")
        return

    auxvars={}
    auxvars["TimeDiff"]=start_index

    df=df.set_index('TIME')

    if args.resample:
        df=PandasResample(df,len(df))

    df[vars[index]]*=args.mult1
    df[vars[index]]+=args.bias1

    if args.bias_list:
        for b in args.bias_list:
            f=b["from"] if b["from"] is not None else df["TIME"].min()
            t=b["to"] if b["to"] is not None else df["TIME"].max()
            df.loc[(df['TIME'] >= f) & (df['TIME'] <= t), vars[index]] *= b["mult1"]
            df.loc[(df['TIME'] >= f) & (df['TIME'] <= t), vars[index]] += b["bias1"]


    df.loc[:, "TimeDiff"] = xdif   #3

    td=[0]*len(xdif)
    for i in range(1,len(xdif)):
        td[i]=td[i-1]+xdif[i-1]/3600

    means=df[vars[index]].mean()
    stdevs=np.sqrt(df[vars[index]].var())
    #Output
    orig=None
    if args.output:
        orig=sys.stdout
        sys.stdout=open(args.output,"w")

    print("=======================================================================================")
    print("Results Summary (%d rows)" % (len(df.index)))
    print("---------------------------------------------------------------------------------------")
    print("a) Flowrate\n")
    print("%20s             Mean          Stdev"%(" "))
    print("%-26s     : %9.4f   %9.4f"%(vars[index]+" (m^3/hr)",means,stdevs))
    print("\n---------------------------------------------------------------------------------------")

    if args.use_adf:
        print("b) Augmented Dickey-Fuller test:")
        result = adfuller(df[vars[index]])
        print("%32s  %-20s"%(" ",vars[index]))
        print("%-32s: %-9.4f" % ('ADF Statistic',result[0]))
        print("%-32s: %-9.4f" % ('p-value',result[1]))

    later = datetime.now()
    difference = (later - now).total_seconds()
    print(f"Time taken:  {difference:.4f} sec")
    print("=======================================================================================")
    if orig:
        sys.stdout=orig

    if args.table:
        df.to_csv(args.table,float_format="%.4f")

    b_show=False

    if args.display_charts_3:
        plot3(df,args.display_charts_3,args,index,None)
        b_show=True
    if args.display_charts_4:
        plot4(df,args,index,None)
        b_show=True
    if args.display_charts_5:
        plot5(df,args,index,None)
        b_show=True

    if b_show:
        plt.show()


import numpy as np
import pandas as pd
from scipy.stats import linregress


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

def group_cosine(data, col1, col2, tolerance=0.5):
    """
    Groups the data into segments where column2 is linearly dependent on column1.
    
    Parameters:
    - data (pd.DataFrame): DataFrame with 'TIME', 'column1', and 'column2'.
    - tolerance (float): Maximum allowed absolute error in linear fit.
    
    Returns:
    - List[Tuple]: List of tuples where each tuple contains:
        (start_time, end_time, (slope, intercept)).
    """
    groups = []
    n = len(data)
    i = 0
    data = data.reset_index()
 
    while i < n:
        for j in range(i + 1, n + 1):
            x = data[col1].iloc[i:j].values
            y = data[col2].iloc[i:j].values
            
            # Perform linear regression
            try:
               sim=cosine_similarity(x,y)
            except Exception as ex:
                continue
            
            if sim<tolerance:
                # Break the group here and save the previous group
                if j - 1 > i:  # Ensure the group is non-empty
                    groups.append((
                        data['TIME'].iloc[i],
                        data['TIME'].iloc[j - 1],
                        sim
                    ))
                i = j - 1  # Update start of next group
                break
        else:
            # If we reach the end of the data without exceeding tolerance
            groups.append((
                data['TIME'].iloc[i],
                data['TIME'].iloc[j - 1],
                sim
            ))
            break
    
    data = data.set_index('TIME')

    return groups



def group_linear_relationship(data, col1, col2, tolerance=0.01):
    """
    Groups the data into segments where column2 is linearly dependent on column1.
    
    Parameters:
    - data (pd.DataFrame): DataFrame with 'TIME', 'column1', and 'column2'.
    - tolerance (float): Maximum allowed absolute error in linear fit.
    
    Returns:
    - List[Tuple]: List of tuples where each tuple contains:
        (start_time, end_time, (slope, intercept)).
    """
    groups = []
    n = len(data)
    i = 0
    data = data.reset_index()
 
    while i < n:
        for j in range(i + 1, n + 1):
            x = data[col1].iloc[i:j].values
            y = data[col2].iloc[i:j].values
            
            # Perform linear regression
            try:
                slope, intercept, _, _, _ = linregress(x, y)
            except Exception as ex:
                continue
            
            # Calculate predicted y-values and error
            y_pred = slope * x + intercept
            error = np.abs(y - y_pred).mean()
            
            if error > tolerance:
                # Break the group here and save the previous group
                if j - 1 > i:  # Ensure the group is non-empty
                    groups.append((
                        data['TIME'].iloc[i],
                        data['TIME'].iloc[j - 1],
                        (slope, intercept)
                    ))
                i = j - 1  # Update start of next group
                break
        else:
            # If we reach the end of the data without exceeding tolerance
            groups.append((
                data['TIME'].iloc[i],
                data['TIME'].iloc[j - 1],
                (slope, intercept)
            ))
            break
    
    data = data.set_index('TIME')

    return groups

def calculate_rmse_and_means(data, groups, col1, col2):
    """
    Calculate RMSE for column2 and means for each column for each group.

    Parameters:
    - data (pd.DataFrame): DataFrame with 'TIME', 'column1', and 'column2'.
    - groups (list): List of tuples where each tuple contains:
        (start_time, end_time, (slope, intercept)).

    Returns:
    - List[dict]: A list of dictionaries with RMSE and column means for each group.
    """
    results = []
    
    for start_time, end_time, (slope, intercept) in groups:
        # Extract the group data
        group_data = data.loc[start_time:end_time]
        x = group_data[col1].values
        y = group_data[col2].values
        
        # Predicted values for column2
        y_pred = slope * x + intercept
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        
        # Calculate means
        mean_column1 = group_data[col1].mean()
        mean_column2 = group_data[col2].mean()
        
        # Append results
        results.append({
            'start_time': start_time,
            'end_time': end_time,
            'rmse': rmse,
            'mean_column1': mean_column1,
            'mean_column2': mean_column2
        })
    
    return results

def add_predictions(data, column1, column2, groups):
    """
    Adds predicted values and differences to a DataFrame based on group linear relationships.

    Parameters:
    - data (pd.DataFrame): Original DataFrame with a time-based index.
    - column1 (str): Name of the independent variable (x).
    - column2 (str): Name of the dependent variable (y).
    - groups (list): List of tuples containing:
        (start_time, end_time, (slope, intercept)).

    Returns:
    - pd.DataFrame: A copy of the original DataFrame with two additional columns:
        - 'predicted_value': Predicted values for column2 based on the linear relationship.
        - 'predicted_diff': Difference between actual and predicted column2 values.
    """
    # Create a copy of the original DataFrame to avoid modifying it
    data_with_predictions = data.copy()

    # Initialize new columns
    data_with_predictions['predicted_value'] = np.nan
    data_with_predictions['predicted_diff'] = np.nan

    for start_time, end_time, (slope, intercept) in groups:
        # Extract the group data
        group_data = data.loc[start_time:end_time]

        # Calculate predicted values
        predicted_values = slope * group_data[column1] + intercept

        # Assign predicted values and differences
        data_with_predictions.loc[start_time:end_time, 'predicted_value'] = predicted_values
        data_with_predictions.loc[start_time:end_time, 'values_diff'] = (group_data[column1] - group_data[column2])
        data_with_predictions.loc[start_time:end_time, 'predicted_diff'] = (group_data[column1] - predicted_values)

    return data_with_predictions

def multi_flow(args,df,indexes=None):
    index1=0
    index2=1
    start_index=-1

    vars=list(df.columns[1:])

    start_index=len(vars)

    now = datetime.now()
    if not indexes and args.flows:
        index1=decode_flow(args.flows[0],vars)
        index2=decode_flow(args.flows[1],vars)
    elif indexes:
        index1=decode_flow(indexes[0],vars)
        index2=decode_flow(indexes[1],vars)

    if index1==index2 or index1==-1 or index2==-1:
         raise Exception("Flow index(es) are incorrect")
    x=[time.mktime(xx.timetuple()) for xx in list(df["TIME"])]

    xdif=[]
    for i in range(len(x)-1):
        xdif.append(x[i+1]-x[i])
    xdif.append(0.0)
    if len(x)==0:
        print("Empty data feed detected")
        return

    auxvars={}
    auxvars["Diff"]=start_index
    auxvars["TimeDiff"]=start_index+1
    auxvars["Trapez"]=start_index+2
    auxvars["Simps"]=start_index+3
    auxvars["Var1Trapez"]=start_index+4
    auxvars["Var1Simps"]=start_index+5
    auxvars["Var2Trapez"]=start_index+6
    auxvars["Var2Simps"]=start_index+7

    df[vars[index1]]*=args.mult1
    df[vars[index2]]*=args.mult2
    df[vars[index1]]+=args.bias1
    df[vars[index2]]+=args.bias2
    if args.bias_list:
        for b in args.bias_list:
            f=b["from"] if b["from"] is not None else df["TIME"].min()
            t=b["to"] if b["to"] is not None else df["TIME"].max()
            df.loc[(df['TIME'] >= f) & (df['TIME'] <= t), vars[index1]] *= b["mult1"]
            df.loc[(df['TIME'] >= f) & (df['TIME'] <= t), vars[index1]] += b["bias1"]
            df.loc[(df['TIME'] >= f) & (df['TIME'] <= t), vars[index2]] *= b["mult2"]
            df.loc[(df['TIME'] >= f) & (df['TIME'] <= t), vars[index2]] += b["bias2"]

    df=df.set_index('TIME')
    df["Diff"] = ((df[vars[index2]]  - df[vars[index1]]))  #2
    if args.abs_value:
        df["Diff"]=abs(df["Diff"])

    df.loc[:, "TimeDiff"] = xdif   #3
    df.loc[:, "Trapez"] = [0.0] * len(x) #4
    df.loc[:, "Simps"] = [0.0] * len(x)  #5
    df.loc[:, "Var1Trapez"] = [0.0] * len(x) #6
    df.loc[:, "Var1Simps"] = [0.0] * len(x)  #7
    df.loc[:, "Var2Trapez"] = [0.0] * len(x) #8
    df.loc[:, "Var2Simps"] = [0.0] * len(x)  #9

    for i in range(len(x)-1):
        #trapezoid calc
        df.iloc[i,auxvars["Trapez"]] = df.iloc[i,auxvars["TimeDiff"]]/3600*0.5*(df.iloc[i,auxvars["Diff"]]+df.iloc[i+1,auxvars["Diff"]])
        #simpson calc
        if i>=len(x)-2:
            df.iloc[i,auxvars["Simps"]]=df.iloc[i-1,auxvars["Simps"]]
        elif (i % 2)==0:
            df.iloc[i,auxvars["Simps"]] = (df.iloc[i,auxvars["TimeDiff"]]+df.iloc[i+1,auxvars["TimeDiff"]])/3600/6*(df.iloc[i,auxvars["Diff"]]+4*df.iloc[i+1,auxvars["Diff"]]+df.iloc[i+2,auxvars["Diff"]])
        else:
            df.iloc[i,auxvars["Simps"]]=df.iloc[i-1,auxvars["Simps"]]
        
    df["AccTrapezoid"] = df["Trapez"].cumsum()
    df["AccSimpson"] = df["Simps"].cumsum()/2

    td=[0]*len(xdif)
    for i in range(1,len(xdif)):
        td[i]=td[i-1]+xdif[i-1]/3600

    for i in range(len(x)-1):
        #trapezoid calc
        df.iloc[i,auxvars["Var1Trapez"]] = df.iloc[i,auxvars["TimeDiff"]]/3600*0.5*(df.iloc[i,index1]+df.iloc[i+1,index1])
        #simpson calc
        if i>=len(x)-2:
            df.iloc[i,auxvars["Var1Simps"]]=df.iloc[i-1,auxvars["Var1Simps"]]
        elif (i % 2)==0:
            df.iloc[i,auxvars["Var1Simps"]] = (df.iloc[i,auxvars["TimeDiff"]]+df.iloc[i+1,auxvars["TimeDiff"]])/3600/6*(df.iloc[i,index1]+4*df.iloc[i+1,index1]+df.iloc[i+2,index1])
        else:
            df.iloc[i,auxvars["Var1Simps"]]=df.iloc[i-1,auxvars["Var1Simps"]]
        
    df["AccVar1Trapez"] = df["Var1Trapez"].cumsum()
    df["AccVar1Simps"] = df["Var1Simps"].cumsum()/2


    for i in range(len(x)-1):
        #trapezoid calc
        df.iloc[i,auxvars["Var2Trapez"]] = df.iloc[i,auxvars["TimeDiff"]]/3600*0.5*(df.iloc[i,index2]+df.iloc[i+1,index2])
        #simpson calc
        if i>=len(x)-2:
            df.iloc[i,auxvars["Var2Simps"]]=df.iloc[i-1,auxvars["Var2Simps"]]
        elif (i % 2)==0:
            df.iloc[i,auxvars["Var2Simps"]] = (df.iloc[i,auxvars["TimeDiff"]]+df.iloc[i+1,auxvars["TimeDiff"]])/3600/6*(df.iloc[i,index2]+4*df.iloc[i+1,index2]+df.iloc[i+2,index2])
        else:
            df.iloc[i,auxvars["Var2Simps"]]=df.iloc[i-1,auxvars["Var2Simps"]]
        
    df["AccVar2Trapez"] = df["Var2Trapez"].cumsum()
    df["AccVar2Simps"] = df["Var2Simps"].cumsum()/2

    ts,pv=scipy.stats.ttest_ind(df[vars[index1]], df[vars[index2]], equal_var=False,alternative="two-sided")

    means=df[[vars[index1],vars[index2]]].mean()
    stdevs=np.sqrt(df[[vars[index1],vars[index2]]].var())
    res1=compute_part1(df[vars[index1]],df[vars[index2]])
    mse=calculate_mse(df[vars[index1]],df[vars[index2]])
    dt=None
    correlation=None
    ks=None
    if args.use_dtw:
        dt=dtw_distance(df[vars[index1]],df[vars[index2]])
    if args.use_correlation:
        correlation=df[vars[index1]].corr(df[vars[index2]])

    if args.use_ks:
        ks=ks_2samp(df[vars[index1]],df[vars[index2]])

    #Output
    orig=None
    if args.output:
        orig=sys.stdout
        sys.stdout=open(args.output,"w")

    print("=======================================================================================")
    print("Results Summary (%d rows)" % (len(df.index)))
    print("---------------------------------------------------------------------------------------")
    print("a) Flowrate\n")
    print("%20s             Mean          Stdev"%(" "))
    print("%-26s     : %9.4f   %9.4f"%(vars[index1]+" (m^3/hr)",means[0],stdevs[0]))
    print("%-26s     : %9.4f   %9.4f"%(vars[index2]+" (m^3/hr)",means[1],stdevs[1]))
    print("\n---------------------------------------------------------------------------------------")

    print("b) Flow Rate Difference\n")
    sums_diffs=df["TimeDiff"].sum()
    sums_rates=df[["Trapez","Simps"]].sum()
    sums_rates[1]/=2
    sums_rates_sp=[trapezoid_integration(list(df["Diff"]),td),simpson_integration(list(df["Diff"]),x=td)]
    print("%30s    Trapezoid     Simpson"%(" "))
    print("%-30s : %9.4f    %9.4f"%("Time Period (hr)",sums_diffs/3600,sums_diffs/3600))
    print("%-30s : %9.4f    %9.4f"%("Sum Volume (m^3)",sums_rates[0],sums_rates[1]))
    print("%-30s : %9.4f    %9.4f"%("Avg Flow rate (m^3/hr)",sums_rates[0]/sums_diffs*3600,sums_rates[1]/sums_diffs*3600))
    if args.use_full_output:
        print("")
        print("%30s    SciPy Trapz  SciPy Simps"%(" "))
        print("%-30s : %9.4f    %9.4f"%("Time Period (hr)",sums_diffs/3600,sums_diffs/3600))
        print("%-30s : %9.4f    %9.4f"%("Sum Volume (m^3)",sums_rates_sp[0],sums_rates_sp[1]))
        print("%-30s : %9.4f    %9.4f"%("Avg Flow rate (m^3/hr)",sums_rates_sp[0]/sums_diffs*3600,sums_rates_sp[1]/sums_diffs*3600))
    print("\n---------------------------------------------------------------------------------------")

    print("c) Accumulated Flow (Volume)\n")
    sums_vol1=df[["Var1Trapez","Var1Simps"]].sum()
    sums_vol1[1]/=2
    sums_vol2=df[["Var2Trapez","Var2Simps"]].sum()
    sums_vol2[1]/=2
    s=vars[index2]+"/"+vars[index1]
    #scipi implementation
    sums_vol1_sp=[trapezoid_integration(list(df[vars[index1]]),td),simpson_integration(list(df[vars[index1]]),x=td)]
    sums_vol2_sp=[trapezoid_integration(list(df[vars[index2]]),td),simpson_integration(list(df[vars[index2]]),x=td)]
    print("%30s   Trapezoid             Simpson"%(" "))
    print("%-30s : %9.4f            %9.4f"%(vars[index1]+" (m^3)",sums_vol1[0],sums_vol1[1]))
    print("%-30s : %9.4f            %9.4f"%(vars[index2]+" (m^3)",sums_vol2[0],sums_vol2[1]))
    print("%-30s : %9.4f            %9.4f"%(s,sums_vol2[0]/sums_vol1[0],sums_vol2[1]/sums_vol1[1]))
    if args.use_full_output:
        print("")
        print("%30s   SciPy Trapz           SciPy Simps"%(" "))
        print("%-30s : %9.4f            %9.4f"%(vars[index1]+" (m^3)",sums_vol1_sp[0],sums_vol1_sp[1]))
        print("%-30s : %9.4f            %9.4f"%(vars[index2]+" (m^3)",sums_vol2_sp[0],sums_vol2_sp[1]))
        print("%-30s : %9.4f            %9.4f"%(s,sums_vol2_sp[0]/sums_vol1_sp[0],sums_vol2_sp[1]/sums_vol1_sp[1]))
    print("\n---------------------------------------------------------------------------------------")

    print("d) Fit Test\n")
    print("%-30s : %.4f"%("MSE",mse))
    print("%-30s : %.4f"%("T-test p-value",pv))

    if args.use_correlation:
        print("%-30s : %.4f"%("Correlation",correlation))
    if args.use_dtw:
        print("%-30s : %.4f"%("Dynamic time warping",dt))

    if args.use_ks:
        print("%-30s : %.4f\t\t%.4f"%("KS statistics and p-value",ks[0],ks[1]))
    if args.use_wd:
        wd=wasserstein_distance(df[vars[index1]],df[vars[index2]])
        print("%-30s : %.4f"%("Wasserstein distance",wd))
    if args.use_coint:
        coin=coint(df[vars[index1]],df[vars[index2]])
        print("%-30s : %.4f"%("Cointegration p-value",coin[1]))
    print("\n---------------------------------------------------------------------------------------")
    if args.use_adf:
        print("e) Augmented Dickey-Fuller test:")
        result = [adfuller(df[vars[index1]]),adfuller(df[vars[index2]]),adfuller(df["Diff"])]
        print("%32s  %-20s  %-20s %-20s"%(" ",vars[index1],vars[index2],"Diff"))
        print("%-32s: %-9.4f             %-9.4f            %-9.4f" % ('ADF Statistic',result[0][0],result[1][0],result[2][0]))
        print("%-32s: %-9.4f             %-9.4f            %-9.4f" % ('p-value',result[0][1],result[1][1],result[2][1]))
        print('Critical Values:')
        kv=[]
        for r in result:
            v1=[]
            for key, value in r[4].items():
                v1.append((key,value))
            kv.append(v1)
        for i in range(len(kv[0])):
            key=kv[0][i][0]
            v1=kv[0][i][1]
            v2=kv[1][i][1]
            v3=kv[2][i][1]
            print("%-32s: %-9.4f             %-9.4f            %-9.4f" % (key,v1,v2,v3))
    
    b_show=False
    if args.cluster_data:
        print("\n---------------------------------------------------------------------------------------")
        groups = group_linear_relationship(df, vars[index1],vars[index2], tolerance=args.cluster_data)
        res = calculate_rmse_and_means(df, groups, vars[index1],vars[index2])
        i=1
        print(" Group      Start                 End               Intercept     Slope          RMSE")
        for group in groups:
            st=group[0].strftime("%Y-%m-%d %H:%M:%S")
            en=group[1].strftime("%Y-%m-%d %H:%M:%S")
            intr=group[2][1]
            slope=group[2][0]
            rmse=res[i-1]["rmse"]
            print(f"  {i:4d}  {st:14} {en:14}  {intr:+9.4f}        {slope:+9.4f}    {rmse:9.4f}") 
            i+=1
        if args.cluster_graph:
            newdf=add_predictions(df,vars[index1],vars[index2],groups)
            plot_cluster(newdf,args,index1,index2)
            b_show=True
        groups = group_cosine(df, vars[index1],vars[index2], tolerance=0.5)
        print("*************** Similarities ****************")
        i=1
        print(" Group      Start                 End                Similadrity")
        for group in groups:
            st=group[0].strftime("%Y-%m-%d %H:%M:%S")
            en=group[1].strftime("%Y-%m-%d %H:%M:%S")
            sim=group[2]
            print(f"  {i:4d}  {st:14} {en:14}  {sim:+9.4f}") 
            i+=1

    if args.calibrate:
        calibrate_and_chart(df,vars[index1],vars[index2],args)
        b_show=True

    if args.match:
        dtw=None
        if args.use_dtw:
            dtw=dt
        m=check_match(df[vars[index1]],df[vars[index2]],df['TimeDiff'][1],dtw=dtw,use_logistic=False)
        print("Match probability: ",m)
        df['Diff1']=df[vars[index1]].diff().fillna(0)
        df['Diff2']=df[vars[index2]].diff().fillna(0)
        #g=grangercausalitytests(df[[vars[index1], vars[index2]]], maxlag=[4])
        #g=grangercausalitytests(df[['Diff1', 'Diff2']], maxlag=[4])
        #print(g)

    later = datetime.now()
    difference = (later - now).total_seconds()
    print(f"Time taken:  {difference:.4f} sec")
    print("=======================================================================================")
    if orig:
        sys.stdout=orig

    if args.table:
        df.to_csv(args.table,float_format="%.4f")



    if args.display_charts_1:
        plot1(df,args,index1,index2)
        b_show=True
    if args.display_charts_2:
        plot2(df,args,index1,index2)
        b_show=True
    if args.display_charts_3:
        plot3(df,args.display_charts_3,args,index1,index2)
        b_show=True
    if args.display_charts_4:
        plot4(df,args,index1,index2)
        b_show=True
    if args.display_charts_5:
        plot5(df,args,index1,index2)
        b_show=True

    if b_show:
        plt.show()



 
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process distance/elevation files')
    parser.add_argument('--input', '-i', required=True, dest='input', type=str, help='Input file name')
    parser.add_argument('--output', '-o', required=False, dest='output', type=str, default=None, help='Output file name')
    parser.add_argument('--config', '-c', required=False, dest='config', type=str, default=None, help='Config file name')
    parser.add_argument('--table', '-b', required=False, dest='table', type=str, default=None, help='Table output file name')
    parser.add_argument('--from', '-f', dest='frm', nargs='?', help='Filter records from this time')
    parser.add_argument('--to', '-t', dest='to', nargs='?', help='Filter records from to time')
    parser.add_argument('--dc', '-dc', dest='display_columns', action='store_true', help='Display coulmns list')
    parser.add_argument('--graph1', '-g1', dest='display_charts_1', action='store_true', help='Display plot of integration results')
    parser.add_argument('--addgroup', '-ag', dest='add_grouping', action='store_true', help='Display grouping plot of the results')
    parser.add_argument('--graph2', '-g2', dest='display_charts_2', action='store_true', help='Display plot of differences')
    parser.add_argument('--graph3', '-g3', dest='display_charts_3', type=int, default=0, help='Display plot of variables (1,2 or 3(both)')
    parser.add_argument('--graph4', '-g4', dest='display_charts_4', action='store_true', default=0, help='Display histograms')
    parser.add_argument('--graph5', '-g5', dest='display_charts_5', action='store_true', default=0, help='Display FFT plots')
    parser.add_argument('--grid', '-d', dest='display_grid', action='store_true', help='Display grid lines')
    parser.add_argument('--abs', '-a', dest='abs_value', action='store_true', help='Use absolute difference')
    parser.add_argument('--full_output', '-fo', dest='use_full_output', action='store_true', help='Provide extended output')
    parser.add_argument('--resample', '-r', dest='resample', action='store_true', help='Resample data to create regular time intervals')
    parser.add_argument('--bias1', '-b1', required=False, dest='bias1', type=float, default=0.0, help='Add bias to the first variable')
    parser.add_argument('--bias2`', '-b2', required=False, dest='bias2', type=float, default=0.0, help='Add bias to the second variable')
    parser.add_argument('--multiplier1', '-m1', required=False, dest='mult1', type=float, default=1.0, help='Multiply by the first variable')
    parser.add_argument('--multiplier2', '-m2', required=False, dest='mult2', type=float, default=1.0, help='Multiply by the second variable')
    parser.add_argument('--dtw', '-dtw', dest='use_dtw', action='store_true', help='Use dynamic time warping')
    parser.add_argument('--wd', '-wd', dest='use_wd', action='store_true', help='Use Wasserstein distance')
    parser.add_argument('--ks', '-ks', dest='use_ks', action='store_true', help='Use Kolmogorov-Smirnov test')
    parser.add_argument('--adf', '-adf', dest='use_adf', action='store_true', help='Use augmented Dickey-Fuller test')
    parser.add_argument('--cointegration', '-coint', dest='use_coint', action='store_true', help='Use Cointegration test')
    parser.add_argument('--corr', '-corr', dest='use_correlation', action='store_true', help='Display correlation')
    parser.add_argument('--singleflow', '-sf', required=False, dest='single_flow', nargs=1, default=None, help='Single flow to process')
    parser.add_argument('--flows', '-fl', required=False, dest='flows', nargs='*', default=None, help='Flows to compare')
    parser.add_argument('--baseflow', '-bfl', required=False, dest='baseflow', nargs=1, default=None, help='Base flow to compare other flows to')
    parser.add_argument('--match', '-m', required=False, dest='match', action='store_true', help='Check match')
    parser.add_argument('--cluster', '-cl', dest='cluster_data', type=float, default=None, help='Cluster data with given threshold')
    parser.add_argument('--clustergraph', '-cg', dest='cluster_graph',action='store_true', help='Graph of clusters. Only valid if cluster is requested')
    parser.add_argument('--calibrate', '-fmc', dest='calibrate',action='store_true', help='Calibrating chart')
    parser.add_argument('--synchronize', '-sa', dest='synchronize_axis', action='store_true', help='Synchrinize Y axis for the calibration display')
    parser.add_argument('--syncindex', '-si', dest='synchronize_index', action='store_true', help='Synchrinize indexes with the beginning of the pipe start')
    parser.add_argument('--useopt', '-opt', dest='useopt', action='store_true', help='Use optimal value calculaton for the calibration display')
    parser.add_argument('--uselsq', '-lsq', dest='uselsq', action='store_true', help='Use LSQ for the calibration display')
    parser.add_argument('--CalibrationOutput', '-co', required=False, dest='calibration_output', type=str, default=None, help='Calibration output file name')
    parser.add_argument('--CalibrationSamplePeriodicityInMinutes', '-caspm', dest='CalibrationSamplePeriodicityInMinutes',type=int,default=5, help='CalibrationSamplePeriodicityInMinutes for calibration')
    parser.add_argument('--CalibrationPeriodicityInHours', '-casph', dest='CalibrationPeriodicityInHours',type=float,default=1, help='CalibrationPeriodicityInHours for calibration')
    parser.add_argument('--MinimumFlowToSample', '-camf', dest='MinimumFlowToSample',type=int,default=500, help='MinimumFlowToSample for calibration')
    parser.add_argument('--NumberOfSamplesForCalibration', '-cans', dest='NumberOfSamplesForCalibration',type=int,default=60, help='NumberOfSamplesForCalibration for calibration')
    parser.add_argument('--CalibrationDelayAfterRestartInHours', '-cad', dest='CalibrationDelayAfterRestartInHours',type=int,default=4, help='CalibrationDelayAfterRestartInHours for calibration')
    parser.add_argument('--CalibrationOffsetSamplesFromCurrentTime', '-cao', dest='CalibrationOffsetSamplesFromCurrentTime',type=int,default=0, help='CalibrationOffsetSamplesFromCurrentTime for calibration')
    args = parser.parse_args()
    if args.display_charts_3!=0 and (args.display_charts_3<1 or args.display_charts_3>3):
        print("Error: graph3 value should be from 1 to 3")
        sys.exit(-1)
    if args.cluster_graph and not args.cluster_data:
        print("Error: cluster graph is only accessible if cluster data is requesed")
        sys.exit(-1)
    try:
        main(args)
    except Exception as ex:
        print(ex) 

