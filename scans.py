import numpy as np
import os
import spinmob as sp
import matplotlib.pyplot as plt
from . import data

def de_interleave(img_in, dedouble = True):
    img_size = img_in.shape
    
    if dedouble == True:
        img_1_out = np.empty(shape = img_size)
        img_2_out = np.empty(shape = img_size)
        
        for idx, row in enumerate(img_in):
            if np.mod(idx,2)==0:
                img_1_out[idx]=row
                if idx+1 < img_size[0]:
                    img_1_out[idx+1]=row
            else:
                img_2_out[idx-1]=row
                img_2_out[idx]=row
        
    else:
        img_1_out = np.empty(shape = [img_size[0], img_size[0]/2])
        img_2_out = np.empty(shape = [img_size[0], img_size[0]/2])
                
        for idx, row in enumerate(img_in):
            if np.mod(idx,2)==0:
                img_1_out[int(idx/2)]=row
            else:
                img_2_out[int(idx/2)]=row
                
    return img_1_out, img_2_out

def plot_scan_raw(xpts,ypts,data,dedouble,converted=True,vmin=None,vmax=None,levels=30,title="",**kwargs):
    cmap = "viridis"
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)
    levels = np.linspace(vmin, vmax, levels)
    
    #data = data.transpose()
    dx = np.mean(np.diff(xpts))
    xmesh = xpts - dx
    xmesh = np.append(xmesh, xpts[-1] + dx)
    dy = np.mean(np.diff(ypts))
    ymesh = ypts - dy
    ymesh = np.append(ymesh,ypts[-1] + dy)
    X,Y = np.meshgrid(xmesh,ymesh)
    
    if dedouble:
        img1, img2 = de_interleave(data,dedouble)
        fig, axes = plt.subplots(1,2,figsize=(5.5,2.5),sharey=True,sharex=True)
        plt.title(title)
        im1 = axes[0].pcolormesh(X,Y,img1,cmap=cmap,vmin=vmin,vmax=vmax)
        im2 = axes[1].pcolormesh(X,Y,img2,cmap=cmap,vmin=vmin,vmax=vmax)
        fig.subplots_adjust(left=0.07,right=0.85,bottom=0.15)
        cbar_ax = fig.add_axes([0.88,0.15,0.025,0.7])
        fig.colorbar(im1,cax=cbar_ax,extend='both')
        for ax in axes:
            ax.set_xlim([min(xmesh),max(xmesh)])
            ax.set_ylim([min(ymesh),max(ymesh)])
        axes[0].set_title("Forward Scan")
        axes[1].set_title("Reverse Scan")
        if converted:
            axes[0].set_xlabel("X Position (um)")
            axes[1].set_xlabel("X Position (um)")
            axes[0].set_ylabel("Y Position (um)")
        else:
            axes[0].set_xlabel("X Effective (V)")
            axes[1].set_xlabel("X Effective (V)")
            axes[0].set_ylabel("Y Effective (X)")
    else:
        fig, axes = plt.subplots(1,1,figsize=(2.5,2.5))
        plt.title(title)
        im = axes[0].pcolormesh(X,Y,data,cmap=cmap,vmin=vmin,vmax=vmax)
        fig.subplots_adjust(left=0.07,right=0.85,bottom=0.15)
        cbar_ax = fig.add_axes([0.88,0.15,0.025,0.7])
        fig.colorbar(im,cax=cbar_ax,extend='both')
        axes[0].set_xlim([min(xpts),max(xpts)])
        if converted:
            axes[0].set_xlabel("X Position (um)")
            axes[0].set_ylabel("Y Position (um)")
        else:
            axes[0].set_xlabel("X Effective (V)")
            axes[0].set_ylabel("Y Effective (V)")
    return fig

def plot_scan(filename,title=None, convert=True, **kwargs):
    scan = data.read(filename)
    if title is None:
        title = filename
    return plot_scan_data(scan, convert=True, **kwargs)
    
def plot_scan_data(scan, convert=True, **kwargs):
    dedouble = True if scan['scan_type'] == 0 else False
    if convert:
        try:
            scan['xs']
        except KeyError:
            convert_units(scan)
        return plot_scan_raw(scan['xs'],scan['ys'],scan['data'],dedouble,converted=True,**kwargs)

    return plot_scan_raw(scan['Vxs'],scan['Vys'],scan['data'],dedouble,converted=False,**kwargs)

def convert_units(scan, pz_gain=None, gv_gain=None, **kwargs):
    scan_type = scan['scan_type']
    xs = scan['Vxs']
    ys = scan['Vys']

    # Piezo scan, conversion is amplifier gain (V/V) times piezo sensitivity (nm/V) converted to um.
    if pz_gain is None:
        pz_gain = -17 * 77 / 1000
    if scan_type == 0:
        if pz_gain < 0:
            scan['data'] = np.flip(scan['data'])
        scan['xs'] = xs * pz_gain
        scan['ys'] = ys * pz_gain
    # Galvo scan, conversion is in um/V
    if gv_gain is None:
        gv_gain = 117
    if scan_type in [1,2]:
        if gv_gain < 0:
            scan['data'] = np.flip(scan['data'])
        scan['xs'] = xs * gv_gain
        scan['ys'] = ys * gv_gain
    # Objective scan, y axis is position in um/12000, x axis is galvo, same as above.
    # Negative values on the objective mean increasing height, so flip the sign for plotting.
    if scan_type == 3:
        if gv_gain < 0:
            scan['data'] = np.flip(scan[data])
        else:
            scan['data'] = np.flip(scan[data],axis=0)
        scan['xs'] = xs * gv_gain
        scan['ys'] = ys * -12000 # Not sure why this is the factor, but it is