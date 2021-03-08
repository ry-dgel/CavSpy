import numpy as np
import os
import spinmob as sp

def de_interleave(img_in, dedouble = True):
    img_size = img_in.shape
    
    if dedouble == True:
        img_1_out = np.empty(shape = img_size)
        img_2_out = np.empty(shape = img_size)
        
        for idx, row in enumerate(img_in):
            if np.mod(idx,2)==0:
                img_1_out[idx]=row
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

def plot_scan_data(xpts,ypts,data,dedouble=False,convertunit=True,vmin=None,vmax=None,levels=30,title="",**kwargs):
    cmap = "viridis"
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)
    levels = np.linspace(vmin, vmax, levels)
    
    data = data.transpose()
    dx = np.mean(np.diff(xpts))
    xmesh = xpts - dx
    xmesh = np.append(xmesh, xpts[-1] + dx)
    dy = np.mean(np.diff(ypts))
    ymesh = ypts - dy
    ymesh = np.append(ymesh,ypts[-1] + dy)
    if convertunit:
        xmesh *= 16 * 66 / 1000 # V/V * nm/V / nm/um
        ymesh *= 16 * 66 / 1000
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
            ax.set_xlim([min(xpts),max(xpts)])
            ax.set_ylim([min(ypts),max(ypts)])
        axes[0].set_title("Forward Scan")
        axes[1].set_title("Reverse Scan")
        if convertunit:
            axes[0].set_xlabel("X Position (um)")
            axes[1].set_xlabel("X Position (um)")
            axes[0].set_ylabel("Y Position (um)")
        else:
            axes[0].set_xlabel("X Effective (V)")
            axes[1].set_xlabel("X Effective (V)")
            axes[0].set_ylabel("Y Effective (X)")
    else:
        fig, ax = plt.subplots(1,1,figsize=(2.5,2.5))
        plt.title(title)
        im = axes[0].pcolormesh(X,Y,data,cmap=cmap,vmin=vmin,vmax=vmax)
        fig.subplots_adjust(left=0.07,right=0.85,bottom=0.15)
        cbar_ax = fig.add_axes([0.88,0.15,0.025,0.7])
        fig.colorbar(im,cax=cbar_ax,extend='both')
        ax.set_xlim([min(xpts),max(xpts)])
        if convertunit:
            axes[0].set_xlabel("X Position (um)")
            axes[0].set_ylabel("Y Position (um)")
        else:
            ax.set_xlabel("X Effective (V)")
            ax.set_ylabel("Y Effective (V)")
    return fig

def plot_scan(filename,title=None, **kwargs):
    header = sp.data.load(filename,delimiter=':').headers
    xs = np.linspace(*[header[name] for name in ['Xstart (V)', 'Xstop (V)', 'Xpoints']])
    ys = np.linspace(*[header[name] for name in ['Ystart ( V)', 'Ystop (V)', 'Ypoints']])
    header = 0
    if title is None:
        title = filename
    data = np.array(sp.data.load(filename,delimiter=',')[:])
    
    return plot_scan_data(xs,ys,data,title=title,**kwargs)