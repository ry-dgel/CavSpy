import numpy as np
import pandas as pd
import spinmob as sp
import re

################
# Lock-In Data #
################
# Figures out the delimiter used in a csv style file.
def get_delim(file):
    with open(file, "r") as f:
        # Reads file line by line, so that only one line needs to be read
        for _, line in enumerate(f):
            # ignore comments and header if present
            if re.search("[a-df-z]", line) is not None or line.strip() == "":
                continue
            
            for delim in [",", ";", ":", "\t"]:
                if delim in line:
                    return delim

    print("No delimiters found")
    return ""

# Determine how many lines of header are in a file.
def get_header(file):
    with open(file, "r") as f:
        i = 0
        for _, line in enumerate(f):
            if re.search("[a-df-zA-DF-Z]", line) is not None or line.strip() == "":
                i += 1
            else:
                break
                
        return i


# Read data from csv file. 
def read_csv_old(file, names=True, delim=None, head=None):
    if delim is None:
        delim = get_delim(file)
    if head is None:
        head = get_header(file)
    if names:
        return np.genfromtxt(file, names=True, delimiter=delim,skip_header=head-1)
    else:
        return np.genfromtxt(file, delimiter=delim, skip_header=head)

def read(filename, **kwargs):
    """
    Checks if a file is of the following type, and uses the appropriate function
    to read it:
     - Spinmob Binary
     - Picoharp 300 Histogram
     - Custom scan file from labview

    Otherwise, reads the file as a csv.

    Parameters
    ----------
    filename : string
        Path to the file to read

    kwargs :
        key word arguments to pass to whatever file reading function gets called.

    Returns
    -------
    object
        Some sort of data container, depending on the filetype and kwargs
    """
    try:
        with open(filename, 'rb') as f:
            if f.read(14).decode('utf-8') == 'SPINMOB_BINARY':
                return read_sp_bin(filename, **kwargs)
        with open(filename, 'r') as f:
            if f.read(33) == "Scan data, number of header lines":
                return read_scan(filename, **kwargs)
        with open(filename, 'r') as f:
            if f.read(29) == "#PicoHarp 300  Histogram Data":
                return read_tcspc(filename, **kwargs)
        with open(filename, 'r') as f:
            if f.read(4) == "date":
                return read_michael_scan(filename, **kwargs)
    except UnicodeDecodeError:
        pass

    return read_csv(filename, **kwargs)

def read_csv(file, df=False, head=None, delim=None, **kwargs):
    """
    Read a csv file using panda's read_csv(). Can either return a dataframe,
    or a numpy array using panda's to_numpy() function.

    Parameters
    ----------
    file : string
        the path to the file. Pandas can also accept urls if that's helpful.
    df : bool, optional
        if True, returns the pandas dataframe, else convert to a numpy array, by default False
    head : int, optional
        the line that contains the column names, by default will read the file for a line of data,
        and backtrack from there.
    delim : str, optional
        The character used to deliminate columns of data, by default will let pandas figure it out.

    Returns
    -------
    [type]
        [description]
    """
    if head is None:
        head = get_header(file)-1
        if head < 0:
            head = None
    data = pd.read_table(file, header=head, sep=delim, **kwargs)

    if df:
        return data
    else:
        return data.to_numpy()

def read_sp_bin(file):
    return sp.data.load(file)

# Get data from csv file exported from lock in.
def unpack(filename, fields = [], delim=None):
    chunks = {}
    with open(filename) as f:
        # Skip header line
        next(f)
        for line in f:
            # Each line has form:
            # chunk;timestamp;size;fieldname;data0;data1;...;dataN
            if delim is None:
                delim = get_delim(filename)
            entries = line.split(delim)

            chunk = entries[0]
            # If this is a new chunk, add to chunks dictionary.
            if chunk not in chunks.keys():
                chunks[chunk] = {}
            # Use chunk dictionary for data storage
            # This separates the runs
            dic = chunks[chunk]

            fieldname = entries[3]
            data = np.array([float(x) for x in entries[4:]])

            # Add named dataset to dictionary for each desired fieldname
            # If no fieldnames specified in fields, just return all.
            if fieldname in fields or len(fields) == 0:
                if fieldname not in dic.keys():
                    dic[fieldname] = data
                else:
                    dic[fieldname] = np.concatenate((dic[fieldname], data))

    data_chunks = list(chunks.values())
    return data_chunks

def read_scan(filename, **kwargs):
    with open(filename, 'r') as scanfile:
        head = scanfile.readline()
        res = re.split(',|:', head)
        head = int(res[2])
    header = pd.read_csv(filename, nrows=head, header=None, sep=':', engine='python')
    # Massage loaded data into nicer dataframe
    header = header.transpose()
    header.columns = header.iloc[0]
    header = header.drop(header.index[0])
    try:
        header = header.drop('Scan data, number of header lines ',axis=1)
    except KeyError:
        try:
            header = header.drop('Scan data, number of header lines',axis=1)
        except KeyError:
            pass
            
    scan = header.to_dict(orient='records')[0]
    
    # Replace some annoying header names
    try:
        scan['Ystart (V)'] = scan.pop('Ystart ( V)')
    except KeyError:
        pass
    scan['scan_type'] = scan.pop('Scan type (0=triangle, 1=raster, 2=raster,slow return, 3=objective)', None)
    if scan['scan_type'] is None:
        scan['scan_type'] = scan.pop('Scan type (0=triangle, 1=raster, 2=raster,slow return)', None)

    if scan['scan_type'] == 5:
        data = load_3d_scan(filename, head)
    else:
        data = load_2d_scan(filename, head)

    scan.update({'data' : data})
    xs = np.linspace(float(scan['Xstart (V)']), float(scan['Xstop (V)']), int(scan['Xpoints']))
    ys = np.linspace(float(scan['Ystart (V)']), float(scan['Ystop (V)']), int(scan['Ypoints']))
    scan['Vxs'] = xs
    scan['Vys'] = ys
    try:
        zs = np.linspace(float(scan['Zstart (V)']), float(scan['Zstop (V)']), int(scan['Zpoints']))
        scan['Vzs'] = zs
    except KeyError:
        pass
    return scan

def read_michael_scan(filename, kwargs):
    data = sp.data.load(filename)
    header = data.headers
    scan = {'scan_type' : 2,
            'Xstart (V)' : header['Vx_min'],
            'Xstop (V)' : header['Vx_max'],
            'Ystart (V)' : header['Vy_min'],
            'Ystop (V)' : header['Vy_max'],
            'Xpoints' : header['Nx'],
            'Ypoints' : header['Ny']}
    scan['data'] = np.array(data)
    xs = np.linspace(float(scan['Xstart (V)']), float(scan['Xstop (V)']), int(scan['Xpoints']))
    ys = np.linspace(float(scan['Ystart (V)']), float(scan['Ystop (V)']), int(scan['Ypoints']))
    scan['Vxs'] = xs
    scan['Vys'] = ys
    print(scan)
    return scan

def load_2d_scan(filename, head):
    data = pd.read_csv(filename, skiprows=head, header=None)
    return data.to_numpy()

def load_3d_scan(filename, head=0):
    with open(filename,'r') as f:
        for _ in range(head):
            next(f)
        data = f.read()

    pages = data.split('\n\n')
    pages = pages[:-1]
    processed_pages = []
    for page in pages:
        page = page.split('\n')[1:]
        page = list(map(lambda line: line.split(','), page)) 
        page = np.array(page,dtype=np.float32)
        processed_pages.append(page)
    data = np.dstack(processed_pages)
    return np.swapaxes(data,1,2)

def read_tcspc(filename, cntr_time=True, **kwargs):
    """ Sample File with Header:

    #PicoHarp 300  Histogram Data           2021-03-22 04:00:21 PM
    #channels per curve
    65536
    #display curve no.
    0   
    #memory block no.
    0   
    #ns/channel
    0.0040  
    #counts
    0   
    0
    .
    .
    .   
    """
    with open(filename, 'r') as file:
        head = [file.readline() for _ in range(10)]
    chns = int(head[2])
    ns_per_chn = float(head[8])
    counts = pd.read_table(filename, sep=',', header=10, names=["counts"])
    times = np.arange(0,(chns-1)*ns_per_chn,ns_per_chn)
    if cntr_time:
        times += ns_per_chn/2
    counts.insert(0,"times",times)
    return counts