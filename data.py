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
            if re.search("[a-df-z]", line) is not None or line.strip() == "":
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
    Checks if a file is a spinmob binary file. If so reads it as such.
    Otherwise, read it as a CSV. In the future this could be extended
    to identify other file types.

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
        if true, returns the pandas dataframe, else , by default False
    head : int, optional
        the line that contains the column names, by default will read the file for a line of data,
        and backtrack from there.
    delim : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """
    if head is None:
        head = get_header(file)-1
        if head < 0:
            head = None
    data = pd.read_csv(file, header=head+1, sep=delim, **kwargs)

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
