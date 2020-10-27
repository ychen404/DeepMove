import pandas as pd
import collections
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

def split_dataframe(dataframe):
    df_split = {}
    for i in range(len(dataframe.time)):
        if dataframe.time[i].split(' ')[0] not in df_split:
            # Create a new array in this slot
            df_split[dataframe.time[i].split(' ')[0]] = [dataframe.pid[i]]
        else:
            # Append the pid to this slot if exists
            df_split[dataframe.time[i].split(' ')[0]].append(dataframe.pid[i])
    
    return df_split


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    # ax.set_xticks(np.arange(data.shape[1]))
    # ax.set_yticks(np.arange(data.shape[0]))
    
    # ... and label them with the respective list entries.
    # ax.set_xticklabels(col_labels)
    # ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    
    # ax.tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #          rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    # ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    # ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

df = pd.read_csv('tweets-cikm-sample-march.txt', delimiter='\x01')

df_full = pd.read_csv('tweets-cikm-w-header.txt', delimiter='\x01')
print(f"Unique loc: {len(df_full.pid.unique())}")

# Split the existing column to two
# df[['date','timestamp']] = df_full.time.str.split(' ', expand=True)
# print(f"Another Unique date: {len(df.date.unique())}")
# print(df.head())

# full_unique_date = []
# for i, r in df_full.iterrows():
#     date = r.time.split(' ')[0]
#     if date not in full_unique_date:
#         full_unique_date.append(date)
# print(f"Unique date: {len(full_unique_date)}")

def count_freq(dateframe):
    date_test = []
    loc_array = []
    count_array = []
    date_array = []
    date_array_nodup = []
    date_pid_count = {}
    date_pid_count = {}
    df_split = split_dataframe(dateframe)
    for k, v in df_split.items():
        print(f"Summary of {k}")
        
        item = [item for item, count in collections.Counter(df_split[k]).items() if count > 2]
        count = [count for item, count in collections.Counter(df_split[k]).items() if count > 2]
        print(f"key={k}, len={len(item)}")
        print([[item, count] for item, count in collections.Counter(df_split[k]).items() if count > 2])
        
        if k not in date_pid_count.items():
            date_pid_count[k] = dict(zip(item, count))
        
        date_array_nodup.append(k)
        for i in range(len(item)):
            if item[i] not in loc_array:
                loc_array.append(item[i])
            date_array.append(k)
        
        count = [count for item, count in collections.Counter(df_split[k]).items() if count > 2]
        for i in range(len(count)):
            count_array.append(count[i])
        
        for i in range(len(item)):
            date_test.append(k)
    print(f"The length of item is {len(item)}")
    print(f"The length of count is {len(count)}")
    return loc_array, date_array, date_array_nodup

unique_time = []
unique_loc = []
for i, r in df_full.iterrows():
    if r.time.split(' ')[0] not in unique_time:
        unique_time.append(r.time.split(' ')[0])
    if r.pid not in unique_loc:
        unique_loc.append(r.pid)

d_index = [i for i in range(len(unique_time))]

num_rows = len(unique_loc)
num_col = len(unique_time)

# num_rows = len(df_full.pid.unique())
# num_col = len(full_unique_date)

density_map = np.zeros(shape=(num_rows, num_col))
density_map = density_map.astype(int)
print(density_map.shape)

# exit("testtest")

date_index = [i for i in range(len(unique_time))]
loc_index = [i for i in range(len(unique_loc))]

date_index_dict = dict(zip(unique_time, date_index))
print(date_index_dict)
loc_index_dict = dict(zip(unique_loc, loc_index))

for index, row in df_full.iterrows():
    user_added = {}
    date = row.time.split(' ')[0]
    user_added[date] = ['None']
    if row.uid not in user_added[date]:
        density_map[loc_index_dict[row.pid]][date_index_dict[date]] += 1
    else:
        pass

print(density_map[0])
print(density_map.shape)
harvest = np.array([[0, 3, 0, 0, 0, 0, 0],
                    [0, 0, 3, 0, 0, 0, 0],
                    [0, 0, 0, 3, 0, 0, 0],
                    [0, 0, 0, 3, 5, 0, 0],
                    [0, 0, 0, 0, 0, 5, 0],
                    [0, 0, 0, 0, 0, 3, 0],
                    [0, 0, 0, 0, 0, 3, 0],
                    [0, 0, 0, 0, 0, 3, 0],
                    [0, 0, 0, 0, 0, 0, 6]])
# print(harvest)
# print(harvest.shape)
# print(loc_index)
# print(date_index)
# exit()

density_map_small = density_map[0:325, 0:325]
print(density_map_small.shape)
# exit()
fig, ax = plt.subplots(figsize=(300,300))

# loc_array, date_array, date_array_nodup = count_freq(df)
# print(f"date_array_nodup = {date_array_nodup}")
# print(f"loc_array = {loc_array}")
# print(loc_index)

loc_index_str = []
date_index_str = []
for e in loc_index:
    loc_index_str.append(str(e))

for e in date_index:
    date_index_str.append(str(e))

# print(date_index)
# print(loc_index_str, date_index_str)
# exit("hey, exiting")

start = time.time()
im, cbar = heatmap(density_map_small, loc_index_str[0:325], date_index_str, ax=ax,
                   cmap="Reds", cbarlabel="Number of Visits")
# im, cbar = heatmap(harvest, loc_array, date_array_nodup, ax=ax,
#                    cmap="Reds", cbarlabel="Number of Visits")
texts = annotate_heatmap(im, valfmt="{x:d} ")
fig.tight_layout()
plt.savefig('heatmap.pdf', bbox_inches='tight', dpi=80)
end = time.time()
print(f"Time elapse {end - start} seconds")