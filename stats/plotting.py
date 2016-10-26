import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
# import matplotlib.figure as mfig
import numpy as np
from math import ceil
import collections
import itertools
import seaborn as sns


def generate_csv(addr, x_axis, data, aux_labels=None):
    """Generates a CSV file with data for plotting a chart. (DEPRECATED)

    addr -- address where the file should be saved.
    x_axis -- x axis value of each item in data
    data -- list of y axis values (or lists of values), each value associated
            to a value in x_axis.
    aux_labels (optional) -- in case lists of values are used in data, if these
                             lists have the same number of items each, they can
                             also be labeled. The labels in aux_labels should
                             be indexed in association with the indexes of the
                             lists inside data."""
    f = open(addr, "w")

    if aux_labels:
        f.write(";".join(aux_labels) + "\n")

    if type(data[0]) is list:
        for i in range(len(x_axis)):
            f.write(";".join(str(d)
                             for d in data[i]) + ";" + str(x_axis[i]) + "\n")
    else:
        for i in range(len(x_axis)):
            f.write(str(data[i]) + ";" + str(x_axis[i]) + "\n")

    f.close()


def plot_bar_chart(addr, title, x_label, x, y_label, y):
    """Plots a bar chart and saves it to a file.

    addr -- address where the file should be saved.
    title -- title of the chart.
    x_label -- label of the x axis.
    x -- list of x axis values.
    y_label -- label of the y axis.
    y -- list of y axis values."""
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.bar(x, y)
    plt.savefig(addr)
    plt.clf()
    plt.close()


def plot_all_and_hits(addr, title, x_label, x, y_label, y_all, y_hits):
    x = np.array(x)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax = plt.subplot(111)
    r1 = ax.bar(x - 0.2, y_all, width=0.4, color='b', align='center')
    r2 = ax.bar(x + 0.2, y_hits, width=0.4, color='g', align='center')
    ax.legend((r1[0], r2[0]), ('All', 'Hits'))
    plt.savefig(addr)
    plt.clf()
    plt.close()


def comma_formatter():
    return tkr.FuncFormatter(lambda x, p: "{:,}".format(int(x)))


def percent_formatter():
    return tkr.FuncFormatter(lambda x, p: "{0}%".format(int(x)))


def plot_frame_histogram(addr, legend_title, x_label, y_label, frame):
    sns.set(style='whitegrid', font_scale=2.1)
    ax = sns.barplot(x='comb_length', hue='isect_size', y='count', data=frame,
                     color="gray")
    ax.set(xlabel=x_label, ylabel=y_label)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + 0.07,
                     box.width * 0.86, box.height * 0.95])
    legend = ax.legend(title=legend_title, bbox_to_anchor=(1.3, 1.1))
    legend.get_title().set_size('medium')

    plt.savefig(addr)
    plt.clf()
    plt.close()


def plot_histogram(addr, title, x_label, x, y_label, y, normed=0, cumul=0,
                   y_range=None):
    """Plots a bar chart and saves it to a file.

    addr -- address where the file should be saved.
    title -- title of the chart.
    x_label -- label of the x axis.
    x -- list of x axis values.
    y_label -- label of the y axis.
    y -- list of y axis values.
    normed (optional) -- if set to 1, y axis values will be normalized so that
                         the chart is a valid probability density function.
    cumul (optional) -- if set to 1, the function will plot a cumulative histo-
                        gram instead of a regular one."""
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.hist(x, bins=len(x), weights=y, normed=normed, cumulative=cumul)
    if y_range:
        plt.ylim(y_range)
    plt.savefig(addr)
    plt.clf()
    plt.close()


def plot_line_hist(addr, title, x_label, x, y_label, y, normed=0, cumul=0,
                   y_range=None):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    ax = plt.subplot(111)

    if normed:
        y = np.array(y) / sum(y)

    if cumul:
        y = np.cumsum(y)

    ax.plot(x, y)

    plt.ylim(y_range)
    plt.savefig(addr)
    plt.clf()
    plt.close()


def plot_multiple_cumul(addr, title, x_label, x, y_label, data, y_range=None):
    sns.set(style='white', font_scale=2.1)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    ax = plt.subplot(111)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + 0.05,
                     box.width, box.height])

    line_styles = ['-', "--", "-.", ":"]
    style = itertools.cycle(line_styles)

    for label, y in data:
        y = np.array(y) / sum(y)
        y = np.cumsum(y)
        ax.plot(x, y, label=label, linestyle=next(style), linewidth=2.5)

    ax.legend(loc='lower right')
    plt.ylim(y_range)
    plt.savefig(addr, bbox='tight')
    plt.clf()
    plt.close()


def plot_actual_histogram(addr, title, values, x_label, y_label, bins=None):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if bins:
        plt.hist(values, bins=bins)
    else:
        plt.hist(values)
    plt.savefig(addr)
    plt.clf()
    plt.close()


def split_matrix(matrix, y_ticks, max_rows):
    rows, cols = matrix.shape
    num_subplots = ceil(rows / max_rows)
    begin_range = np.arange(num_subplots) * max_rows
    end_range = begin_range + max_rows
    matrices = []
    y_ticks_list = []
    for begin, end in zip(begin_range, end_range):
        matrices.append(matrix[begin:end, :])
        y_ticks_list.append(y_ticks[begin:end])
    return matrices, y_ticks_list

DEFAULT_FIGSIZE = (45, 30)
DEFAULT_DPI = 100
MAX_FIG_DIM = 32767  # width and height must each be below 32768


def plot_matrix_heatmap(addr, title, x_label, x_ticks, y_label, y_ticks,
                        matrix, cmap='hot', color_min=None, color_max=None,
                        text=True, text_size='medium', rows_per_plot=None,
                        figsize=DEFAULT_FIGSIZE):

    scale_factor = min([1] + [MAX_FIG_DIM / (dim * DEFAULT_DPI)
                              for dim in figsize])

    figsize = (dim * scale_factor for dim in figsize)

    if not rows_per_plot:
        rows_per_plot = len(y_ticks)

    matrices, y_ticks_list = split_matrix(matrix, y_ticks, rows_per_plot)

    fig, axes = plt.subplots(1, len(matrices), figsize=figsize)

    # plt.subplots(1,1) returns the axes object directly, whereas larger shapes
    # return np.ndarray with the axes objects inside. The fix below makes axes
    # iterable so that it can be zipped.
    if not isinstance(axes, collections.Iterable):
        axes = [axes]

    fig.suptitle(title, size='xx-large')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for i, (matrix, y_ticks, ax)in enumerate(zip(matrices,
                                                 y_ticks_list,
                                                 axes)):

        cell_width = 1
        cell_height = 1

        x = (np.arange(len(matrix[0]) + 1)) * cell_width

        y = (np.arange(len(matrix) + 1)) * cell_height

        x_tick_pos = x[:-1] + cell_width / 2
        y_tick_pos = y[:-1] + cell_height / 2

        ax.set_xticks(x_tick_pos)
        ax.set_xticklabels(x_ticks, rotation='vertical', size=text_size)
        ax.set_yticks(y_tick_pos)
        ax.set_yticklabels(y_ticks, size=text_size)

        mesh = ax.pcolormesh(matrix, cmap=cmap, vmin=color_min, vmax=color_max)

        if text:
            coords = mesh._coordinates
            rows, cols, _ = coords.shape
            rows, cols = rows - 1, cols - 1

            for i in range(rows):
                for j in range(cols):
                    pos = (coords[i, j, :] + coords[i + 1, j + 1, :]) / 2
                    ax.text(pos[0], pos[1], str(int(matrix[i, j])),
                            color='green', horizontalalignment='center',
                            verticalalignment='center', size=text_size)

    fig.colorbar(mesh, ax=ax)
    plt.savefig(addr, bbox='tight')
    plt.clf()
    plt.close()


def plot_scatter(addr, title, x, x_label, y, y_label):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(x, y)
    plt.savefig(addr)
    plt.clf()
    plt.close()
