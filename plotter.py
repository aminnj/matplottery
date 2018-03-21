import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils

def set_defaults():
    from matplotlib import rcParams
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = 'helvetica, Helvetica, Arial, Nimbus Sans L, Mukti Narrow, FreeSans, Liberation Sans'
    rcParams['legend.fontsize'] = 'large'
    rcParams['axes.labelsize'] = 'x-large'
    rcParams['axes.titlesize'] = 'x-large'
    rcParams['xtick.labelsize'] = 'large'
    rcParams['ytick.labelsize'] = 'large'
    rcParams['figure.subplot.hspace'] = 0.1
    rcParams['figure.subplot.wspace'] = 0.1


def add_cms_info(ax, typ="Simulation", lumi="75.0"):
    ax.text(0.0, 1.01,"CMS", horizontalalignment='left', verticalalignment='bottom', transform = ax.transAxes, weight="bold", size="x-large")
    ax.text(0.10, 1.01,typ, horizontalalignment='left', verticalalignment='bottom', transform = ax.transAxes, style="italic", size="x-large")
    ax.text(0.99, 1.01,"%s fb${}^{-1}$ (13 TeV)" % (lumi), horizontalalignment='right', verticalalignment='bottom', transform = ax.transAxes, size="x-large")

def plot_stack(bgs=[],data=None,sigs=[], ratio=None,
        title="", xlabel="", ylabel="", filename="",
        mpl_hist_params={}, mpl_data_params={}, mpl_ratio_params={},
        mpl_figure_params={}, mpl_legend_params={},
        cms_type=None, lumi="-1",
        ):
    set_defaults()

    colors = [bg.get_attr("color") for bg in bgs]
    labels = [bg.get_attr("label") for bg in bgs]
    if not all(colors):
        # print "Not enough colors specified, so using automatic colors"
        colors = None

    if bgs:
        bins = bgs[0].get_edges()
    elif data:
        bins = data.get_edges()
    else:
        print "What are you even trying to plot?"
        return


    centers = [h.get_bin_centers() for h in bgs]
    weights = [h.get_counts() for h in bgs]

    total_integral = sum(bgs,utils.Hist1D()).get_integral()
    label_map = { bg.get_attr("label"):"{:.0f}%".format(100.0*bg.get_integral()/total_integral) for bg in bgs }
    # label_map = { label:"{:.1f}".format(hist.get_integral()) for label,hist in zip(labels,bgs) }

    mpl_bg_hist = {
            "alpha": 0.9,
            "histtype": "stepfilled",
            "stacked": True,
            }
    mpl_bg_hist.update(mpl_hist_params)
    mpl_data_hist = {
            "color": "k",
            "linestyle": "",
            "marker": "o",
            "markersize": 3,
            "linewidth": 1.5,
            }
    mpl_data_hist.update(mpl_data_params)

    do_ratio = False
    if data is not None:
        do_ratio = True
    if ratio is not None:
        do_ratio = True

    if do_ratio:
        fig, (ax_main,ax_ratio) = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios':[9, 2]},**mpl_figure_params)
    else:
        fig, ax_main = plt.subplots(1,1,**mpl_figure_params)

    ax_main.hist(centers,bins=bins,weights=weights,label=labels,color=colors,**mpl_bg_hist)
    if data:
        data_xerr = None
        # data_xerr = data.get_bin_widths()/2
        ax_main.errorbar(data.get_bin_centers(),data.get_counts(),yerr=data.get_errors(),xerr=data_xerr,label=data.get_attr("label", "Data"), **mpl_data_hist)
    if sigs:
        for sig in sigs:
            ax_main.hist(sig.get_bin_centers(),bins=bins,weights=sig.get_counts(),color="r",histtype="step", label=sig.get_attr("label","sig"))

    ax_main.set_ylabel(ylabel, horizontalalignment="right", y=1.)
    ax_main.set_title(title)
    ax_main.legend(
            handler_map={ matplotlib.patches.Patch: utils.TextPatchHandler(label_map) },
            **mpl_legend_params
            )
    ylims = ax_main.get_ylim()
    ax_main.set_ylim([0.0,ylims[1]])

    if cms_type is not None:
        add_cms_info(ax_main, cms_type, lumi)

    # ax_main.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    if do_ratio:

        if ratio is not None:
            ratios = ratio
        else:
            ratios = data/sum(bgs,utils.Hist1D())


        mpl_opts_ratio = {
                "yerr": ratios.get_errors(),
                "label": "Data/MC",
                # "xerr": data_xerr,
                }
        if ratios.get_errors_up() is not None:
            mpl_opts_ratio["yerr"] = [ratios.get_errors_down(),ratios.get_errors_up()]

        mpl_opts_ratio.update(mpl_data_hist)
        mpl_opts_ratio.update(mpl_ratio_params)

        ax_ratio.errorbar(ratios.get_bin_centers(),ratios.get_counts(),**mpl_opts_ratio)
        ax_ratio.set_autoscale_on(False)
        ylims = ax_ratio.get_ylim()
        ax_ratio.plot([ax_ratio.get_xlim()[0],ax_ratio.get_xlim()[1]],[1,1],color="gray",linewidth=1.,alpha=0.5)
        ax_ratio.set_ylim(ylims)
        ax_ratio.legend()
        # ax_ratio.set_ylim([0.,1.])

        ax_ratio.set_xlabel(xlabel, horizontalalignment="right", x=1.)
    else:
        ax_main.set_xlabel(xlabel, horizontalalignment="right", x=1.)

    if filename:
        fig.tight_layout()

        dirname = os.path.dirname(filename)
        if dirname and not os.path.isdir(dirname):
            os.system("mkdir -p {}".format(dirname))

        fig.savefig(filename)

    return fig, fig.axes
