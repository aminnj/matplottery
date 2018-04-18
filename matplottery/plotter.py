from __future__ import print_function

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


import matplottery.utils as utils

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

def add_cms_info(ax, typ="Simulation", lumi="75.0", xtype=0.1):
    ax.text(0.0, 1.01,"CMS", horizontalalignment='left', verticalalignment='bottom', transform = ax.transAxes, weight="bold", size="x-large")
    ax.text(xtype, 1.01,typ, horizontalalignment='left', verticalalignment='bottom', transform = ax.transAxes, style="italic", size="x-large")
    ax.text(0.99, 1.01,"%s fb${}^{-1}$ (13 TeV)" % (lumi), horizontalalignment='right', verticalalignment='bottom', transform = ax.transAxes, size="large")

def plot_stack(bgs=[],data=None,sigs=[], ratio=None,
        title="", xlabel="", ylabel="", filename="",
        mpl_hist_params={}, mpl_data_params={}, mpl_ratio_params={},
        mpl_figure_params={}, mpl_legend_params={},
        cms_type=None, lumi="-1",
        ratio_range=[],
        do_bkg_syst=False,
        ):
    set_defaults()

    colors = [bg.get_attr("color") for bg in bgs]
    labels = [bg.get_attr("label") for bg in bgs]
    if not all(colors):
        # print("Not enough colors specified, so using automatic colors")
        colors = None

    if bgs:
        bins = bgs[0].edges
    elif data:
        bins = data.edges
    else:
        print("What are you even trying to plot?")
        return


    centers = [h.get_bin_centers() for h in bgs]
    weights = [h.counts for h in bgs]

    sbgs = sum(bgs)
    total_integral = sbgs.get_integral()
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
        fig, (ax_main,ax_ratio) = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios':[9, 2],"top":0.94},**mpl_figure_params)
    else:
        fig, ax_main = plt.subplots(1,1,**mpl_figure_params)

    ax_main.hist(centers,bins=bins,weights=weights,label=labels,color=colors,**mpl_bg_hist)

    if do_bkg_syst:
        tot_vals = sbgs.counts
        tot_errs = sbgs.errors
        double_edges = np.repeat(sbgs.edges,2,axis=0)[1:-1]
        his = np.repeat(tot_vals+tot_errs,2)
        los = np.repeat(tot_vals-tot_errs,2)
        ax_main.fill_between(double_edges,his,los, step="mid",
                alpha=0.4, facecolor='#cccccc', edgecolor='#aaaaaa', linewidth=0.5, linestyle='-', zorder=5)

    if data:
        data_xerr = None
        select = data.counts != 0
        # data_xerr = (data.get_bin_widths()/2)[select]
        ax_main.errorbar(
                data.get_bin_centers()[select],
                data.counts[select],
                yerr=data.errors[select],
                xerr=data_xerr,
                label=data.get_attr("label", "Data"), 
                zorder=6, **mpl_data_hist)
    if sigs:
        for sig in sigs:
            # ax_main.hist(sig.get_bin_centers(),bins=bins,weights=sig.counts,color="r",histtype="step", label=sig.get_attr("label","sig"))
            ax_main.errorbar(sig.get_bin_centers(),sig.counts,yerr=sig.errors,xerr=None,label=sig.get_attr("label", "Data"), markersize=3,linewidth=1.5, linestyle="",marker="o",color=sig.get_attr("color"))

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
            ratios = data/sum(bgs)


        mpl_opts_ratio = {
                "yerr": ratios.errors,
                "label": "Data/MC",
                # "xerr": data_xerr,
                }
        if ratios.get_errors_up() is not None:
            mpl_opts_ratio["yerr"] = [ratios.get_errors_down(),ratios.get_errors_up()]

        mpl_opts_ratio.update(mpl_data_hist)
        mpl_opts_ratio.update(mpl_ratio_params)

        ax_ratio.errorbar(ratios.get_bin_centers(),ratios.counts,**mpl_opts_ratio)
        ax_ratio.set_autoscale_on(False)
        ylims = ax_ratio.get_ylim()
        ax_ratio.plot([ax_ratio.get_xlim()[0],ax_ratio.get_xlim()[1]],[1,1],color="gray",linewidth=1.,alpha=0.5)
        ax_ratio.set_ylim(ylims)
        # ax_ratio.legend()
        if ratio_range:
            ax_ratio.set_ylim(ratio_range)

        if do_bkg_syst:
            double_edges = np.repeat(ratios.edges,2,axis=0)[1:-1]
            his = np.repeat(1.+np.abs(sbgs.get_relative_errors()),2)
            los = np.repeat(1.-np.abs(sbgs.get_relative_errors()),2)
            ax_ratio.fill_between(double_edges, his, los, step="mid",
                    alpha=0.4, facecolor='#cccccc', edgecolor='#aaaaaa', linewidth=0.5, linestyle='-')

        ax_ratio.set_ylabel(mpl_opts_ratio["label"], horizontalalignment="right", y=1.)
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

def plot_2d(hist,
        title="", xlabel="", ylabel="", filename="",
        mpl_hist_params={}, mpl_data_params={}, mpl_ratio_params={},
        mpl_figure_params={}, mpl_legend_params={},
        cms_type=None, lumi="-1",
        do_log=False, do_projection=False, do_profile=False,
        cmap="PuBu_r", do_colz=False, colz_fmt=".1f",
        logx=False, logy=False,
        xticks=[], yticks=[],
        ):
    set_defaults()

    if do_projection:
        projx = hist.get_x_projection()
        projy = hist.get_y_projection()
    elif do_profile:
        projx = hist.get_x_profile()
        projy = hist.get_y_profile()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.subplots_adjust(left=0.14, right=1.0, top=0.92)
    do_marginal = do_projection or do_profile


    if do_marginal:
        gs = matplotlib.gridspec.GridSpec(2, 3, width_ratios=[4,1,0.1], height_ratios=[1,4], wspace=0.05, hspace=0.05, left=0.1, top=0.94, right=0.92)
        ax = plt.subplot(gs[1,0])
        axz = plt.subplot(gs[1,2])
        axx = plt.subplot(gs[0,0], sharex=ax)  # top x projection
        axy = plt.subplot(gs[1,1], sharey=ax)  # right y projection
        axx.label_outer()
        axy.label_outer()
        fig = plt.gcf()

        col = matplotlib.cm.get_cmap(cmap)(0.4)
        lw = 1.5
        axx.hist(projx.get_bin_centers(), bins=projx.edges, weights=np.nan_to_num(projx.counts), histtype="step", color=col, linewidth=lw)
        axx.errorbar(projx.get_bin_centers(), projx.counts, yerr=projx.errors, linestyle="", marker="o", markersize=0, linewidth=lw, color=col)
        axy.hist(projy.get_bin_centers(), bins=projy.edges, weights=np.nan_to_num(projy.counts), histtype="step", color=col, orientation="horizontal", linewidth=lw)
        axy.errorbar(projy.counts, projy.get_bin_centers(), xerr=projy.errors, linestyle="", marker="o", markersize=0, linewidth=lw, color=col)


    ax.set_xlabel(xlabel, horizontalalignment="right", x=1.)
    ax.set_ylabel(ylabel, horizontalalignment="right", y=1.)

    mpl_2d_hist = {
            "cmap": cmap,
            }

    H = hist.counts
    X, Y = np.meshgrid(*hist.edges)
    if do_log:
        mpl_2d_hist["norm"] = matplotlib.colors.LogNorm(vmin=H[H>H.min()].min(), vmax=H.max())
        if do_marginal:
            axx.set_yscale("log", nonposy='clip')
            axy.set_xscale("log", nonposx='clip')
    mappable = ax.pcolorfast(X, Y, H, **mpl_2d_hist)

    if logx:
        ax.set_xscale("log", nonposx='clip')
    if logy:
        ax.set_yscale("log", nonposy='clip')

    if do_colz:
        xedges, yedges = hist.edges
        xcenters, ycenters = hist.get_bin_centers()
        counts = hist.counts.flatten()
        errors = hist.errors.flatten()
        pts = np.array([
            xedges,
            np.zeros(len(xedges))+yedges[0]
            ]).T
        x = ax.transData.transform(pts)[:,0]
        fxwidths = (x[1:] - x[:-1]) / (x.max() - x.min())
        info = np.c_[
                np.tile(xcenters,len(ycenters)),
                np.repeat(ycenters,len(xcenters)),
                np.tile(fxwidths,len(ycenters)),
                counts,
                errors
                ]
        norm = mpl_2d_hist.get("norm", matplotlib.colors.Normalize(vmin=H.min(),vmax=H.max()))
        val_to_rgba = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba
        fs = min(int(30.0/min(len(xcenters),len(ycenters))),15)

        def val_to_text(bv,be):
            # return ("{:%s}\n$\pm${:%s}" % (colz_fmt,colz_fmt)).format(bv,be)
            # return ("{:%s}\n$\pm${:%s}\n($\pm${:.1f}%%)" % (colz_fmt,colz_fmt)).format(bv,be,100.0*be/bv)
            # buff = ("{:%s}\n$\pm${:%s}\n($\pm${:.1f}%%)" % (colz_fmt,colz_fmt)).format(bv,be,100.0*be/bv)
            buff = ("{:%s}\n($\pm${:.1f}%%)" % (colz_fmt)).format(bv,100.0*be/bv)
            # return buff.replace("e-0","e-")
            return buff

        do_autosize = True
        for x,y,fxw,bv,be in info:
            if do_autosize:
                fs_ = 4.5*fxw*fs
            else:
                fs_ = 1.0*fs
            color = "w" if (utils.compute_darkness(*val_to_rgba(bv)) > 0.45) else "k"
            ax.text(x,y,val_to_text(bv,be),
                    color=color, ha="center", va="center", fontsize=fs_,
                    wrap=True)

    if do_marginal:
        plt.colorbar(mappable, cax=axz)
    else:
        plt.colorbar(mappable)

    if do_marginal:
        if cms_type is not None:
            add_cms_info(axx, cms_type, lumi, xtype=0.12)
        axx.set_title(title)
    else:
        if cms_type is not None:
            add_cms_info(ax, cms_type, lumi, xtype=0.12)
        ax.set_title(title)


    if len(xticks):
        ax.xaxis.set_ticks(xticks)
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if len(yticks):
        ax.yaxis.set_ticks(yticks)
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    if filename:

        dirname = os.path.dirname(filename)
        if dirname and not os.path.isdir(dirname):
            os.system("mkdir -p {}".format(dirname))

        fig.savefig(filename)

    return fig, fig.axes
