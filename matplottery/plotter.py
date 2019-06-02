from __future__ import print_function

import os
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import warnings


import matplottery.utils as utils

def set_defaults():
    from matplotlib import rcParams
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Liberation Sans", "Bitstream Vera Sans", "DejaVu Sans"]
    # rcParams['mathtext.fontset'] = 'custom'
    # rcParams['mathtext.rm'] = 'Liberation Sans'
    # rcParams['mathtext.it'] = 'Liberation Sans:italic'
    # rcParams['mathtext.bf'] = 'Liberation Sans:bold'
    rcParams['legend.fontsize'] = 11
    rcParams['legend.labelspacing'] = 0.2
    rcParams['hatch.linewidth'] = 0.5  # https://stackoverflow.com/questions/29549530/how-to-change-the-linewidth-of-hatch-in-matplotlib
    rcParams['axes.xmargin'] = 0.0 # rootlike, no extra padding within x axis
    rcParams['axes.labelsize'] = 'x-large'
    rcParams['axes.formatter.use_mathtext'] = True
    rcParams['legend.framealpha'] = 0.65
    rcParams['axes.labelsize'] = 'x-large'
    rcParams['axes.titlesize'] = 'large'
    rcParams['xtick.labelsize'] = 'large'
    rcParams['ytick.labelsize'] = 'large'
    rcParams['figure.subplot.hspace'] = 0.1
    rcParams['figure.subplot.wspace'] = 0.1
    rcParams['figure.subplot.right'] = 0.96
    rcParams['figure.max_open_warning'] = 0
    rcParams['figure.dpi'] = 125
    rcParams["axes.formatter.limits"] = [-5,4] # scientific notation if log(y) outside this

def add_cms_info(ax, typ="Simulation", lumi="75.0", xtype=0.09):
    ax.text(0.0, 1.01,"CMS", horizontalalignment='left', verticalalignment='bottom', transform = ax.transAxes, weight="bold", size="large")
    ax.text(xtype, 1.01,typ, horizontalalignment='left', verticalalignment='bottom', transform = ax.transAxes, style="italic", size="large")
    ax.text(0.99, 1.01,"%s fb${}^{-1}$ (13 TeV)" % (lumi), horizontalalignment='right', verticalalignment='bottom', transform = ax.transAxes, size="large")

def fill_between(ax,double_edges,his,los,**kwargs):
    return ax.fill_between(double_edges,his,los, step="mid",
            hatch="///////", facecolor="none",
            edgecolor=(0.4,0.4,0.4), linewidth=0.0, linestyle='-',
            **kwargs
            )

def plot_hist(h,ax=None,**kwargs):
    if not ax: ax = plt.gca()
    kwargs["fmt"] = kwargs.get("fmt","o")
    kwargs["linewidth"] = kwargs.get("linewidth",1.5)
    kwargs["markersize"] = kwargs.get("marksersize",5.0)
    if h.get_attr("color"): kwargs["color"] = h.get_attr("color")
    if h.get_attr("label"): kwargs["label"] = h.get_attr("label")
    do_text = kwargs.pop("text",False)
    counts = h.counts
    yerrs = h.errors
    xerrs = 0.5*h.get_bin_widths()
    centers = h.get_bin_centers()
    width = centers[1]-centers[0]
    good = counts != 0.
    patches = ax.errorbar(centers[good],counts[good],xerr=xerrs[good],yerr=yerrs[good],**kwargs)
    if do_text:
        for x,y,yerr in zip(centers[good],counts[good],yerrs[good]):
            # ax.text(x,y+yerr,"{:.2f}".format(y), horizontalalignment="center",verticalalignment="bottom", fontsize=12, color=patches[0].get_color())
            ax.text(x-width*0.45,y,"{:.2f}".format(y), horizontalalignment="left",verticalalignment="bottom", fontsize=10, color=patches[0].get_color())
    return patches

def plot_stack(bgs=[],data=None,sigs=[], ratio=None,
        title="", xlabel="", ylabel="", filename="",
        mpl_hist_params={}, mpl_data_params={}, mpl_ratio_params={},
        mpl_figure_params={}, mpl_legend_params={}, mpl_sig_params={},
        mpl_title_params={},
        mpl_xtick_params={},
        cms_type=None, lumi="-1",
        do_log=False,
        ratio_range=[],
        do_stack=True,
        do_bkg_syst=False,override_bkg_syst=None,do_bkg_errors=False,
        xticks=[],
        return_bin_coordinates=False,
        ax_main_callback=None,
        ax_ratio_callback=None,
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
            "alpha": 1.0,
            "histtype": "stepfilled",
            "stacked": do_stack,
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
        fig, ax_main = plt.subplots(1,1,gridspec_kw={"top":0.94},**mpl_figure_params)

    _, _, patches = ax_main.hist(centers,bins=bins,weights=weights,label=labels,color=colors,**mpl_bg_hist)
    # for p in patches:
    #     print(p[0])
    #     print(p[0].get_transform())
    if do_bkg_errors:
        for bg,patch in zip(bgs,patches):
            try:
                patch = patch[0]
            except TypeError:
                pass
            ax_main.errorbar(
                    bg.get_bin_centers(),
                    bg.counts,
                    yerr=bg.errors,
                    markersize=patch.get_linewidth(),
                    marker="o",
                    linestyle="",
                    linewidth=patch.get_linewidth(),
                    color=patch.get_edgecolor(),
                    )


    if do_bkg_syst:
        double_edges = np.repeat(sbgs.edges,2,axis=0)[1:-1]
        tot_vals = sbgs.counts
        if override_bkg_syst is not None:
            tot_errs = override_bkg_syst
        else:
            tot_errs = sbgs.errors
        his = np.repeat(tot_vals+tot_errs,2)
        los = np.repeat(tot_vals-tot_errs,2)
        # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.fill_between.html
        fill_between(ax_main,double_edges,his,los,zorder=5)

    if data:
        data_xerr = None
        select = data.counts != 0
        # data_xerr = (data.get_bin_widths()/2)[select]
        ax_main.errorbar(
                data.get_bin_centers()[select],
                data.counts[select],
                yerr=(data.errors[select] if data.get_errors_up() is None else [data.get_errors_down()[select],data.get_errors_up()[select]]),
                xerr=data_xerr,
                label=data.get_attr("label", "Data"),
                zorder=6, **mpl_data_hist)
    if sigs:
        for sig in sigs:
            if mpl_sig_params.get("hist",True):
                ax_main.hist(sig.get_bin_centers(),bins=bins,weights=sig.counts,color=sig.get_attr("color"),histtype="step", label=sig.get_attr("label","sig"))
                ax_main.errorbar(sig.get_bin_centers(),sig.counts,yerr=sig.errors,xerr=None,markersize=1,linewidth=1.5, linestyle="",marker="o",color=sig.get_attr("color"))
            else:
                select = sig.counts != 0
                ax_main.errorbar(sig.get_bin_centers()[select],sig.counts[select],yerr=sig.errors[select],xerr=None,markersize=3,linewidth=1.5, linestyle="",marker="o",color=sig.get_attr("color"), label=sig.get_attr("label","sig"))

    ax_main.set_ylabel(ylabel, horizontalalignment="right", y=1.)
    ax_main.set_title(title,**mpl_title_params)
    legend = ax_main.legend(
            handler_map={ matplotlib.patches.Patch: utils.TextPatchHandler(label_map) },
            **mpl_legend_params
            )
    legend.set_zorder(10)
    if do_log:
        ax_main.set_yscale("log",nonposy="clip")
    else:
        ylims = ax_main.get_ylim()
        ax_main.set_ylim([0.0,ylims[1]])
    ax_main.yaxis.get_offset_text().set_x(-0.095)

    if ax_main_callback:
        ax_main_callback(ax_main)

    if cms_type is not None:
        add_cms_info(ax_main, cms_type, lumi)

    # ax_main.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    if do_ratio:

        mpl_opts_ratio = {
                "label": "Data/bkg.",
                "color": "k",
                # "xerr": data_xerr,
                }
        # If user specified a ratio, use it (now call it ratios with an s)
        if ratio is not None:
            ratios = ratio
            mpl_opts_ratio["label"] = ratios.get_attr("label",mpl_opts_ratio["label"])
            mpl_opts_ratio["color"] = ratios.get_attr("color",mpl_opts_ratio["color"])
        else:
            # Otherwise, if they don't want bkg systs, just divide
            # If they do, zero out bkg error and divide, to preserve only data error
            if not do_bkg_syst:
                ratios = data/sbgs
            else:
                # if we show bkg syst in ratio, we don't want to "double count" the bkg err in the data/sum(bgs)!
                zerobgs = sbgs.copy()
                zerobgs._errors *= 0.
                ratios = data/zerobgs

        mpl_opts_ratio.update(mpl_data_hist)
        mpl_opts_ratio.update(mpl_ratio_params)

        mpl_opts_ratio["yerr"] = ratios.errors
        if ratios.get_errors_up() is not None:
            mpl_opts_ratio["yerr"] = [ratios.get_errors_down(),ratios.get_errors_up()]

        ax_ratio.errorbar(ratios.get_bin_centers(),ratios.counts,**mpl_opts_ratio)
        ax_ratio.set_autoscale_on(False)
        ylims = ax_ratio.get_ylim()
        ax_ratio.plot([ax_ratio.get_xlim()[0],ax_ratio.get_xlim()[1]],[1,1],color="gray",linewidth=1.,alpha=0.5)
        ax_ratio.set_ylim(ylims)
        # ax_ratio.legend()
        if ratio_range:
            ax_ratio.set_ylim(ratio_range)

        if do_bkg_syst:
            if override_bkg_syst is not None:
                rel_errs = np.abs(override_bkg_syst/sbgs.counts)
            else:
                rel_errs = np.abs(sbgs.get_relative_errors())
            double_edges = np.repeat(ratios.edges,2,axis=0)[1:-1]
            his = np.repeat(1.+rel_errs,2)
            los = np.repeat(1.-rel_errs,2)
            fill_between(ax_ratio,double_edges,his,los)

        ax_ratio.set_ylabel(mpl_opts_ratio["label"], horizontalalignment="right", y=1.)
        ax_ratio.set_xlabel(xlabel, horizontalalignment="right", x=1.)

        if len(xticks):

            ax_ratio.xaxis.set_ticks(ratios.get_bin_centers())
            params = dict(horizontalalignment='center',fontsize=7,rotation=90)
            params.update(mpl_xtick_params)
            ax_ratio.set_xticklabels(xticks, **params)

        if ax_ratio_callback:
            ax_ratio_callback(ax_ratio)

    else:
        ax_main.set_xlabel(xlabel, horizontalalignment="right", x=1.)

        if len(xticks):
            ax_main.xaxis.set_ticks(sbgs.get_bin_centers())
            params = dict(horizontalalignment='center',fontsize=12)
            params.update(mpl_xtick_params)
            ax_main.set_xticklabels(xticks, **params)


    if filename:
        # https://stackoverflow.com/questions/22227165/catch-matplotlib-warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout()

        dirname = os.path.dirname(filename)
        if dirname and not os.path.isdir(dirname):
            os.system("mkdir -p {}".format(dirname))

        fig.savefig(filename)
        if ".pdf" in filename:
            fig.savefig(filename.replace(".pdf",".png"))

    if return_bin_coordinates:
        totransform = []
        for count,ep in zip(sbgs.counts,zip(sbgs.edges[:-1],sbgs.edges[1:])):
            totransform.append([ep[0],0.])
            totransform.append([ep[1],count])
        totransform = np.array(totransform)
        disp_coords = ax_main.transData.transform(totransform)
        fig_coords = fig.transFigure.inverted().transform(disp_coords)
        return [fig, fig.axes, fig_coords]
    else:
        return [fig, fig.axes]

def plot_2d(hist,
        title="", xlabel="", ylabel="", filename="",
        mpl_hist_params={}, mpl_2d_params={}, mpl_ratio_params={},
        mpl_figure_params={}, mpl_legend_params={},
        cms_type=None, lumi="-1",
        do_log=False, do_projection=False, do_profile=False,
        cmap="PuBu_r", do_colz=False, colz_fmt=".1f", colz_doerror=True, colz_sizemultiplier=1.,
        logx=False, logy=False,
        xticks=[], yticks=[],
        zrange=[],
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

        # axx.set_ylim([0.,1.])
        # axy.set_xlim([0.,1.])


    ax.set_xlabel(xlabel, horizontalalignment="right", x=1.)
    ax.set_ylabel(ylabel, horizontalalignment="right", y=1.)

    mpl_2d_hist = {
            "cmap": cmap,
            }
    mpl_2d_hist.update(mpl_2d_params)
    if zrange:
        mpl_2d_hist["vmin"] = zrange[0]
        mpl_2d_hist["vmax"] = zrange[1]

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
        y = ax.transData.transform(pts)[:,1]
        fxwidths = (x[1:] - x[:-1]) / (x.max() - x.min())
        info = np.c_[
                np.tile(xcenters,len(ycenters)),
                np.repeat(ycenters,len(xcenters)),
                np.tile(fxwidths,len(ycenters)),
                counts,
                errors
                ]
        norm = mpl_2d_hist.get("norm",
                matplotlib.colors.Normalize(
                    mpl_2d_hist.get("vmin",H.min()),
                    mpl_2d_hist.get("vmax",H.max()),
                    ))
        val_to_rgba = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba
        fs = min(int(30.0/min(len(xcenters),len(ycenters))),15)

        def val_to_text(bv,be):
            # return ("{:%s}\n$\pm${:%s}" % (colz_fmt,colz_fmt)).format(bv,be)
            # return ("{:%s}\n$\pm${:%s}\n($\pm${:.1f}%%)" % (colz_fmt,colz_fmt)).format(bv,be,100.0*be/bv)
            # buff = ("{:%s}\n$\pm${:%s}\n($\pm${:.1f}%%)" % (colz_fmt,colz_fmt)).format(bv,be,100.0*be/bv)
            if bv < 1.0e-6:
                pcterr = 0.
            else:
                pcterr = 100.0*be/bv
            if bv < 1.0e-6 and be < 1.0e-6:
                buff = "0"
            else:
                if colz_doerror:
                    buff = ("{:%s}\n($\pm${:%s}%%)" % (colz_fmt,colz_fmt.replace("e","f"))).format(bv,pcterr)
                else:
                    buff = ("{:%s}" % (colz_fmt)).format(bv)
            # return buff.replace("e-0","e-")
            return buff

        do_autosize = True
        if len(np.unique(np.diff(xcenters).round(2)))+len(np.unique(np.diff(ycenters).round(2))) == 2:
            # for equidistant bins in x and y, don't autosize the bin text
            do_autosize = False
        for x,y,fxw,bv,be in info:
            if do_autosize:
                fs_ = min(5.5*fxw*fs,14)
            else:
                fs_ = 2.5*fs
            fs_ *= colz_sizemultiplier
            color = "w" if (utils.compute_darkness(*val_to_rgba(bv)) > 0.45) else "k"
            # if bv < 0.01: continue
            ax.text(x,y,val_to_text(bv,be),
                    color=color, ha="center", va="center", fontsize=fs_,
                    wrap=True)

    if do_marginal:
        plt.colorbar(mappable, cax=axz)
    else:
        plt.colorbar(mappable)

    if do_marginal:
        if cms_type is not None:
            add_cms_info(axx, cms_type, lumi, xtype=0.10)
        axx.set_title(title)
    else:
        if cms_type is not None:
            add_cms_info(ax, cms_type, lumi, xtype=0.10)
        ax.set_title(title)

    if len(xticks):
        ax.xaxis.set_ticks(xticks)
        # ax.xaxis.set_ticklabels(xticks)
        if all(isinstance(x,(int,float)) for x in xticks):
            ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if len(yticks):
        ax.yaxis.set_ticks(yticks)
        # ax.yaxis.set_ticklabels(yticks)
        if all(isinstance(x,(int,float)) for x in yticks):
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=3))

    if filename:

        dirname = os.path.dirname(filename)
        if dirname and not os.path.isdir(dirname):
            os.system("mkdir -p {}".format(dirname))

        fig.savefig(filename)
        fig.savefig(filename.replace(".pdf",".png"))

    return fig, fig.axes
