import array
import matplotlib
import numpy as np
import copy

MET_LATEX = "E$\\!\\!\\! \\backslash{}_\\mathrm{T}$"

def compute_darkness(r,g,b,a=1.0):
    # darkness = 1 - luminance
    return a*(1.0 - (0.299*r + 0.587*g + 0.114*b))

def clopper_pearson_error(passed, total, level=0.6827):
    """
    matching TEfficiency::ClopperPearson()
    """
    import scipy.stats
    alpha = 0.5*(1.-level)
    low = scipy.stats.beta.ppf(alpha, passed, total-passed+1)
    high = scipy.stats.beta.ppf(1 - alpha, passed+1, total-passed)
    return low, high

def fill_fast(hist, xvals, yvals=None, weights=None):
    """
    partially stolen from root_numpy implementation
    using for loop with TH1::Fill() is slow, so use
    numpy to convert array to C-style array, and then FillN
    """
    two_d = False
    if yvals is not None:
        two_d = True
        yvals = array.array("d", yvals)
    if weights is None:
        weights = np.ones(len(xvals))
    xvals = array.array("d", xvals)
    weights = array.array("d",weights)
    if not two_d:
        hist.FillN(len(xvals),xvals,weights)
    else:
        hist.FillN(len(xvals),xvals,yvals,weights)

class TextPatchHandler(object):
    def __init__(self, label_map={}):
        self.label_map = label_map
        super(TextPatchHandler, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        label = orig_handle.get_label()
        fc = orig_handle.get_facecolor()
        ec = orig_handle.get_edgecolor()
        lw = orig_handle.get_linewidth()
        color = "w" if (compute_darkness(*fc) > 0.45) else "k"
        text = self.label_map.get(label,"")
        patch1 = matplotlib.patches.Rectangle([x0, y0], width, height, facecolor=fc, edgecolor=ec, linewidth=lw, transform=handlebox.get_transform())
        patch2 = matplotlib.text.Text(x0+0.5*width,y0+0.45*height,text,transform=handlebox.get_transform(),fontsize=0.55*fontsize, color=color, ha="center",va="center")
        handlebox.add_artist(patch1)
        handlebox.add_artist(patch2)
        return patch1

class Hist1D(object):

    def __init__(self, obj=[], **kwargs):
        tstr = str(type(obj))

        self._counts = None
        self._edges = None
        self._errors = None
        self._errors_up = None    # only handled when dividing with binomial errors
        self._errors_down = None  # only handled when dividing with binomial errors
        self._extra = {}
        kwargs = self.init_extra(**kwargs)
        if "ROOT." in tstr:
            self.init_root(obj,**kwargs)
        elif "uproot" in tstr:
            self.init_uproot(obj,**kwargs)
        elif "ndarray" in tstr or "list" in tstr:
            self.init_numpy(obj,**kwargs)

    def copy(self):
        hnew = self.__class__()
        hnew.__dict__.update(copy.deepcopy(self.__dict__))
        return hnew

    def init_numpy(self, obj, **kwargs):
        if "errors" in kwargs:
            self._errors = kwargs["errors"]
            del kwargs["errors"]

        self._counts, self._edges = np.histogram(obj,**kwargs)
        self._counts = self._counts.astype(np.float64)

        # poisson defaults if not specified
        if self._errors is None:
            if "weights" not in kwargs:
                self._errors = np.sqrt(self._counts)
            else:
                # if weighted entries, need to get sum of sq. weights per bin
                # and sqrt of that is bin error
                kwargs["weights"] = kwargs["weights"]**2.
                counts, _ = np.histogram(obj,**kwargs)
                self._errors = np.sqrt(counts)
        self._errors = self._errors.astype(np.float64)

    def init_root(self, obj, **kwargs):
        low_edges = np.array([1.0*obj.GetBinLowEdge(ibin) for ibin in range(obj.GetNbinsX()+1)])
        bin_widths = np.array([1.0*obj.GetBinWidth(ibin) for ibin in range(obj.GetNbinsX()+1)])
        self._counts = np.array([1.0*obj.GetBinContent(ibin) for ibin in range(1,obj.GetNbinsX()+1)],dtype=np.float64)
        self._errors = np.array([1.0*obj.GetBinError(ibin) for ibin in range(1,obj.GetNbinsX()+1)],dtype=np.float64)
        self._edges = low_edges + bin_widths

    def init_uproot(self, obj, **kwargs):
        (self._counts, self._edges) = obj.numpy
        self._errors = np.sqrt(obj.fSumw2)[1:-1]
        if len(self._errors) == 0:
            self._errors = np.zeros(len(self._counts))
        self._edges = np.array(self._edges)
        self._counts = np.array(self._counts)

    def init_extra(self, **kwargs):
        if "color" in kwargs:
            self._extra["color"] = kwargs["color"]
            del kwargs["color"]
        if "label" in kwargs:
            self._extra["label"] = kwargs["label"]
            del kwargs["label"]
        return kwargs

    def fill_random(self, pdf="gaus", N=1000):
        if pdf not in ["gaus", "uniform"]:
            print("Warning: {} not a supported function.".format(pdf))
            return
        low, high = self._edges[0], self._edges[-1]
        cent = 0.5*(self._edges[0] + self._edges[-1])
        width = high-low
        if pdf == "gaus": vals = np.random.normal(cent, 0.2*width, N)
        elif pdf == "uniform": vals = np.random.uniform(low, high, N)
        counts, _ = np.histogram(vals, bins=self._edges)
        self._counts += counts
        self._errors = np.sqrt(self._errors**2. + counts)

    def get_errors(self):
        return self._errors

    def get_errors_up(self):
        return self._errors_up

    def get_errors_down(self):
        return self._errors_down

    def get_counts(self):
        return self._counts

    def get_counts_errors(self):
        return self._counts, self._errors

    def get_edges(self):
        return self._edges

    def get_bin_centers(self):
        return 0.5*(np.array(self._edges[1:])+np.array(self._edges[:-1]))

    def get_bin_widths(self):
        return self._edges[1:]-self._edges[:-1]

    def get_integral(self):
        return np.sum(self._counts)

    def get_integral_and_error(self):
        return np.sum(self._counts), np.sum(self._errors**2.0)**0.5

    def _check_consistency(self, other):
        if len(self._edges) != len(other._edges):
            raise Exception("These histograms cannot be combined due to different binning")
        return True

    def __eq__(self, other):
        if not self._check_consistency(other): return False
        eps = 1.e-6
        return np.all(np.abs(self._counts - other.get_counts()) < eps) \
            and np.all(np.abs(self._edges - other.get_edges()) < eps) \
            and np.all(np.abs(self._errors - other.get_errors()) < eps)

    def __add__(self, other):
        if type(other) == int and other == 0:
            return self
        if self._counts is None:
            return other
        self._check_consistency(other)
        hnew = self.__class__()
        hnew._counts = self._counts + other._counts
        hnew._errors = (self._errors**2. + other._errors**2.)**0.5
        hnew._edges = self._edges
        hnew._extra = self._extra
        return hnew

    __radd__ = __add__

    def __sub__(self, other):
        self._check_consistency(other)
        hnew = self.__class__()
        hnew._counts = self._counts - other._counts
        hnew._errors = (self._errors**2. + other._errors**2.)**0.5
        hnew._edges = self._edges
        hnew._extra = self._extra
        return hnew

    def __div__(self, other):
        if type(other) in [float,int]:
            return self.__mul__(1.0/other)
        else:
            return self.divide(other)

    def divide(self, other, binomial=False):
        self._check_consistency(other)
        hnew = self.__class__()
        hnew._edges = self._edges
        hnew._extra = self._extra
        with np.errstate(divide="ignore",invalid="ignore"):
            if not binomial:
                hnew._counts = self._counts / other._counts
                hnew._errors = (
                        (self._errors/other._counts)**2.0 +
                        (other._errors*self._counts/(other._counts)**2.0)**2.0
                        )**0.5
            else:
                hnew._errors_down, hnew._errors_up = clopper_pearson_error(self._counts,other._counts)
                hnew._counts = self._counts/other._counts
                hnew._errors = 0.*hnew._counts
                # these are actually the positions for down and up, but we want the errors
                # wrt to the central value
                hnew._errors_up = hnew._errors_up - hnew._counts
                hnew._errors_down = hnew._counts - hnew._errors_down
        return hnew


    def __mul__(self, fact):
        if type(fact) in [float,int]:
            hnew = self.copy()
            hnew._counts *= fact
            hnew._errors *= fact
            return hnew
        else:
            raise Exception("Can't multiply histogram by non-scalar")

    __rmul__ = __mul__

    def __pow__(self, expo):
        if type(expo) in [float,int]:
            hnew = self.copy()
            with np.errstate(divide="ignore",invalid="ignore"):
                hnew._counts = hnew._counts ** expo
                hnew._errors *= hnew._counts**(expo-1) * expo
            return hnew
        else:
            raise Exception("Can't multiply histogram by non-scalar")

    def __repr__(self):
        use_ascii = False
        if use_ascii: sep = "+-"
        else: sep = u"\u00B1".encode("utf-8")
        # trick: want to use numpy's smart formatting (truncating,...) of arrays
        # so we convert value,error into a complex number and format that 1D array :)
        formatter = {"complex_kind": lambda x:"%5.2f {} %4.2f".format(sep) % (np.real(x),np.imag(x))}
        a2s = np.array2string(self._counts+self._errors*1j,formatter=formatter, suppress_small=True, separator="   ")
        return "<{}:\n{}\n>".format(self.__class__.__name__,a2s)

    def set_attr(self, attr, val):
        self._extra[attr] = val

    def get_attr(self, attr, default=None):
        return self._extra.get(attr,default)

    def get_attrs(self):
        return self._extra

class Hist2D(Hist1D):

    def init_numpy(self, obj, **kwargs):
        if "errors" in kwargs:
            self._errors = kwargs["errors"]
            del kwargs["errors"]

        if len(obj) == 0:
            xs, ys = [],[]
        else:
            xs, ys = obj[:,0], obj[:,1]
        counts, edgesx, edgesy = np.histogram2d(xs, ys, **kwargs)
        # each row = constant y, lowest y on top
        self._counts = counts.T
        self._edges = edgesx, edgesy
        self._counts = self._counts.astype(np.float64)

        # poisson defaults if not specified
        if self._errors is None:
            if "weights" not in kwargs:
                self._errors = np.sqrt(self._counts)
            else:
                # if weighted entries, need to get sum of sq. weights per bin
                # and sqrt of that is bin error
                kwargs["weights"] = kwargs["weights"]**2.
                counts, _, _ = np.histogram2d(obj[:,0],obj[:,1],**kwargs)
                self._errors = np.sqrt(counts.T)
        self._errors = self._errors.astype(np.float64)

    def init_root(self, obj, **kwargs):
        xaxis = obj.GetXaxis()
        yaxis = obj.GetYaxis()
        low_edges_x = np.array([1.0*xaxis.GetBinLowEdge(ibin) for ibin in range(xaxis.GetNbins()+1)])
        bin_widths_x = np.array([1.0*xaxis.GetBinWidth(ibin) for ibin in range(xaxis.GetNbins()+1)])
        low_edges_y = np.array([1.0*yaxis.GetBinLowEdge(ibin) for ibin in range(yaxis.GetNbins()+1)])
        bin_widths_y = np.array([1.0*yaxis.GetBinWidth(ibin) for ibin in range(yaxis.GetNbins()+1)])
        counts, errors = [], []
        for iy in range(1,obj.GetNbinsY()+1):
            counts_y, errors_y = [], []
            for ix in range(1,obj.GetNbinsX()+1):
                cnt = obj.GetBinContent(ix,iy)
                err = obj.GetBinError(ix,iy)
                counts_y.append(cnt)
                errors_y.append(err)
            counts.append(counts_y[:])
            errors.append(errors_y[:])
        self._counts = np.array(counts, dtype=np.float64)
        self._errors = np.array(errors, dtype=np.float64)
        self._edges = low_edges_x + bin_widths_x, low_edges_y + bin_widths_y

    def init_uproot(self, obj, **kwargs):
        # these arrays are (Nr+2)*(Nc+2) in size
        # note that we can't use obj.values because
        # uproot chops off the first and last elements
        # https://github.com/scikit-hep/uproot/blob/master/uproot/hist.py#L79
        err2 = np.array(obj.fSumw2)
        vals = np.array(obj)
        xedges = obj.fXaxis.fXbins
        yedges = obj.fYaxis.fXbins
        self._counts = vals.reshape(len(yedges)+1,len(xedges)+1)[1:-1, 1:-1]
        self._errors = np.sqrt(err2.reshape(len(yedges)+1,len(xedges)+1)[1:-1, 1:-1])
        self._edges = np.array(xedges), np.array(yedges)

    def _check_consistency(self, other):
        if len(self._edges[0]) != len(other._edges[0]) \
                or len(self._edges[1]) != len(other._edges[1]):
            raise Exception("These histograms cannot be combined due to different binning")
        return True

    def __eq__(self, other):
        if not self._check_consistency(other): return False
        eps = 1.e-6
        return np.all(np.abs(self._counts - other.get_counts()) < eps) \
            and np.all(np.abs(self._edges[0] - other.get_edges()[0]) < eps) \
            and np.all(np.abs(self._edges[1] - other.get_edges()[1]) < eps) \
            and np.all(np.abs(self._errors - other.get_errors()) < eps)

    def get_bin_centers(self):
        xcenters = 0.5*(self._edges[0][1:]+self._edges[0][:-1])
        ycenters = 0.5*(self._edges[1][1:]+self._edges[1][:-1])
        return (xcenters,ycenters)

    def get_bin_widths(self):
        xwidths = self._edges[0][1:]-self._edges[0][:-1]
        ywidths = self._edges[1][1:]-self._edges[1][:-1]
        return (xwidths,ywidths)

    def get_x_projection(self):
        hnew = Hist1D()
        hnew._counts = self._counts.sum(axis=0)
        hnew._errors = np.sqrt((self._errors**2).sum(axis=0))
        hnew._edges = self._edges[0]
        return hnew

    def get_y_projection(self):
        hnew = Hist1D()
        hnew._counts = self._counts.sum(axis=1)
        hnew._errors = np.sqrt((self._errors**2).sum(axis=1))
        hnew._edges = self._edges[1]
        return hnew

    def _calculate_profile(self, counts, errors, edges_to_sum, edges):
        centers = 0.5*(edges_to_sum[:-1]+edges_to_sum[1:])
        num = np.matmul(counts.T,centers)
        den = np.sum(counts,axis=0)
        num_err = np.matmul(errors.T**2,centers**2)**0.5
        den_err = np.sum(errors**2, axis=0)**0.5
        with np.errstate(divide="ignore",invalid="ignore"):
            r_val = num/den
            r_err = ((num_err/den)**2 + (den_err*num/den**2.0)**2.0)**0.5
        hnew = Hist1D()
        hnew._counts = r_val
        hnew._errors = r_err
        hnew._edges = edges
        return hnew

    def get_x_profile(self):
        xedges = self._edges[0]
        yedges = self._edges[1]
        return self._calculate_profile(self._counts, self._errors, yedges, xedges)

    def get_y_profile(self):
        xedges = self._edges[0]
        yedges = self._edges[1]
        return self._calculate_profile(self._counts.T, self._errors.T, xedges, yedges)

def register_root_palettes():
    # RGB stops taken from
    # https://github.com/root-project/root/blob/9acb02a9524b2d9d5edb57c519aea4f4ab8022ac/core/base/src/TColor.cxx#L2523

    palettes = {
            "kBird": {
                "reds": [ 0.2082, 0.0592, 0.0780, 0.0232, 0.1802, 0.5301, 0.8186, 0.9956, 0.9764 ],
                "greens": [ 0.1664, 0.3599, 0.5041, 0.6419, 0.7178, 0.7492, 0.7328, 0.7862, 0.9832 ],
                "blues": [ 0.5293, 0.8684, 0.8385, 0.7914, 0.6425, 0.4662, 0.3499, 0.1968, 0.0539 ],
                "stops": np.linspace(0.,1.,9),
                },
            "kRainbow": {
                "reds": [ 0./255., 5./255., 15./255., 35./255., 102./255., 196./255., 208./255., 199./255., 110./255.],
                "greens": [ 0./255., 48./255., 124./255., 192./255., 206./255., 226./255., 97./255., 16./255., 0./255.],
                "blues": [ 99./255., 142./255., 198./255., 201./255., 90./255., 22./255., 13./255., 8./255., 2./255.],
                "stops": np.linspace(0.,1.,9),
                },
            "SUSY": {
                "reds": [0.50, 0.50, 1.00, 1.00, 1.00],
                "greens": [0.50, 1.00, 1.00, 0.60, 0.50],
                "blues": [1.00, 1.00, 0.50, 0.40, 0.50],
                "stops": [0.00, 0.34, 0.61, 0.84, 1.00],
                },
            }

    for key in palettes:
        stops = palettes[key]["stops"]
        reds = palettes[key]["reds"]
        greens = palettes[key]["greens"]
        blues = palettes[key]["blues"]
        cdict = {
            "red": zip(stops,reds,reds),
            "green": zip(stops,greens,greens),
            "blue": zip(stops,blues,blues)
        }
        matplotlib.pyplot.register_cmap(name=key, data=cdict)

