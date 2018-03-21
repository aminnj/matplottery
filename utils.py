import matplotlib
import numpy as np

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

    def __init__(self, obj=None, **kwargs):
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
        elif "ndarray" in tstr:
            self.init_numpy(obj,**kwargs)

    def init_numpy(self, obj, **kwargs):
        if "errors" in kwargs:
            self._errors = kwargs["errors"]
            del kwargs["errors"]

        self._counts, self._edges = np.histogram(obj,**kwargs)
        self._counts = self._counts.astype(np.float64)

        # poisson defaults if not specified
        if self._errors is None:
            self._errors = np.sqrt(self._counts)
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

    def init_extra(self, **kwargs):
        if "color" in kwargs:
            self._extra["color"] = kwargs["color"]
            del kwargs["color"]
        if "label" in kwargs:
            self._extra["label"] = kwargs["label"]
            del kwargs["label"]
        return kwargs

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
        return 0.5*(self._edges[1:]+self._edges[:-1])

    def get_bin_widths(self):
        return self._edges[1:]-self._edges[:-1]

    def get_integral(self):
        return np.sum(self._counts)

    def _check_consistency(self, other):
        if len(self._edges) != len(other._edges):
            raise Exception("These histograms cannot be combined due to different binning")

    def __add__(self, other):
        if self._counts is None: 
            return other
        self._check_consistency(other)
        hnew = Hist1D()
        hnew._counts = self._counts + other._counts
        hnew._errors = (self._errors**2. + other._errors**2.)**0.5
        hnew._edges = self._edges
        hnew._extra = self._extra
        return hnew

    def __sub__(self, other):
        self._check_consistency(other)
        hnew = Hist1D()
        hnew._counts = self._counts - other._counts
        hnew._errors = (self._errors**2. + other._errors**2.)**0.5
        hnew._edges = self._edges
        hnew._extra = self._extra
        return hnew

    def __div__(self, other):
        return self.divide(other)

    def divide(self, other, binomial=False):
        self._check_consistency(other)
        hnew = Hist1D()
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
            self._counts *= fact
            self._errors *= fact**0.5
            return self
        else:
            raise Exception("Can't multiply histogram by non-scalar")

    __rmul__ = __mul__

    def __pow__(self, expo):
        if type(expo) in [float,int]:
            with np.errstate(divide="ignore",invalid="ignore"):
                self._counts = self._counts ** expo
                self._errors *= self._counts**(expo-1) * expo
            return self
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
        return "<Hist1D:\n{}\n>".format(a2s)

    def set_attr(self, attr, val):
        self._extra[attr] = val

    def get_attr(self, attr, default=None):
        return self._extra.get(attr,default)

    def get_attrs(self):
        return self._extra
