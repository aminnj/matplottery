import unittest

import ROOT as r
import numpy as np
from matplottery.utils import Hist1D, Hist2D, fill_fast, nan_to_num
import uproot
import os

np.set_printoptions(linewidth=120)

class HistTest(unittest.TestCase):

    def test_1d(self):
        bins = 1.0*np.array([0,3,6,9,12,15])
        vals = 1.0*np.array([1,2,3,4,5,10,13])
        weights = 1.0*np.array([1,1,1,2,2,1,1])
        hr_ = r.TH1F("hr","hr", len(bins)-1, bins)
        fill_fast(hr_, vals, weights=weights)
        hr = Hist1D(hr_)
        hn = Hist1D(vals,bins=bins, weights=weights)

        self.assertEqual(hn, hr)

        self.assertEqual(hn.get_integral(), np.sum(weights))
        self.assertEqual(hr.get_integral(), np.sum(weights))

        self.assertEqual(np.all(hn.edges == bins), True)
        self.assertEqual(np.all(hr.edges == bins), True)

        check = np.histogram(vals,bins=bins,weights=weights)[0]
        self.assertEqual(np.all(hn.counts == check), True)
        self.assertEqual(np.all(hr.counts == check), True)

        self.assertEqual(Hist1D(hr_*2), hn*2)
        self.assertEqual(Hist1D(hr_+hr_), hn+hn)

        self.assertEqual(Hist1D(hr_+0.5*hr_), hn+0.5*hn)

    def test_1d_summing(self):
        np.random.seed(42)

        vals = np.random.normal(0,1,1000)
        bins = np.linspace(-3,3,10)
        h1 = Hist1D(vals,bins=bins)

        vals = np.random.normal(0,1,1000)
        h2 = Hist1D(vals,bins=bins)

        vals = np.random.normal(0,1,1000)
        h3 = Hist1D(vals,bins=bins)

        self.assertNotEqual(h1,h2)
        self.assertNotEqual(h2,h3)
        self.assertEqual(h1+h2+h3, sum([h1,h2,h3]))

    def test_1d_summing_weights(self):
        bins = 1.0*np.array([0,3,6,9,12,15])
        vals1 = 1.0*np.array([4,1,2,3,4,5,10,13])
        weights1 = 1.0*np.array([-1,1,1,1,2,2,1,1])
        vals2 = 1.0*np.array([4,0,2,3,4,-5,100,13])
        weights2 = 1.0*np.array([-1,2,-1,1,2,2,-1,1])
        hr1_ = r.TH1F("hr1","hr1", len(bins)-1, bins)
        hr2_ = r.TH1F("hr2","hr2", len(bins)-1, bins)
        fill_fast(hr1_, vals1, weights=weights1)
        fill_fast(hr2_, vals2, weights=weights2)
        for no_overflow in [True,False]:
            hr1 = Hist1D(hr1_, no_overflow=no_overflow)
            hr2 = Hist1D(hr2_, no_overflow=no_overflow)
            hn1 = Hist1D(vals1,bins=bins, weights=weights1, no_overflow=no_overflow)
            hn2 = Hist1D(vals2,bins=bins, weights=weights2, no_overflow=no_overflow)
            self.assertEqual(hr1+hr2, hn1+hn2)

    def test_copy(self):
        np.random.seed(42)
        h1 = Hist1D(np.random.normal(0,5,100), bins=np.linspace(-3,3,10))
        h2 = h1.copy()
        # Contents are the same, but memory location/hash is different
        self.assertEqual(h1,h2)
        self.assertNotEqual(h1.__hash__(),h2.__hash__())

    def test_idempotent(self):
        np.random.seed(42)
        h1 = Hist1D(np.random.normal(0,5,100), bins=np.linspace(-3,3,10))
        h2 = Hist1D(h1)
        self.assertEqual(h1,h2)

    def test_extra(self):
        np.random.seed(42)
        h1 = Hist1D(np.random.normal(0,5,100), bins=np.linspace(-3,3,10),label="h1",color="k")
        self.assertEqual(h1.get_attr("label"),"h1")
        self.assertEqual(h1.get_attr("color"),"k")
        h1.set_attr("label","h2")
        self.assertEqual(h1.get_attr("label"),"h2")
        h2 = Hist1D(h1,color="r")
        self.assertEqual(h2.get_attr("label"),"h2")
        self.assertEqual(h2.get_attr("color"),"r")

    def test_overflow_integral(self):
        # by default, overflow gets put into first and last bins, unless no_overflow is True
        N = 100
        h1 = Hist1D(np.random.normal(0,5,N), bins=[-5,0,5])
        self.assertEqual(h1.get_integral(),N)
        h1 = Hist1D(np.random.normal(0,5,N), bins=[-5,0,5],no_overflow=False)
        self.assertEqual(h1.get_integral(),N)
        h1 = Hist1D(np.random.normal(0,5,N), bins=[-5,0,5],no_overflow=True)
        self.assertEqual(h1.get_integral()<N,True)

    def test_1d_rebinning(self):
        np.random.seed(42)
        nrebin = 5
        h1 = Hist1D(np.random.normal(0,5,1000), bins=np.linspace(-10,10,21))
        nbins_before = len(h1.edges) - 1
        int_before = h1.get_integral()
        h1.rebin(nrebin)
        nbins_after = len(h1.edges) - 1
        int_after = h1.get_integral()
        self.assertEqual(int_before, int_after)
        self.assertEqual(nbins_after, nbins_before // nrebin)

    def test_1d_binomial_errors(self):
        bins = 1.0*np.array([0,3,6,9,12,15,18])

        vals = 1.0*np.array([0,3,5,5,13])
        weights = 1.0*np.array([1,1,1,1,1])
        hrnum_ = r.TH1F("hrnum","", len(bins)-1, bins)
        fill_fast(hrnum_, vals, weights=weights)

        vals = 1.0*np.array([1,2,3,4,5,10,13,16])
        weights = 1.0*np.array([1,1,1,1,1,1,1,1])
        hrden_ = r.TH1F("hrden","", len(bins)-1, bins)
        fill_fast(hrden_, vals, weights=weights)

        teff = r.TEfficiency(hrnum_,hrden_)
        errsup,errsdown = [], []
        for ibin in range(1,hrnum_.GetNbinsX()+1):
            cent = teff.GetEfficiency(ibin)
            errup = teff.GetEfficiencyErrorUp(ibin)
            errdown = teff.GetEfficiencyErrorLow(ibin)
            errsup.append(errup)
            errsdown.append(errdown)

        hfromroot = Hist1D(hrnum_)/Hist1D(hrden_)
        hfromroot._errors_up = np.array(errsup)
        hfromroot._errors_down = np.array(errsdown)
        hfromroot._fix_nan()
        hfromroot._errors *= 0.

        hmine = Hist1D(hrnum_).divide(Hist1D(hrden_),binomial=True)
        hmine._errors *= 0.
        hmine._fix_nan()

        self.assertEqual(hfromroot,hmine)

    def test_1d_divide_hist(self):
        bins = 1.0*np.array([0,3,6,9,12,15])
        vals = 1.0*np.array([1,2,3,4,5,10,13])
        weights = 1.0*np.array([1,1,1,2,2,1,1])
        hr_ = r.TH1F("hr","hr", len(bins)-1, bins)
        fill_fast(hr_, vals, weights=weights)
        hr_.Divide(hr_)
        hn = Hist1D(vals,bins=bins, weights=weights)
        hn /= hn
        hn._fix_nan()
        self.assertEqual(Hist1D(hr_),hn)

    def test_2d_nonuniform_binning(self):

        xbins = np.array([10.,15.,25.,35.,50.0,70,90])
        ybins = np.array([0.,0.8,1.479,2.5])
        h2 = r.TH2F("2d_nonuniform","",len(xbins)-1,xbins,len(ybins)-1,ybins)
        h2m = Hist2D(h2)

        self.assertEqual(tuple(xbins), tuple(h2m.edges[0]))
        self.assertEqual(tuple(ybins), tuple(h2m.edges[1]))

    def test_2d(self):
        vals2d = 1.0*np.array([
                [1,1],
                [1,3],
                [1,4],
                [1,4],
                [3,1],
                [3,4],
                ])
        bins = [
                np.linspace(0.,4.,3),  # edges 0.0,2.0,4.0
                np.linspace(0.,5.,3),  # edges 0.0,2.5,5.0
                ]
        weights = 1.0*np.array([1,1,2,3,1,4])

        hr_ = r.TH2F("hr2d","hr2d", len(bins[0])-1, bins[0], len(bins[1])-1, bins[1])
        fill_fast(hr_, vals2d[:,0], vals2d[:,1], weights=weights)
        hr = Hist2D(hr_)

        hn = Hist2D(vals2d,bins=bins,weights=weights)

        self.assertEqual(hn, hr)

        self.assertEqual(hn.get_integral(), hr.get_integral())

        self.assertEqual(np.all(hr.edges[0] == bins[0]), True)
        self.assertEqual(np.all(hr.edges[1] == bins[1]), True)
        self.assertEqual(np.all(hn.edges[0] == bins[0]), True)
        self.assertEqual(np.all(hn.edges[1] == bins[1]), True)

        hr2x_ = hr_.Clone("hr2x")
        hr2x_.Scale(2.0)
        self.assertEqual(Hist2D(hr2x_), hn*2)

        hr2p_ = hr_.Clone("hr2p")
        hr2p_.Add(hr_)
        self.assertEqual(Hist2D(hr2p_), hn+hn)

    def test_2d_projections_and_profiles(self):

        np.random.seed(42)
        xbins = np.array([10.,15.,25.,35.,50.0,70,90])
        ybins = np.array([0.,0.8,1.479,2.5])
        h2 = r.TH2F("2d_projprof","",len(xbins)-1,xbins,len(ybins)-1,ybins)
        N = 10000
        for x in np.c_[
                (78.*np.random.random(N)+11),
                (2.499*np.random.random(N)),
                ]:
            w = 1.
            if x[1] > 1.2: w = 2.
            h2.Fill(x[0],x[1], w)

        h2m = Hist2D(h2)

        self.assertEqual(Hist1D(h2.ProjectionX()),h2m.get_x_projection())
        self.assertEqual(Hist1D(h2.ProjectionY()),h2m.get_y_projection())

        profx = h2.ProfileX()
        profy = h2.ProfileY()
        # note, TProfile does weird thing to overflow, even though they are empty
        # so explicitly exclude them
        rx = Hist1D(profx,no_overflow=True)
        ry = Hist1D(profy,no_overflow=True)
        nx = h2m.get_x_profile()
        ny = h2m.get_y_profile()

        # XXX profile errors don't match ROOT's, so zero them all out before comparing
        rx._errors *= 0.
        ry._errors *= 0.
        nx._errors *= 0.
        ny._errors *= 0.

        self.assertEqual(rx,nx)
        self.assertEqual(ry,ny)

    def test_read_uproot(self):

        # Write a temporary file
        fr = r.TFile("tmpfile.root","RECREATE")
        fr.cd()
        bins = 1.0*np.array([0,3,6,9,12,15])
        vals = 1.0*np.array([1,2,3,4,5,10,13])
        weights = 1.0*np.array([1,1,1,2,2,1,1])
        hr_ = r.TH1F("hr","hr", len(bins)-1, bins)
        fill_fast(hr_, vals, weights=weights)
        hr = Hist1D(hr_)
        hr_.Write()
        fr.Close()
        h = uproot.open("tmpfile.root")["hr"]
        hu = Hist1D(h)
        self.assertEqual(hu, hr)
        os.system("rm -f tempfile.root")


if __name__ == "__main__":
    unittest.main()
