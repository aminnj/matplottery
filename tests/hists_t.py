import unittest

import ROOT as r
import numpy as np
from matplottery.utils import Hist1D, Hist2D, fill_fast

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

        self.assertEqual(h1+h2+h3, sum([h1,h2,h3]))

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


if __name__ == "__main__":
    unittest.main()
