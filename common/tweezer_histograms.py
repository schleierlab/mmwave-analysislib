from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import seaborn as sns
from matplotlib.axes import Axes
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

from analysislib.common.image import ROI, Image


class TweezerThresholder:
    INDEX_NAME: ClassVar[str] = 'Tweezer index'
    COUNTS_NAME: ClassVar[str] = 'Counts'
    df: pd.DataFrame
    rois = list[ROI]
    gmms: list[TweezerCountGMM]

    def __init__(
            self,
            images: Sequence[Image],
            rois: Sequence[ROI],
            weights: Sequence[NDArray | float] | float = 1,
            background_subtract: bool = False,
    ):
        self.rois = list(rois)

        weight_fns =  weights
        if isinstance(weights, float):
            weight_fns = (weights,) * len(self.n_sites)
        roi_counts = [
            [
                np.sum(
                    (image if background_subtract else image.raw_image()).roi_view(roi)
                    * weight_fn
                )
                for roi, weight_fn in zip(rois, weight_fns)
            ]
            for image in images
        ]
        self.df = pd.DataFrame(roi_counts).melt(var_name=self.INDEX_NAME, value_name=self.COUNTS_NAME)

    @property
    def n_sites(self):
        return len(self.rois)

    def violinplot(self, ax: Optional[Axes] = None):
        sns.violinplot(
            self.df,
            x='Tweezer index',
            y='Counts',
            inner='point',
            linewidth=0.3,
            ax=ax,
        )

    def fit_gmms(self):
        self.gmms = [
            TweezerCountGMM(self.df[self.df['Tweezer index'] == i]['Counts'])
            for i in range(self.n_sites)
        ]
        self.means = np.array([gmm.means for gmm in self.gmms])
        self.stds = np.array([gmm.stds for gmm in self.gmms])
        self.thresholds = np.array([gmm.balanced_threshold() for gmm in self.gmms])
        self.loading_rates = np.array([gmm.weights[1] for gmm in self.gmms])
        self.infidelities = np.array([gmm.infidelity_at_threshold() for gmm in self.gmms])

    def plot_spreads(self, ax: Optional[Axes] = None, color='C0'):
        if ax is None:
            fig, ax = plt.subplots()

        inds = np.arange(self.n_sites)
        ax.plot(inds, self.thresholds, color='red', linestyle='dashed')
        for i in range(2):
            mean_i = self.means[:, i]
            std_i = self.stds[:, i]

            ax.plot(inds, mean_i, marker='.', color=color)
            for j in range(1, 3 + 1):
                ax.fill_between(
                    inds,
                    mean_i - j * std_i,
                    mean_i + j * std_i,
                    alpha=0.2,
                    color=color,
                )

    def plot_loading_rate(self, ax: Optional[Axes] = None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        plot_kw = dict(marker='.') | kwargs
        ax.plot(np.arange(self.n_sites), self.loading_rates, **plot_kw)
        ax.axhline(0.5, color='0.5', linestyle='dashed')

    def plot_infidelity(self, ax: Optional[Axes] = None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        plot_kw = dict(marker='.') | kwargs
        ax.plot(np.arange(self.n_sites), self.infidelities, **plot_kw)
        ax.set_yscale('log')


class TweezerCountGMM:
    gmm: GaussianMixture
    _order: NDArray  # either [0, 1] or [1, 0] depending on order of means in self.gmm

    def __init__(self, data):
        '''
        Parameters
        ----------
        data: array_like
            Can be a list, numpy NDArray, or pandas Series
        '''
        self.gmm = GaussianMixture(n_components=2)
        self.gmm.fit(np.asarray(data)[:, np.newaxis])
        self._order = np.argsort(self.gmm.means_.flatten())

    @property
    def means(self):
        return self.gmm.means_.flatten()[self._order]

    @property
    def stds(self):
        return np.sqrt(self.gmm.covariances_.flatten()[self._order])

    @property
    def weights(self):
        return self.gmm.weights_.flatten()[self._order]

    def balanced_threshold(self):
        '''
        Threshold for which the infidelity is minimized given the inferred filling fraction.
        Balances the probability of observing a false negative P(0_actual, 1_predicted)
        and the probability of observing a false positive P(1_actual, 0_predicted).

        Returns
        -------
        threshold: float
        '''
        mean0, mean1 = self.means
        std0, std1 = self.stds
        weight0, weight1 = self.weights

        def d_err_d_thresh(x):
            return (
                weight1 * norm.pdf(x, loc=mean1, scale=std1)
                - weight0 * norm.pdf(x, loc=mean0, scale=std0)
            )

        return scipy.optimize.fsolve(d_err_d_thresh, x0=(mean0 + mean1)/2).item()

    def infidelity_at_threshold(self, threshold=None):
        '''
        Compute infidelity using the fitted Gaussian mixture model parameters
        for the infidelity-minimizing threshold.

        Parameters
        ----------
        threshold: float, optional
            Threshold for computing infidelity. If None, uses the infidelity-minimizing threshold.

        Returns
        -------
        infidelity: float
        '''
        thresh = threshold
        if threshold is None:
            thresh = self.balanced_threshold()

        predict_1_given_0 = 1 - norm.cdf(thresh, loc=self.means[0], scale=self.stds[0])
        predict_0_given_1 = norm.cdf(thresh, loc=self.means[1], scale=self.stds[1])

        return predict_1_given_0 * self.weights[0] + predict_0_given_1 * self.weights[1]
