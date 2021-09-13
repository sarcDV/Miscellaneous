#!/bin/python
"""
Normality Tests

    Shapiro-Wilk Test
    D’Agostino’s K^2 Test
    Anderson-Darling Test

Correlation Tests

    Pearson’s Correlation Coefficient
    Spearman’s Rank Correlation
    Kendall’s Rank Correlation
    Chi-Squared Test

Stationary Tests

    Augmented Dickey-Fuller
    Kwiatkowski-Phillips-Schmidt-Shin

Parametric Statistical Hypothesis Tests

    Student’s t-test
    Paired Student’s t-test
    Analysis of Variance Test (ANOVA)
    Repeated Measures ANOVA Test

Nonparametric Statistical Hypothesis Tests

    Mann-Whitney U Test
    Wilcoxon Signed-Rank Test
    Kruskal-Wallis H Test
    Friedman Test

"""
from scipy.stats import shapiro, normaltest, anderson, \
                        ttest_ind, ttest_rel, f_oneway, mannwhitneyu, wilcoxon, \
                        pearsonr, spearmanr, kendalltau, chi2_contingency, \
                        kruskal, friedmanchisquare

from statsmodels.tsa.stattools import adfuller, kpss

#### 
def ShapiroWilkTest(sample1):
    """
    Shapiro-Wilk Test
    Tests whether a data sample has a Gaussian distribution.
    Assumptions
        Observations in each sample are independent and identically distributed (iid).
    Interpretation
        H0: the sample has a Gaussian distribution.
        H1: the sample does not have a Gaussian distribution.
    """
    # Example of the Shapiro-Wilk Normality Test
    
    data = sample1
    stat, p = shapiro(data)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably Gaussian for the Shapiro-Wilk Test')
    else:
        print('Probably not Gaussian for the Shapiro-Wilk Test ')

    return stat, p

def DAgostinoKTwoTest(sample1):
    """D’Agostino’s K^2 Test
    Tests whether a data sample has a Gaussian distribution.
    Assumptions
        Observations in each sample are independent and identically distributed (iid).
    Interpretation
        H0: the sample has a Gaussian distribution.
        H1: the sample does not have a Gaussian distribution.
    """
    if len(sample1)>=20:
        stat, p = normaltest(sample1)
        print('stat=%.3f, p=%.3f' % (stat, p))
        if p > 0.05:
            print('Probably Gaussian for D`Agostino`s K^2 Test')
        else:
            print('Probably not Gaussian for D`Agostino`s K^2 Test')
    else:
        print("Sample data too small for D`Agostino`s K^2 Test")
        stat, p = normaltest(sample1)
        if p > 0.05:
            print('Probably Gaussian for D`Agostino`s K^2 Test')
        else:
            print('Probably not Gaussian for D`Agostino`s K^2 Test')


    return stat, p

def AndersonDarlingTest(sample1):
    """Anderson-Darling Test
    Tests whether a data sample has a Gaussian distribution.
    Assumptions
        Observations in each sample are independent and identically distributed (iid).
    Interpretation
        H0: the sample has a Gaussian distribution.
        H1: the sample does not have a Gaussian distribution.
    """
    result = anderson(sample1)
    print('stat=%.3f' % (result.statistic))
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < cv:
            print('Probably Gaussian at the %.1f%% level for the Anderson-Darling Test' % (sl))
        else:
            print('Probably not Gaussian at the %.1f%% level for the Anderson-Darling Test' % (sl))
    return result

def PearsonCorrCoeff(sample1, sample2):
    """Pearson’s Correlation Coefficient
    Tests whether two samples have a linear relationship.
    Assumptions
        Observations in each sample are independent and identically distributed (iid).
        Observations in each sample are normally distributed.
        Observations in each sample have the same variance.
    Interpretation
        H0: the two samples are independent.
        H1: there is a dependency between the samples.
    """
    stat, p = pearsonr(sample1, sample2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably independent Pearson’s Correlation Coefficient')
    else:
        print('Probably dependent Pearson’s Correlation Coefficient')

    return stat, p

def SpearmanRankCorrelation(sample1, sample2):
    """Spearman’s Rank Correlation
    Tests whether two samples have a monotonic relationship.
    Assumptions
        Observations in each sample are independent and identically distributed (iid).
        Observations in each sample can be ranked.
    Interpretation
        H0: the two samples are independent.
        H1: there is a dependency between the samples.
    """
    stat, p = spearmanr(sample1, sample2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably independent Spearman’s Rank Correlation')
    else:
        print('Probably dependent Spearman’s Rank Correlation')

    return stat, p

def KendalRankCorr(sample1, sample2):
    """Kendall’s Rank Correlation
    Tests whether two samples have a monotonic relationship.
    Assumptions
        Observations in each sample are independent and identically distributed (iid).
        Observations in each sample can be ranked.
    Interpretation
        H0: the two samples are independent.
        H1: there is a dependency between the samples.
    """
    stat, p = kendalltau(sample1, sample2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably independent Kendall’s Rank Correlation')
    else:
        print('Probably dependent Kendall’s Rank Correlation')

    return stat, p

def ChiSquaredTest(table):
    """Chi-Squared Test
    Tests whether two categorical variables are related or independent.
    Assumptions
        Observations used in the calculation of the contingency table are independent.
        25 or more examples in each cell of the contingency table.
    Interpretation
        H0: the two samples are independent.
        H1: there is a dependency between the samples.
    """
    stat, p, dof, expected = chi2_contingency(table)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably independent Chi-Squared Test')
    else:
        print('Probably dependent Chi-Squared Test')

    return stat, p, dof, expected 


def AugmentedDickeyFullerURTest(sample):
    """
    Augmented Dickey-Fuller Unit Root Test
    Tests whether a time series has a unit root, e.g. has a trend or more generally is autoregressive.
    Assumptions
        Observations in are temporally ordered.
    Interpretation
        H0: a unit root is present (series is non-stationary).
        H1: a unit root is not present (series is stationary).
    """ 
    stat, p, lags, obs, crit, t = adfuller(sample)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably not Stationary Augmented Dickey-Fuller Unit Root Test')
    else:
        print('Probably Stationary Augmented Dickey-Fuller Unit Root Test')

    return stat, p, lags, obs, crit, t

def KwiatkowskiPhillipsSchmidtShin(sample):
    """Kwiatkowski-Phillips-Schmidt-Shin
    Tests whether a time series is trend stationary or not.
    Assumptions
        Observations in are temporally ordered.
    Interpretation
        H0: the time series is not trend-stationary.
        H1: the time series is trend-stationary.
    """ 
    stat, p, lags, crit = kpss(sample, nlags='auto')
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably not Stationary Kwiatkowski-Phillips-Schmidt-Shin')
    else:
        print('Probably Stationary Kwiatkowski-Phillips-Schmidt-Shin')

    return stat, p, lags, crit

def StudentTTest(sample1, sample2):
    """Student’s t-test
    Tests whether the means of two independent samples are significantly different.
    Assumptions
        Observations in each sample are independent and identically distributed (iid).
        Observations in each sample are normally distributed.
        Observations in each sample have the same variance.
    Interpretation
        H0: the means of the samples are equal.
        H1: the means of the samples are unequal.
    """ 
    stat, p = ttest_ind(sample1, sample2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution Student’s t-test')
    else:
        print('Probably different distributions Student’s t-test')

    return stat, p

def PairedStudentTTest(sample1, sample2):
    """Paired Student’s t-test
    Tests whether the means of two paired samples are significantly different.
    Assumptions
        Observations in each sample are independent and identically distributed (iid).
        Observations in each sample are normally distributed.
        Observations in each sample have the same variance.
        Observations across each sample are paired.
    Interpretation
        H0: the means of the samples are equal.
        H1: the means of the samples are unequal.
    """ #ttest_rel
    stat, p = ttest_rel(sample1, sample2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution Paired Student’s t-test')
    else:
        print('Probably different distributions Paired Student’s t-test')
    
    return stat, p

def AnovaTest(sample1, sample2, sample3):
    """Analysis of Variance Test (ANOVA)
    Tests whether the means of two or more independent samples are significantly different.
    Assumptions
        Observations in each sample are independent and identically distributed (iid).
        Observations in each sample are normally distributed.
        Observations in each sample have the same variance.
    Interpretation
        H0: the means of the samples are equal.
        H1: one or more of the means of the samples are unequal.
    """
    stat, p = f_oneway(sample1, sample2, sample3)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution Analysis of Variance Test (ANOVA)')
    else:
        print('Probably different distributions Analysis of Variance Test (ANOVA)')

    return stat, p

def MannWhitneyTest(sample1, sample2):
    """Mann-Whitney U Test
    Tests whether the distributions of two independent samples are equal or not.
    Assumptions
        Observations in each sample are independent and identically distributed (iid).
        Observations in each sample can be ranked.
    Interpretation
        H0: the distributions of both samples are equal.
        H1: the distributions of both samples are not equal.
    """
    stat, p = mannwhitneyu(sample1, sample2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution Mann-Whitney U Test')
    else:
        print('Probably different distributions Mann-Whitney U Test')
    
    return stat, p
def WilcoxonSignedRankTest(sample1, sample2):
    """Wilcoxon Signed-Rank Test
    Tests whether the distributions of two paired samples are equal or not.
    Assumptions
        Observations in each sample are independent and identically distributed (iid).
        Observations in each sample can be ranked.
        Observations across each sample are paired.
    Interpretation
        H0: the distributions of both samples are equal.
        H1: the distributions of both samples are not equal.
    """ #wilcoxon
    stat, p = wilcoxon(sample1, sample2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution Wilcoxon Signed-Rank Test')
    else:
        print('Probably different distributions Wilcoxon Signed-Rank Test')

    return stat, p

def KruskalWallisHTest(sample1, sample2):
    """Kruskal-Wallis H Test
    Tests whether the distributions of two or more independent samples are equal or not.
    Assumptions
        Observations in each sample are independent and identically distributed (iid).
        Observations in each sample can be ranked.
    Interpretation
        H0: the distributions of all samples are equal.
        H1: the distributions of one or more samples are not equal.
    """
    stat, p = kruskal(sample1, sample2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution Kruskal-Wallis H Test')
    else:
        print('Probably different distributions Kruskal-Wallis H Test')

    return stat, p

def FriedmanTest(sample1, sample2, sample3):
    """Friedman Test
    Tests whether the distributions of two or more paired samples are equal or not.
    Assumptions
        Observations in each sample are independent and identically distributed (iid).
        Observations in each sample can be ranked.
        Observations across each sample are paired.
    Interpretation
        H0: the distributions of all samples are equal.
        H1: the distributions of one or more samples are not equal.
    """
    stat, p = friedmanchisquare(sample1, sample2, sample3)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution Friedman Test')
    else:
        print('Probably different distributions Friedman Test')

    return stat, p

#### ----------------------------------------------------------    
sample0 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
sample1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
sample2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]
sample3 = [-0.208, 0.696, 0.928, -1.148, -0.213, 0.229, 0.137, 0.269, -0.870, -1.204]
table = [[10, 20, 30],[6,  9,  17]]
# ShapiroWilkTest(sample1)
# DAgostinoKTwoTest(sample1)
# AndersonDarlingTest(sample1)
# PearsonCorrCoeff(sample1, sample2)
# SpearmanRankCorrelation(sample1, sample2)
# KendalRankCorr(sample1, sample2)
# ChiSquaredTest(table)
# AugmentedDickeyFullerURTest(sample0)
# KwiatkowskiPhillipsSchmidtShin(sample0)
# StudentTTest(sample1, sample2)
# PairedStudentTTest(sample1, sample2)
# AnovaTest(sample1, sample2, sample3)
# MannWhitneyTest(sample1, sample2)
# WilcoxonSignedRankTest(sample1, sample2)
# KruskalWallisHTest(sample1, sample2)
# FriedmanTest(sample1, sample2, sample3)