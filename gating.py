import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class Smooth(object):
    """
    Methods fpr piece-wise smooth function
    """
    def __init__(self, low = 0.0, high = 1.0):
        """
        Parameters:
        low float
            for x < low, the return value is 0
        high    gloat
            for x > high, the return value is 1
        """
        self.low = low
        self.high = high
    def eval(self, x):
        xp = (x - self.low) / (self.high - self.low)
        y = 3.0*np.power(xp,2) - 2.0*np.power(xp,3)
        y[x < self.low] = 0
        y[x > self.high] = 1
        return y
    def count_center(self, x):
        return np.sum((x >= self.low) & (x <= self.high))
    def count_low(self, x):
        return np.sum((x <= self.low))
    def count_high(self, x):
        return np.sum((x >= self.high))

class BiExp(object):
    """
    Methods for Bi-exponential transformations
    """
    def __init__(self, a, b, c, d, f, min_x, max_x, n_th = 1000):
        """
        Parameters
        ----------
        a,b,c,d,f   float
            Parameters of the biexp transformation
        min_x.max_x float
            Minimum/maximum value of the range used for inverting the biexp function
        n_th    int
            Number of interpolating values used for function inversion
        """
        from scipy.optimize import minimize
        from scipy.interpolate import interp1d
        if (a <= 0.0):
            raise ValueError('ERROR: wrong parameter in BiExp tranformation a = {0:f}'.format(a))
        if (c <= 0.0):
            raise ValueError('ERROR: wrong parameter in BiExp tranformation c = {0:f}'.format(c))
        if (b < 0.0):
            raise ValueError('ERROR: wrong parameter in BiExp tranformation b = {0:f}'.format(b))
        if (d < 0.0):
            raise ValueError('ERROR: wrong parameter in BiExp tranformation d = {0:f}'.format(d))
        if (min_x >= max_x):
            raise ValueError('ERROR: wrong parameters in BiExp tranformation')
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.f = f
        opt = minimize(fun = lambda y: (self.back(y) - min_x)**2.0, x0 = 0.0, method = 'Nelder-Mead', options = {'maxiter':1e8}, tol = 1e-3)
        if not opt.success:
            raise ValueError('ERROR: convergence problem in BiExp inversion - min range')
        min_y = opt.x
        opt = minimize(fun = lambda y: (self.back(y) - max_x)**2.0, x0 = 10.0, method = 'Nelder-Mead', options = {'maxiter':1e8}, tol = 1e-3)
        if not opt.success:
            raise ValueError('ERROR: convergence problem in BiExp inversion - max range')
        max_y = opt.x
        y_th = np.linspace(min_y, max_y, n_th)
        x_th = self.back(y_th)
        self.forward = interp1d(x_th.flatten(), y_th.flatten(), fill_value = 'extrapolate')
    def back(self, y):
        return self.a * np.exp(self.b*y) - self.c * np.exp(-self.d*y) + self.f

class Logicle(BiExp):
    """
    Methods for logicle transformation
    """
    def __init__(self, a, b, c, d, f, min_x, max_x, x1, n_th = 1000):
        self.x1 = x1
        super(Logicle, self).__init__(a, b, c, d, f, min_x, max_x, n_th)
    def back(self, y):
        if isinstance(y, float):
            x = np.empty(1)
            y = np.array(y)
        elif isinstance(y, np.ndarray):
            x = np.empty(y.shape)
        elif isinstance(y, list):
            y = np.array(y)
            x = np.empty(y.shape)
        else:
            raise ValueError('ERROR: wrong data format in logicle transformation')
        inds_pos = y >= self.x1
        if np.any(inds_pos):
            x[inds_pos] = self.a * np.exp(self.b*y[inds_pos]) - self.c * np.exp(-self.d*y[inds_pos]) + self.f
        inds_neg = np.logical_not(inds_pos)
        if np.any(inds_neg):
            x[inds_neg] = - (self.a * np.exp(self.b*(2*self.x1-y[inds_neg])) - self.c * np.exp(-self.d*(2*self.x1-y[inds_neg])) + self.f) 
        return x

class Linear(object):
    """
    Methods for linear transformation (so nothing...)
    """
    def forward(self, x):
        return x
    def back(self, y):
        return y

class FcsData(object):
    """
    Methods for a set of FCS data

    Attributes
    ----------
    name    str
        name of the dataset
    fcs dict
        key    str
            Sample names
        value  FCS
    features    list
        The list of features used for the analyses
    data    pd.DataFrame
        Dataframe with columns:
            sample_names
            features in self.features --> raw or transformed data (after self.apply_transform)
            p_[feature] --> probability of 1D classification (after self.gate1D)
    transform   dict
        key str
            Feature names
        value   transforming class
    thrs    dict
    ups     dict
    downs   dict
    steps   dict
        All these attributes are defined by self.gate1D
        They are dicts with key the feature name and values:
            - gating threshold
            - lower boundary of the uncertainity classification range
            - upper boundary of the uncertainity classification range
            - the class used to calculate the clustering probability 
    """
    def __init__(self):
        self.name = ''
        self.fcs = {}
        self.features = []
        self.data = pd.DataFrame()
        self.transform = {}
        self.thrs = {}
        self.ups = {}
        self.downs = {}
        self.steps = {}
    def add_sample(self, file_name, sample_name = ''):
        from FlowCytometryTools import FCMeasurement   
        if not os.path.exists(file_name):
            raise ValueError('ERROR: {} does not exist'.format(file_name))
        self.name = sample_name
        if not self.name:
            self.name = os.path.basename(file_name)
        sample = FCMeasurement(ID = self.name, datafile = file_name)
        self.fcs[sample_name] = sample
    def set_data(self, features):
        self.features = features
        self.data = pd.DataFrame(columns = ['sample',]+self.features)
        for name, sample in self.fcs.items():
            data_sample = sample.data[self.features]
            data_sample['sample'] = name
            self.data = pd.concat((self.data, data_sample), axis = 0)
    def fit_transform(self, mode = 'logicle', **kwargs):
        """
        Two transformations are implemented:
        * logicle
            If not provided as kwargs parameters optimal parameters are automatically calculated
        * linear
            This is an identity transformation

        Return
        ------
        dict
            key str
                Name of the parameter
            value   float
                Value of the parameter
        
        These are the parameters of the transformation
        It is useful if parameters are first calculated and then applied (the same ?) to other datasets
        """
        parm_transform = {}
        if 'features' in kwargs:
            iter_features = kwargs['features']
        else:
            iter_features = self.features
        for feature in iter_features:
            data = self.data[feature]
            if mode == 'logicle':
                from scipy.optimize import minimize
                #--- A,M,T,W parameters
                log_prm_A = kwargs.get(feature+'_A', 0.0)*np.log(10)
                log_prm_M = kwargs.get(feature+'_M', 4.5)*np.log(10)
                log_prm_T = kwargs.get(feature+'_T',np.max(data.values))
                if feature+'_W' in kwargs:
                    log_prm_W = kwargs[feature+'_W']
                else:
                    neg_values = data[data<0]
                    if len(neg_values):
                        log_prm_W = 0.5*(log_prm_M - np.log(log_prm_T/np.abs(np.percentile(neg_values,5.0)))) 
                    else:
                        log_prm_W = -10
                #print('Logicle parameters for {} - A {}, M {}, T {}, W {}'.format(feature, log_prm_A, log_prm_M, log_prm_T, log_prm_W))
                #--- Compute a,b,c,d,f
                b = log_prm_A + log_prm_M
                opt = minimize(fun= lambda x: ((log_prm_W / (log_prm_A + log_prm_M)) - 2.0 * (np.log(b) - np.log(x)) / (b + x) )**2.0
                    , x0 = 1.0, method = 'Nelder-Mead', options = {'maxiter':1e8}, tol = 1e-3)
                if not opt.success:
                    raise ValueError('ERROR: convergence problem in Logicle inversion - d calculation')
                d = opt.x[0]
                x2 = log_prm_A / (log_prm_A + log_prm_M)
                x1 = x2 + log_prm_W / (log_prm_A + log_prm_M)
                x0 = x1 + log_prm_W / (log_prm_A + log_prm_M)
                c_a = np.exp(x0*(b+d))
                f_a = np.exp(x0*(b+d)-x1*d) - np.exp(b*x1)
                log_prm_a = log_prm_T / (np.exp(b) - c_a*np.exp(-d) + f_a)
                log_prm_b = b
                log_prm_c = c_a * log_prm_a
                log_prm_d = d
                log_prm_f = f_a * log_prm_a
                #print('Logicle parameters for {} - a {}, b {}, c {}, d {}, f {}'.format(feature, log_prm_a, log_prm_b, log_prm_c, log_prm_d, log_prm_f))
                #--- Initialize the biexp function
                self.transform[feature] = Logicle(log_prm_a, log_prm_b, log_prm_c, log_prm_d, log_prm_f, np.min(data), np.max(data), x1)
                parm_transform[feature+'_A'] = log_prm_A/np.log(10)
                parm_transform[feature+'_M'] = log_prm_M/np.log(10)
                parm_transform[feature+'_T'] = log_prm_T
                parm_transform[feature+'_W'] = log_prm_W
            elif mode == 'linear':
                parm_transform = {}
                self.transform[feature] = Linear()
            else:
                raise NotImplemented('ERROR: {} is not implemented'.format(mode))
        return parm_transform
    def apply_transform(self):
        for feature in self.features:
            self.data[feature] = self.transform[feature].forward(self.data[feature])
    def show2D(self, features, nbins = 100, pdf = None):
        """
        2D histogram plotting on PDF file
        """
        import numpy.ma as ma
        if len(features) != 2:
            raise ValueError('ERROR: 2 features are needed for show2D')
        H, xe, ye = np.histogram2d(self.data[features[0]], self.data[features[1]], density = False, bins = nbins)
        xb = 0.5*(xe[1:]+xe[:-1])
        yb = 0.5*(ye[1:]+ye[:-1])
        X, Y = np.meshgrid(xb,yb)
        iX, iY = np.meshgrid(np.arange(len(xb)), np.arange(len(yb)))
        H = H.transpose()
        H[H == 0] = np.nan
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
        cax = ax.pcolormesh(X, Y, np.log10(H), cmap = 'inferno', shading = 'nearest')
        f.colorbar(cax)
        #--- change axis labels to actual values / x-axis
        possible_ticks = [-1000,-100,-10,0,10,100,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,200000,300000,400000,500000,600000,700000,800000,900000]
        possible_ticklabels = ['-10^3','-10^2','-10^-1','0','10^1','10^2','10^3','','','','','','','','','10^4','','','','','','','','','10^5','','','','','','','','']
        x_norm_min, x_norm_max = ax.get_xlim()
        x_min, x_max = self.transform[features[0]].back([x_norm_min, x_norm_max])
        x_ticks = []
        x_ticklabels = []
        for ind, x_tick in enumerate(possible_ticks):
            if (x_tick > x_min) and (x_tick < x_max):
                x_ticks.append(x_tick)
                x_ticklabels.append(possible_ticklabels[ind])
        x_ticks = np.array(x_ticks)
        x_ticklabels = np.array(x_ticklabels)
        x_ticks_norm = self.transform[features[0]].forward(x_ticks)
        inds = np.isfinite(x_ticks_norm)
        ax.set_xticks(x_ticks_norm[inds])
        ax.set_xticklabels(x_ticklabels[inds])
        #--- change labels to actual values / y-axis
        y_norm_min, y_norm_max = ax.get_ylim()
        y_min, y_max = self.transform[features[1]].back([y_norm_min, y_norm_max])
        y_ticks = []
        y_ticklabels = []
        for ind, y_tick in enumerate(possible_ticks):
            if (y_tick > y_min) and (y_tick < y_max):
                y_ticks.append(y_tick)
                y_ticklabels.append(possible_ticklabels[ind])
        y_ticks = np.array(y_ticks)
        y_ticklabels = np.array(y_ticklabels)
        y_ticks_norm = self.transform[features[1]].forward(y_ticks)
        inds = np.isfinite(y_ticks_norm)
        ax.set_yticks(y_ticks_norm[inds])
        ax.set_yticklabels(y_ticklabels[inds])
        #--- save&close figure
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title(self.name)
        if pdf is not None:
            pdf.savefig()
            plt.close()
        return ax
    def gate1D(self, pdf, feature, manual_gate, nbins = 100):
        """
        Performs gating along feature

        Parameters
        ----------
        pdf PdfPages
            Where plots are saved
        feature str
            The feature used for gating
        nbins   int
            This is used to distretize the density (and so to calculate density peaks and minimum)
        manual_gate float
            The gate cannot be lower than this value
        """
        from sklearn.cluster import KMeans
        #---  kmeans clustering in 2 clusters
        s = self.data[feature].values.reshape((-1,1))
        cls = KMeans(n_clusters = 2)
        cls.fit(s)
        cnt0 = cls.cluster_centers_.flatten().min()
        cnt1 = cls.cluster_centers_.flatten().max()
        if cls.cluster_centers_.flatten()[0] < cls.cluster_centers_.flatten()[1]:
            cls_low, cls_high = 0, 1
        else:
            cls_low, cls_high = 1, 0
        #--- histograms of low/high clusters
        h, xe = np.histogram(s, density = False, bins = nbins)
        xb = 0.5*(xe[1:]+xe[:-1])
        h1, dummy = np.histogram(s[cls.labels_ == cls_low], density = False, bins = xe)
        h2, dummy = np.histogram(s[cls.labels_ == cls_high], density = False, bins = xe)
        #--- treshold as minimum in density between the two cluster centers
        self.thrs[feature] = xb[(xb > cnt0) & (xb < cnt1)][np.argmin(h[(xb > cnt0) & (xb < cnt1)])]
        h_thr = h[(xb > cnt0) & (xb < cnt1)][np.argmin(h[(xb > cnt0) & (xb < cnt1)])]
        h1_cnt = h1[(xb > cnt0) & (xb < cnt1)][np.argmax(h1[(xb > cnt0) & (xb < cnt1)])]
        h2_cnt = h2[(xb > cnt0) & (xb < cnt1)][np.argmax(h2[(xb > cnt0) & (xb < cnt1)])]
        delta_h1 = h_thr / h1_cnt
        delta_h2 = h_thr / h2_cnt
        norm_h = np.max(h)
        h = h/norm_h
        h1 = h1/norm_h
        h2 = h2/norm_h
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
        ax.plot(xb, h, '-b')
        ax.plot(xb, h1, '-g')
        ax.plot(xb, h2, '-m')
        #--- region of uncertain classification
        self.downs[feature] = self.thrs[feature] - delta_h1*(self.thrs[feature]-cnt0)
        self.ups[feature] = self.thrs[feature] + delta_h2*(cnt1-self.thrs[feature])
        #--- in case manual gate is defined check that the threshold is above, otherwise apply manual gate
        manual_gate_trans = self.transform[feature].forward(manual_gate)
        ax.plot([manual_gate_trans, manual_gate_trans], [0, h.max()], '-b', label = 'manual gate')
        if self.thrs[feature] < manual_gate_trans:
            ax.plot([self.thrs[feature], self.thrs[feature]], [0, h.max()], '--r', label = 'gate before correction')
            self.downs[feature] = manual_gate_trans - 1e-4*(np.max(s) - np.min(s))
            self.ups[feature] = manual_gate_trans + 1e-4*(np.max(s) - np.min(s))
            self.thrs[feature] = manual_gate_trans
        #--- define the smoothing function
        self.steps[feature] = Smooth(low = self.downs[feature], high = self.ups[feature])
        #--- add classification probability into high cluster to self.data
        self.data['p_{}'.format(feature)] = self.steps[feature].eval(self.data[feature].values)
        pc =  np.mean(np.power(np.round(self.data['p_{}'.format(feature)]) - self.data['p_{}'.format(feature)], 2.0))
        ax.plot([cnt0, cnt0], [0, h.max()],'--m', label = 'center cluster_0')
        ax.plot([self.downs[feature], self.downs[feature]], [0, h.max()],'--k', label = '< --> p=0')
        ax.plot([self.thrs[feature], self.thrs[feature]], [0, h.max()],':r', label = 'gate')
        ax.plot([cnt1, cnt1], [0, h.max()],':m', label = 'center cluster_1')
        ax.plot([self.ups[feature], self.ups[feature]], [0, h.max()],':k', label = '> --> p=1')
        ax.plot(xb, self.steps[feature].eval(xb), '-y', label = 'p')
        plt.xlabel(feature)
        plt.ylabel('#cells')
        plt.legend()
        plt.ylim([0,1.1])
        #--- change axis labels to actual values / x-axis
        possible_ticks = [-1000,-100,-10,0,10,100,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,200000,300000,400000,500000,600000,700000,800000,900000]
        possible_ticklabels = ['-10^3','-10^2','-10^-1','0','10^1','10^2','10^3','','','','','','','','','10^4','','','','','','','','','10^5','','','','','','','','']
        x_norm_min, x_norm_max = ax.get_xlim()
        x_min, x_max = self.transform[feature].back([x_norm_min, x_norm_max])
        x_ticks = []
        x_ticklabels = []
        for ind, x_tick in enumerate(possible_ticks):
            if (x_tick > x_min) and (x_tick < x_max):
                x_ticks.append(x_tick)
                x_ticklabels.append(possible_ticklabels[ind])
        x_ticks = np.array(x_ticks)
        x_ticklabels = np.array(x_ticklabels)
        x_ticks_norm = self.transform[feature].forward(x_ticks)
        inds = np.isfinite(x_ticks_norm)
        ax.set_xticks(x_ticks_norm[inds])
        ax.set_xticklabels(x_ticklabels[inds])
        plt.title('{0:s}\nPC {1:6.3e}'.format(self.name, pc))
        pdf.savefig()
        plt.close()
    def gate2D(self, pdf, features, manual_gate, nbins = 100, what = 'or'):
        """
        Performs gating along features

        Parameters
        ----------
        pdf PdfPages
            Where plots are saved
        features list of  str
            The features used for gating
        manual_gate list of float
            Threshold for manual gate
            Here it is used for plotting only
        nbins   int
            This is used for plotting only
        what    str
            and = both features high
            or = one or two features high
        """
        if len(features) != 2:
            raise ValueError('ERROR: 2 features are needed for gate2D')
        if len(manual_gate) != 2:
            raise ValueError('ERROR: 2 manual gate thresholds are needed for gate2D')
        if not all(['p_'+feature in self.data.columns for feature in features]):
            raise ValueError('ERROR: missing probability values for 1D gating, first run self.gate1D for all the features')
        if (features[0] not in self.thrs) or (features[1] not in self.thrs):
            raise ValueError('ERROR: missing threshold values for 1D gating, first run self.gate1D for all the features')
        H, xe, ye = np.histogram2d(self.data[features[0]], self.data[features[1]], density = False, bins = nbins)
        xb = 0.5*(xe[1:]+xe[:-1])
        yb = 0.5*(ye[1:]+ye[:-1])
        X, Y = np.meshgrid(xb,yb)
        iX, iY = np.meshgrid(np.arange(len(xb)), np.arange(len(yb)))
        H = H.transpose()
        if what == 'and': 
            p_c1 = self.data['p_'+features[0]] * self.data['p_'+features[1]]
            p_c0 = 1-p_c1.values
            def f_p_c1(x, y):
                p_feature0 =  self.steps[features[0]].eval(x)
                p_feature1 =  self.steps[features[1]].eval(y)
                return p_feature0 * p_feature1
            def f_p_c0(x, y):
                p_feature0 = self.steps[features[0]].eval(x)
                p_feature1 = self.steps[features[1]].eval(y)
                return 1 - (p_feature0 * p_feature1)
        elif what == 'or':
            p_c1 = 1.0 - ((1-self.data['p_'+features[0]]) * (1-self.data['p_'+features[1]]))
            p_c0 = 1-p_c1.values
            def f_p_c1(x, y):
                p_feature0 = 1.0 - self.steps[features[0]].eval(x)
                p_feature1 = 1.0 - self.steps[features[1]].eval(y)
                return 1.0 - (p_feature0 * p_feature1)
            def f_p_c0(x, y):
                p_feature0 = 1 - self.steps[features[0]].eval(x)
                p_feature1 = 1 - self.steps[features[1]].eval(y)
                return p_feature0 * p_feature1
        else:
            raise ValueError('ERROR: unknown gating method')
        #--- computing averages 
        data_feature0_back = self.transform[features[0]].back(self.data[features[0]].values.flatten())
        data_feature1_back = self.transform[features[1]].back(self.data[features[1]].values.flatten())
        fuzzy = np.average(data_feature0_back, weights = p_c1), np.average(data_feature1_back, weights = p_c1), np.sum(p_c1)/len(p_c1)
        inds_or_gate = (data_feature0_back > manual_gate[0]) | (data_feature1_back > manual_gate[1])
        or_gate = np.mean(data_feature0_back[inds_or_gate]), np.mean(data_feature1_back[inds_or_gate]), np.sum(inds_or_gate)/len(inds_or_gate)
        or_gate_geom = self.transform[features[0]].back(np.exp(np.nanmean(np.log(self.data[features[0]].values[inds_or_gate]))))[0], self.transform[features[1]].back(np.exp(np.nanmean(np.log(self.data[features[1]].values[inds_or_gate]))))[0], np.sum(inds_or_gate)/len(inds_or_gate)
        or_gate_median = np.median(data_feature0_back[inds_or_gate]), np.median(data_feature1_back[inds_or_gate]), np.sum(inds_or_gate)/len(inds_or_gate)
        #--- plotting
        ax = self.show2D(features, nbins)
        ax.plot([self.downs[features[0]], self.downs[features[0]]],[np.min(yb), np.max(yb)],'--k', label = '< --> p=0')
        ax.plot([self.ups[features[0]], self.ups[features[0]]],[np.min(yb), np.max(yb)],':k', label = '> --> p=1')
        ax.plot([np.min(xb), np.max(xb)],[self.downs[features[1]], self.downs[features[1]]],'--k', label = '< --> p=0')
        ax.plot([np.min(xb), np.max(xb)],[self.ups[features[1]], self.ups[features[1]]],':k', label = '> --> p=1')
        ax.plot(self.transform[features[0]].forward([manual_gate[0], manual_gate[0]]),[np.min(yb), np.max(yb)],'--b', label = 'manual gate')
        ax.plot([np.min(xb), np.max(xb)],self.transform[features[1]].forward([manual_gate[1], manual_gate[1]]),'--b', label = 'manual gate')
        ax.plot(self.transform[features[0]].forward(or_gate[0]), self.transform[features[1]].forward(or_gate[1]), '^g', label = 'or')
        ax.plot(self.transform[features[0]].forward(or_gate_geom[0]), self.transform[features[1]].forward(or_gate_geom[1]), '<g', label = 'or geom.')
        ax.plot(self.transform[features[0]].forward(or_gate_median[0]), self.transform[features[1]].forward(or_gate_median[1]), 'vg', label = 'or median')
        ax.plot(self.transform[features[0]].forward(fuzzy[0]), self.transform[features[1]].forward(fuzzy[1]), 'Pg', label = 'fuzzy')
        ax.plot([self.thrs[features[0]], self.thrs[features[0]]],[np.min(yb), np.max(yb)],':r', label = 'gate')
        ax.plot([np.min(xb), np.max(xb)],[self.thrs[features[1]], self.thrs[features[1]]],':r', label = 'gate')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title(self.name)
        plt.legend()
        pdf.savefig()
        plt.close()
        return or_gate, or_gate_geom, or_gate_median, fuzzy
    def gate3D(self, pdf, features, nbins = 100, what = 'not_off_off'):
        """
        Performs gating along features

        Parameters
        ----------
        pdf PdfPages
            Where plots are saved
        features list of  str
            The features used for gating
        nbins   int
            This is used for plotting only
        what    str
            and = both features high
            or = one or two features high
        """
        if len(features) != 3:
            raise ValueError('ERROR: 3 features are needed for hist3D')
        if (features[0] not in self.thrs) or (features[1] not in self.thrs) or (features[2] not in self.thrs):
            raise ValueError('ERROR: missing threshold values for 1D gating, first run self.gate1D for all the features')
        if not all(['p_'+feature in self.data.columns for feature in features]):
            raise ValueError('ERROR: missing probability values for 1D gating, first run self.gate1D for all the features')
        if what == 'or':
            p_c1 = 1.0 - ((1-self.data['p_'+features[0]]) * (1-self.data['p_'+features[1]]) * (1-self.data['p_'+features[2]]))
            p_c0 = 1-p_c1.values
            def f_p_c1(x, y, z):
                p_feature0 = 1.0 - self.steps[features[0]].eval(x)
                p_feature1 = 1.0 - self.steps[features[1]].eval(y)
                p_feature2 = 1.0 - self.steps[features[2]].eval(z)
                return 1.0 - (p_feature0 * p_feature1 * p_feature2)
            def f_p_c0(x, y, z):
                p_feature0 = 1 - self.steps[features[0]].eval(x)
                p_feature1 = 1 - self.steps[features[1]].eval(y)
                p_feature2 = 1 - self.steps[features[2]].eval(z)
                return p_feature0 * p_feature1 * p_feature2
        elif what == 'and':
            p_c1 = self.data['p_'+features[0]] * self.data['p_'+features[1]] * self.data['p_'+features[2]]
            p_c0 = 1-p_c1.values
            def f_p_c1(x, y, z):
                p_feature0 = self.steps[features[0]].eval(x)
                p_feature1 = self.steps[features[1]].eval(y)
                p_feature2 = self.steps[features[2]].eval(z)
                return  (p_feature0 * p_feature1 * p_feature2)
            def f_p_c0(x, y, z):
                p_feature0 = self.steps[features[0]].eval(x)
                p_feature1 = self.steps[features[1]].eval(y)
                p_feature2 = self.steps[features[2]].eval(z)
                return 1.0 - p_feature0 * p_feature1 * p_feature2
        else:
            raise ValueError('ERROR: unknown gating method')
        data_feature0_back = self.transform[features[0]].back(self.data[features[0]].values.flatten())
        data_feature1_back = self.transform[features[1]].back(self.data[features[1]].values.flatten())
        data_feature2_back = self.transform[features[2]].back(self.data[features[2]].values.flatten())
        fuzzy = np.average(data_feature0_back, weights = p_c1), np.average(data_feature1_back, weights = p_c1), np.average(data_feature2_back, weights = p_c1), np.sum(p_c1)/len(p_c1)
        #--- plotting
        H, xe, ye = np.histogram2d(self.data[features[0]], self.data[features[1]], density = False, bins = nbins)
        xb = 0.5*(xe[1:]+xe[:-1])
        yb = 0.5*(ye[1:]+ye[:-1])
        X, Y = np.meshgrid(xb,yb)
        iX, iY = np.meshgrid(np.arange(len(xb)), np.arange(len(yb)))
        H = H.transpose()
        ax = self.show2D([features[0], features[1]], nbins)
        ax.plot(self.transform[features[0]].forward(fuzzy[0]), self.transform[features[1]].forward(fuzzy[1]), 'Pg', label = 'fuzzy')
        ax.plot([self.thrs[features[0]], self.thrs[features[0]]],[np.min(yb), np.max(yb)],':r', label = 'gate')
        ax.plot([np.min(xb), np.max(xb)],[self.thrs[features[1]], self.thrs[features[1]]],':r', label = 'gate')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title(self.name)
        plt.legend()
        pdf.savefig()
        plt.close()
        H, xe, ye = np.histogram2d(self.data[features[1]], self.data[features[2]], density = False, bins = nbins)
        xb = 0.5*(xe[1:]+xe[:-1])
        yb = 0.5*(ye[1:]+ye[:-1])
        X, Y = np.meshgrid(xb,yb)
        iX, iY = np.meshgrid(np.arange(len(xb)), np.arange(len(yb)))
        H = H.transpose()
        ax = self.show2D([features[1], features[2]], nbins)
        ax.plot(self.transform[features[1]].forward(fuzzy[1]), self.transform[features[2]].forward(fuzzy[2]), 'Pg', label = 'fuzzy')
        ax.plot([self.thrs[features[1]], self.thrs[features[1]]],[np.min(yb), np.max(yb)],':r', label = 'gate')
        ax.plot([np.min(xb), np.max(xb)],[self.thrs[features[2]], self.thrs[features[2]]],':r', label = 'gate')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title(self.name)
        plt.legend()
        pdf.savefig()
        plt.close()
        return fuzzy
    def __str__(self):
        output = 'FcsData item\n'
        for name, sample in self.fcs.items():
            output += '\tsample {}\n'.format(name)
        return output[:-1]
