from re import X
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import PurePath

# from engine.inferer import plotGraph

# TODO convert into class
class EpochDataProcessor:
    def __init__(self, classes_labels = None, are_class_labels_centered = True):
        # initialize tables
        self.exact_percentage = np.array([])
        self.outliers_percentage = np.array([])
        self.F0_error_median_all = np.array([])
        self.F0_error_mean_all = np.array([])
        self.F0_error_std_all = np.array([])
        self.F0_error_relative_median_all = np.array([])
        self.F0_error_relative_mean_all = np.array([])
        self.F0_error_relative_std_all = np.array([])
        self.train_accuracy_all = np.array([])
        self.trained_SNR_dB_all = np.array([]) 

        self.no_of_data_per_class_all = list()
        self.no_of_outliers_per_class_all = list()
        self.est_mean_error_per_class_all = list()
        self.est_std_error_per_class_all = list()

        self.set_classes_labels(classes_labels, are_class_labels_centered)
        
        self.epoch_indexes = []

    def reset(self, classes_labels, are_class_labels_centered):
        self.__init__(classes_labels, are_class_labels_centered)

    def set_classes_labels(self, classes_labels, are_class_labels_centered = True):
        self.classes_labels = classes_labels

        # average distance between classes (average class width)
        if classes_labels is not None:
            self.classes_dF0 = np.mean(np.array(self.classes_labels)[1:]-np.array(self.classes_labels[0:-1]))
            self.no_of_classes = len(self.classes_labels)
        else:
            self.classes_dF0 = 0
            self.no_of_classes = 0
        # self.classes_dF0 /= 3 # TODO remove later / just for tests

        self.are_class_labels_centered = are_class_labels_centered
        if self.are_class_labels_centered == False:
            self.F0_data_hist_edges = []

            if self.no_of_classes > 0:
                self.F0_data_hist_edges = self.classes_labels
                closing_edge = self.classes_labels[-1] + self.classes_dF0
                self.F0_data_hist_edges = np.append(self.F0_data_hist_edges, closing_edge)

        else:
            self.F0_data_hist_edges = []

            if self.no_of_classes > 0:
                self.F0_data_hist_edges = self.classes_labels - self.classes_dF0/2
                closing_edge = self.classes_labels[-1] + self.classes_dF0/2
                self.F0_data_hist_edges = np.append(self.F0_data_hist_edges, closing_edge)

        self.F0_error_hist_edges = []

        if self.no_of_classes > 0:
            no_of_error_bins = self.no_of_classes + 1 - (self.no_of_classes % 2)

            self.F0_error_hist_edges = [0] * (no_of_error_bins + 1)
            for n in range(no_of_error_bins):
                center = (n - (no_of_error_bins-1)/2) * self.classes_dF0
                edge = center - self.classes_dF0/2

                self.F0_error_hist_edges[n] = edge

            center = (no_of_error_bins-1)/2 * self.classes_dF0
            edge = center + self.classes_dF0/2
            self.F0_error_hist_edges[-1] = edge


    def get_classes_labels(self):
        return self.classes_labels

    # TODO move save_path to constructor
    def append_epoch_data(self, infer_results, train_accuracy, trained_SNR_dB, epoch_ind = -1, do_draw_epoch = False):
        logging.info("Processing epoch data")

        self.epoch_indexes.append(epoch_ind)
        epoch_str = "epoch_" + str(epoch_ind)

        # TODO F0_ref - get label

        # # TODO check this: should this be F0_ref_labels or simply F0_est?
        # F0_ref_labels_unique = np.unique(np.array(infer_results["F0_ref_labels"]))
        # F0_ref_labels_diff = np.unique(np.array(F0_ref_labels_unique[1:] - F0_ref_labels_unique[:-1]))
        # dF0_ref_labels= sorted(F0_ref_labels_diff)[:2]
        #
        # logging.info(f"F0_ref_labels_unique: {F0_ref_labels_unique}")
        # logging.info(f"dF0_ref_labels: {dF0_ref_labels}")

        ok_mask = infer_results["F0_ref"] > 0
        F0_ref = infer_results["F0_ref"][ok_mask]
        F0_est = infer_results["F0_est"][ok_mask]
        if infer_results["F0_ref_labels"] is not None:
            F0_ref_labels = infer_results["F0_ref_labels"][ok_mask]
        else:
            F0_ref_labels = None

        # using F0_ref, F0_est, F0_ref_labels instead of infer_results["F0_ref"], ...
        
        outliers_percentage_threshold = 0.2 # in percents

        F0_error = F0_ref-F0_est
        if F0_ref_labels is not None:
            F0_class_error = F0_ref_labels-F0_est
            F0_exact = F0_error[(F0_class_error == 0) & (F0_ref > 0)]
        else:
            F0_exact = None

        outliers_mask = abs(F0_error)/F0_ref > outliers_percentage_threshold
        no_outliers_mask = abs(F0_error)/F0_ref <= outliers_percentage_threshold
        F0_error_outliers = F0_error[outliers_mask]
        F0_error_no_outliers = F0_error[no_outliers_mask] # exclude outliers

        # ============================================
        # TODO use the following
        F0_error_median = np.median(F0_error)
        F0_error_mean = np.mean(F0_error_no_outliers)
        F0_error_std = np.std(F0_error_no_outliers)
        # F0_error_mean = np.mean(F0_error)
        # F0_error_std = np.std(F0_error)

        F0_log_relative = np.log(F0_est / F0_ref)
        
        F0_error_relative = F0_error / F0_ref
        # TODO F0_error_relative_outliers > +/-5% or 10% error
        F0_error_relative_no_outliers = F0_error_relative[no_outliers_mask]
        F0_error_relative_median = np.median(F0_error_relative)
        F0_error_relative_mean = np.mean(F0_error_relative_no_outliers)
        F0_error_relative_std = np.std(F0_error_relative_no_outliers)
        # F0_error_relative_mean = np.mean(F0_error_relative)
        # F0_error_relative_std = np.std(F0_error_relative)

        # TODO analyse if these can be used for overall network evaluation
        # weighted error: mean + std + outliers
        F0_weighted_error = (F0_error_mean * F0_error_mean) + (F0_error_std * F0_error_std) # what with outliers ???
        # weighted relative error: mean + std + outliers
        F0_weighted_relative_error = (F0_error_relative_mean * F0_error_relative_mean) + (F0_error_relative_std * F0_error_relative_std) # what with outliers ???

        # ============================================

        if F0_exact is not None:
            # self.exact_percentage = np.append(self.exact_percentage, 100*F0_exact.size/F0_error[(F0_ref > 0)].size)
            self.exact_percentage = np.append(self.exact_percentage, 100*F0_exact.size/F0_ref.size)
        else:
            self.exact_percentage = np.append(self.exact_percentage, np.nan)
        self.outliers_percentage = np.append(self.outliers_percentage, 100*F0_error_outliers.size/F0_error.size)
        self.F0_error_median_all = np.append(self.F0_error_median_all, F0_error_median)
        self.F0_error_mean_all = np.append(self.F0_error_mean_all, F0_error_mean)
        self.F0_error_std_all = np.append(self.F0_error_std_all, F0_error_std)
        self.F0_error_relative_median_all = np.append(self.F0_error_relative_median_all, F0_error_relative_median)
        self.F0_error_relative_mean_all = np.append(self.F0_error_relative_mean_all, F0_error_relative_mean)
        self.F0_error_relative_std_all = np.append(self.F0_error_relative_std_all, F0_error_relative_std)
        self.train_accuracy_all = np.append(self.train_accuracy_all, train_accuracy)
        self.trained_SNR_dB_all = np.append(self.trained_SNR_dB_all, trained_SNR_dB)

        logging.info(f"Size{F0_error.size} Outliers:{F0_error_outliers.size} %%:{100*F0_error_outliers.size/F0_error.size}")
        logging.info(f"With outlayers:    mean:{np.mean(F0_error)} std:{np.std(F0_error)}")
        logging.info(f"Without outlayers: mean:{np.mean(F0_error_no_outliers)} std:{np.std(F0_error_no_outliers)}")

        # check/draw number of results and number of outlayers per class
        # check/draw mean and std per class
        self.classes_labels
        if F0_ref_labels is not None:
            if self.classes_labels is None:
                F0_ref_labels_unique = np.unique(np.array(F0_ref_labels))
            else:
                F0_ref_labels_unique = self.classes_labels
            no_of_data_per_class = np.array([0] * F0_ref_labels_unique.size)
            no_of_outliers_per_class = np.array([0] * F0_ref_labels_unique.size)
            est_mean_error_per_class = np.array([0.0] * F0_ref_labels_unique.size)
            est_std_error_per_class = np.array([0.0] * F0_ref_labels_unique.size)

            for idx, F0_val in enumerate(F0_ref_labels_unique):
                #mask = (infer_results["F0_est"] == F0_val)
                label_mask = ((F0_ref_labels == F0_val) & (F0_ref > 0))

                logging.info(f"F0_val:{F0_val}")
                F0_error_mask = F0_ref[label_mask]-F0_est[label_mask]
                F0_error_mask_outliers = F0_error_mask[abs(F0_error_mask)/F0_ref[label_mask] > outliers_percentage_threshold] 
                F0_error_mask_2 = F0_error_mask[abs(F0_error_mask)/F0_ref[label_mask] <= outliers_percentage_threshold] # exclude outliers

                no_of_data_per_class[idx]  = F0_error_mask.size
                if F0_error_mask.size > 0:
                    no_of_outliers_per_class[idx]  = F0_error_mask_outliers.size
                    est_mean_error_per_class[idx]  = np.mean(F0_error_mask_2)
                    est_std_error_per_class[idx]  = np.std(F0_error_mask_2)

                    logging.info(f"Size:{F0_error_mask.size}, Outliers:{F0_error_mask_outliers.size} %%:{100*F0_error_mask_outliers.size/F0_error_mask.size}")
                    logging.info(f"With outlayers:    mean:{np.mean(F0_error_mask)}, std:{np.std(F0_error_mask)}")
                    logging.info(f"Without outlayers: mean:{np.mean(F0_error_mask_2)}, std:{np.std(F0_error_mask_2)}")
                else:
                    no_of_outliers_per_class[idx]  = 0
                    est_mean_error_per_class[idx]  = np.nan
                    est_std_error_per_class[idx]  = np.nan

                    logging.info("===== no data =======")
                #input()
            
            self.no_of_data_per_class_all.insert(epoch_ind, no_of_data_per_class)
            self.no_of_outliers_per_class_all.insert(epoch_ind, no_of_outliers_per_class)
            self.est_mean_error_per_class_all.insert(epoch_ind, est_mean_error_per_class)
            self.est_std_error_per_class_all.insert(epoch_ind, est_std_error_per_class)

        if do_draw_epoch:
            if F0_ref_labels is not None:
                fig = plt.figure(epoch_str)
                axs = fig.subplots(3)
                axs_ind = 0
                axs[axs_ind].scatter(F0_ref_labels_unique, no_of_data_per_class, color='blue', s=1)
                axs[axs_ind].scatter(F0_ref_labels_unique, no_of_outliers_per_class, color='red', s=1)
                axs[axs_ind].set_title("no_of_data & no_of_outliers")

                textstr = 'no_of_data = ' + str(np.sum(no_of_data_per_class)) + '\n'
                textstr += 'no_of_outliers = ' + str(np.sum(no_of_outliers_per_class)) + '\n'
                textstr += 'no_of_outliers = ' + format(np.sum(no_of_outliers_per_class)/np.sum(no_of_data_per_class) * 100, '.2f') + "%%" + '\n'

                # these are matplotlib.patch.Patch properties
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                # place a text box in upper left in axes coords
                axs[axs_ind].text(0.05, 0.95, textstr[:-2], transform=axs[axs_ind].transAxes, fontsize=8,
                verticalalignment='top', bbox=props)

                axs_ind += 1
                axs[axs_ind].scatter(F0_ref_labels_unique, est_mean_error_per_class, color='blue', s=1)
                axs[axs_ind].plot(F0_ref_labels_unique, np.array([np.mean(F0_error_no_outliers)] * F0_ref_labels_unique.size), color='red')
                axs[axs_ind].set_title("est_mean_error")
                axs_ind += 1
                axs[axs_ind].scatter(F0_ref_labels_unique, est_std_error_per_class, color='blue', s=1)
                axs[axs_ind].plot(F0_ref_labels_unique, np.array([np.std(F0_error_no_outliers)] * F0_ref_labels_unique.size), color='red')
                axs[axs_ind].set_title("est_std_error")


                fig.show()
                fig.canvas.flush_events()

            # input("Press Enter")

            # TODO implement figures saving
            # self.plot_epoc_data(infer_results["F0_ref"], infer_results["F0_est"], epoch_str, "reference (blue) vs estimates (red)", save_path)
            self.plot_epoc_data(F0_ref, F0_est, epoch_str, "reference (blue) vs estimates (red)", None)


    def plot_overall_data(self, setup_idx):
        logging.info("Plotting overall data")

        no_of_epochs = self.exact_percentage.size

        figs = {}
        figs[1] = plt.figure(f"{setup_idx}: overall results")
        axs = figs[1].subplots(2)
        axs_ind = 0
        h_ex, = axs[axs_ind].plot(range(no_of_epochs), self.exact_percentage, 'k')
        h_ta, = axs[axs_ind].plot(range(no_of_epochs), self.train_accuracy_all, 'r')
        axs[axs_ind].legend([h_ex, h_ta], ["infer_exact_percentage", "train_accuracy_all"])
        axs_ind += 1
        h_ex, = axs[axs_ind].plot(range(1, no_of_epochs), np.diff(self.exact_percentage), 'k')
        h_ta, = axs[axs_ind].plot(range(1, no_of_epochs), np.diff(self.train_accuracy_all), 'r')
        axs[axs_ind].plot(range(1, no_of_epochs+1), np.zeros_like(self.train_accuracy_all), 'b', linewidth=0.5)
        axs[axs_ind].legend([h_ex, h_ta], ["infer_exact_percentage diff", "train_accuracy_all diff"])

        figs[1].show()

        figs[2] = plt.figure(f"{setup_idx}: overall results (diff)")
        axs = figs[2].subplots(3)
        axs_ind = 0
        h_op, = axs[axs_ind].plot(self.outliers_percentage, 'b')
        axs[axs_ind].set_title("outliers_percentage")
        axs_ind += 1
        axs[axs_ind].plot(self.F0_error_mean_all)
        axs[axs_ind].plot(np.zeros_like(self.F0_error_mean_all), 'b', linewidth=0.5)
        axs[axs_ind].set_title("est_mean_error_all (without outliers)")
        axs_ind += 1
        axs[axs_ind].plot(self.F0_error_std_all)
        axs[axs_ind].set_title("est_std_error_all (without outliers)")

        figs[2].show()
        figs[2].canvas.flush_events()

        return figs

    def plot_epoc_data(self, F0_ref, F0_est, epoch_str, regressorName, save_path):
        # data scatter plots & histograms
        fig = plt.figure(epoch_str + ": F0_ref and F0_est")
        axs = fig.subplots(3)
        axs[0].scatter(range(len(F0_ref)), F0_ref, color='blue', s=2)
        axs[0].scatter(range(len(F0_est)), F0_est, color='red', s=2)
        axs[0].set_title(regressorName)
        
        #dF0 = np.mean(np.array(self.classes_labels)[1:]-np.array(self.classes_labels[0:-1]))
        # hist, bin_edges = np.histogram(F0_ref, self.classes_labels, density=True)
        hist, bin_edges = np.histogram(F0_ref, self.F0_data_hist_edges, density=True)
        axs[1].bar(bin_edges[:-1], hist, width=self.classes_dF0, align='edge', edgecolor="blue", linewidth=0.7)
        
        #hist, bin_edges = np.histogram(F0_est, self.classes_labels, density=True)
        hist, bin_edges = np.histogram(F0_est, self.F0_data_hist_edges, density=True)
        axs[1].bar(bin_edges[:-1]+0.2*self.classes_dF0, hist, width=0.6*self.classes_dF0, align='edge', facecolor = 'tomato', edgecolor="red", linewidth=0.7)

        mask_F0_ref_not_0 = tuple([F0_ref > 0])
        hist, bin_edges = np.histogram(F0_ref[mask_F0_ref_not_0], self.F0_data_hist_edges, density=True)
        axs[2].bar(bin_edges[:-1], hist, width=self.classes_dF0, align='edge', edgecolor="blue", linewidth=0.7)
        hist, bin_edges = np.histogram(F0_est[mask_F0_ref_not_0], self.F0_data_hist_edges, density=True)
        axs[2].bar(bin_edges[:-1]+0.2*self.classes_dF0, hist, width=0.6*self.classes_dF0, align='edge', facecolor = 'tomato', edgecolor="red", linewidth=0.7)
        axs[2].set_title("excluded F0_ref == 0")

        fig.show()

        # F0_error scatter plot & histogram
        fig = plt.figure(epoch_str + ": F0_error")
        axs = fig.subplots(3)

        F0_ref_not_0 = F0_ref[mask_F0_ref_not_0]
        F0_est_not_0 = F0_est[mask_F0_ref_not_0]

        F0_error = F0_est[mask_F0_ref_not_0] - F0_ref[mask_F0_ref_not_0]
        F0_index = np.array(range(len(F0_ref)))
        F0_index = F0_index[mask_F0_ref_not_0]

        outliers_percentage_threshold = 0.2 # in percents
        no_outliers_mask = abs(F0_error) / F0_ref <= outliers_percentage_threshold

        F0_error_no_outliers = F0_error[no_outliers_mask] # exclude outliers
        F0_ref_no_outliers = F0_ref_not_0[no_outliers_mask] # exclude outliers
        F0_est_no_outliers = F0_est_not_0[no_outliers_mask] # exclude outliers
        F0_index_no_outliers = F0_index[no_outliers_mask] # exclude outliers

        axs[0].scatter(F0_ref, F0_est, color='blue', s=1)
        axs[0].scatter(F0_ref_no_outliers, F0_est_no_outliers, color='red', s=1)
        axs[0].set_title('F0_est (red) vs F_ref (blue)')

        axs[1].scatter(F0_index, F0_error, color='blue', s=1)
        axs[1].scatter(F0_index_no_outliers, F0_error_no_outliers, color='red', s=1)
        axs[1].set_title('error: with outliers (blue) & without outliers (red)')

        # print("Unique y_err: ", np.unique(np.array(y_err)))

        # hist, bin_edges = np.histogram(y_err, bins=100, density=True)
        
        # Plot estimation error  histogram (including outliers)
        hist, bin_edges = np.histogram(F0_error, self.F0_error_hist_edges, density=True)
        axs[2].bar(bin_edges[:-1], hist, width=self.classes_dF0, align='edge', edgecolor="white", linewidth=0.7)
        #hist_min = np.min(hist); hist_max = np.max(hist)
        #axs[2].plot(self.est_mean_error_all[[-1, -1]], [hist_min, hist_max], color='r', linestyle='--')
        #axs[2].autoscale(enable=False, axis='y')
        axs[2].set_title('histogram (with outliers)')

        lower_lim = self.F0_error_mean_all[-1] - 3 * self.F0_error_std_all[-1]
        upper_lim = self.F0_error_mean_all[-1] + 3 * self.F0_error_std_all[-1]
        hist_max = np.max(hist)

        axs[2].set_ylim(axs[2].get_ylim())

        axs[2].plot(self.F0_error_mean_all[[-1, -1]], axs[2].get_ylim(), color='r', linestyle='--')
        textstr = 'Without outliers:\n' 
        textstr += '  mean = ' + format(self.F0_error_mean_all[-1], '.3f') + '\n'
        # # https://matplotlib.org/stable/tutorials/text/annotations.html
        # axs[2].annotate('error mean = ' + str(self.F0_error_mean_all[-1]),
        #                  xy=(self.F0_error_mean_all[-1], 0.95*hist_max), xycoords='data', 
        #                 xytext=(upper_lim, 0.9*hist_max), textcoords='data', 
        #                 arrowprops=dict(facecolor='black', linewidth=1, 
        #                                 arrowstyle="->",mutation_scale=10), #frac=0.1),
        #                 horizontalalignment='left', verticalalignment='top')

        axs[2].plot(self.F0_error_median_all[[-1, -1]], axs[2].get_ylim(), color='m', linestyle=':')
        textstr += '  median = ' + format(self.F0_error_median_all[-1], '.3f') + '\n'
        # axs[1].annotate('error median = ' + str(self.F0_error_median_all[-1]),
        #                  xy=(self.F0_error_median_all[-1], 0.85*hist_max), xycoords='data', 
        #                 xytext=(upper_lim, 0.8*hist_max), textcoords='data', 
        #                 arrowprops=dict(facecolor='black', linewidth=1, 
        #                                 arrowstyle="->",mutation_scale=10), #frac=0.1),
        #                 horizontalalignment='left', verticalalignment='top')

        axs[2].plot([lower_lim, lower_lim], axs[2].get_ylim(), color='k', linestyle='--')
        axs[2].plot([upper_lim, upper_lim], axs[2].get_ylim(), color='k', linestyle='--')
        textstr += '  std = ' + format(self.F0_error_std_all[-1], '.3f') + '\n'
        # axs[2].annotate('error std = ' + str(self.F0_error_std_all[-1]),
        #                  xy=(upper_lim, 0.7*hist_max), xycoords='data', 
        #                 xytext=(2.0*upper_lim, 0.65*hist_max), textcoords='data', 
        #                 arrowprops=dict(facecolor='black', linewidth=1, 
        #                                 arrowstyle="->",mutation_scale=10), #frac=0.1),
        #                 horizontalalignment='left', verticalalignment='top')

        #ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
        #    ylim=(0, 8), yticks=np.arange(1, 8))

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        axs[2].text(0.05, 0.95, textstr[:-2], transform=axs[2].transAxes, fontsize=8,
        verticalalignment='top', bbox=props)
    
        fig.show()
    

        # relatice F0_error scatter plot & histogram
        fig = plt.figure(epoch_str + ": F0_error_relative")
        axs = fig.subplots(2)

        F0_error_relative = F0_error / F0_ref[mask_F0_ref_not_0]
        F0_error_relative_no_outliers = F0_error_relative[no_outliers_mask]

        axs[0].scatter(F0_ref[mask_F0_ref_not_0], F0_error_relative, color='blue', s=1)
        axs[0].scatter(F0_ref_no_outliers, F0_error_relative_no_outliers, color='red', s=1)
        axs[0].set_title('relative error vs F0_ref')


        # Plot relative estimation error  histogram
        # TODO check if different edges are needed for relative error
        hist_edges_scalling_factor = 1/500
        hist, bin_edges = np.histogram(F0_error_relative, np.array(self.F0_error_hist_edges)*hist_edges_scalling_factor, density=True)
        axs[1].bar(bin_edges[:-1], hist, width=self.classes_dF0*hist_edges_scalling_factor, align='edge', edgecolor="white", linewidth=0.7)

        lower_lim = self.F0_error_relative_mean_all[-1] - 3 * self.F0_error_relative_std_all[-1]
        upper_lim = self.F0_error_relative_mean_all[-1] + 3 * self.F0_error_relative_std_all[-1]
        hist_max = np.max(hist)

        axs[1].set_ylim(axs[1].get_ylim())

        axs[1].plot(self.F0_error_relative_mean_all[[-1, -1]], axs[1].get_ylim(), color='r', linestyle='--')
        textstr = 'mean = ' + format(self.F0_error_relative_mean_all[-1], '.6f') + '\n'

        axs[1].plot(self.F0_error_relative_median_all[[-1, -1]], axs[1].get_ylim(), color='m', linestyle=':')
        textstr += 'median = ' + format(self.F0_error_relative_median_all[-1], '.6f') + '\n'

        axs[1].plot([lower_lim, lower_lim], axs[1].get_ylim(), color='k', linestyle='--')
        axs[1].plot([upper_lim, upper_lim], axs[1].get_ylim(), color='k', linestyle='--')
        textstr += 'std = ' + format(self.F0_error_relative_std_all[-1], '.6f') + '\n'

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        axs[1].text(0.05, 0.95, textstr[:-1], transform=axs[1].transAxes, fontsize=8,
                    verticalalignment='top', bbox=props)


        if save_path:
            fig.savefig(PurePath(save_path / "label_vs_prediction.png"), dpi=200)
        #plt.show(block=False)
        fig.show()
        fig.canvas.flush_events()
        return


    def process_accuracy(self, epoch_index, model_accuracy_table, accu_skip_step = 50, model_batch_size = 128):
        # accu_skip_step = 50 # 8
        model_accuracy_table = np.array(model_accuracy_table)

        if epoch_index == 0:
            # reset 
            self.d_correct_table = np.zeros(shape=(0,))
            self.d_total_table = np.zeros(shape=(0,))
            self.model_accuracy_table_all = np.zeros(shape=(0,))
            self.epochs_accuracy = np.zeros(shape=(0,))
            self.correct_table_offset = 0
            self.averaged_model_accuracy_table_size = None
        else:
            if not hasattr(self, "d_correct_table"):
                self.d_correct_table = np.zeros(shape=(0,))
            if not hasattr(self, "d_total_table"):
                self.d_total_table = np.zeros(shape=(0,))
            if not hasattr(self, "model_accuracy_table_all"):
                self.model_accuracy_table_all = np.zeros(shape=(0,))
            if not hasattr(self, "epochs_accuracy"):
                self.epochs_accuracy = np.zeros(shape=(0,))
            if not hasattr(self, "correct_table_offset"):
                self.correct_table_offset = 0
            if not hasattr(self, "averaged_model_accuracy_table_size"):
                self.averaged_model_accuracy_table_size = None


        # model_accuracy_table_size = (model_accuracy_table_size + (accu_skip_step-1)) / accu_skip_step
        # model_accuracy_table_size = int(model_accuracy_table_size)

        accu_model_batch_size = model_batch_size * accu_skip_step

        averaged_model_accuracy_table = 0
        if self.averaged_model_accuracy_table_size is None:
            self.averaged_model_accuracy_table_size = int((len(model_accuracy_table)+1)/accu_skip_step)
        for offset in range(0,accu_skip_step):
            averaged_model_accuracy_table += (model_accuracy_table[offset:self.averaged_model_accuracy_table_size*accu_skip_step:accu_skip_step]/accu_skip_step)

        model_accuracy_table = averaged_model_accuracy_table
        model_accuracy_table_size = len(model_accuracy_table)

        self.d_correct_table = np.append(self.d_correct_table, np.zeros(shape=(model_accuracy_table_size, 1)))
        self.d_total_table = np.append(self.d_total_table, np.zeros(shape=(model_accuracy_table_size, 1)))
        self.model_accuracy_table_all = np.append(self.model_accuracy_table_all, model_accuracy_table)

        # for i in range(0,model_accuracy_table_size):
        #     accuracy = model_accuracy_table[i]
        #     d_correct = (accuracy - last_accuracy) * last_total + accuracy * model_batch_size
        #     # correct += d_correct

        #     last_accuracy = accuracy
        #     last_total += model_batch_size

        #     d_correct_table[correct_table_offset + i] = d_correct
        #     d_total_table[correct_table_offset + i] = model_batch_size
        # correct_table_offset += model_accuracy_table_size
        accuracy = np.array(model_accuracy_table)
        d_correct = np.zeros_like(accuracy)
        d_correct[0] = accuracy[0] * accu_model_batch_size
        cumsum_total = np.arange(1, len(accuracy)+1) * accu_model_batch_size
        d_correct[1:] = (accuracy[1:] - accuracy[0:-1]) * cumsum_total[0:-1] + accuracy[1:] * accu_model_batch_size

        self.d_correct_table[self.correct_table_offset:] = d_correct
        self.d_total_table[self.correct_table_offset:] = accu_model_batch_size
        self.correct_table_offset += model_accuracy_table_size

        self.epochs_accuracy = np.append(self.epochs_accuracy, np.sum(d_correct) / cumsum_total[-1])

        return {"d_correct_table": self.d_correct_table, "d_total_table": self.d_total_table, 
                "model_accuracy_table_all": self.model_accuracy_table_all, "averaged_model_accuracy_table_size": self.averaged_model_accuracy_table_size,
                "epochs_accuracy": self.epochs_accuracy}

    def get_processed_accuracy(self):
        return {"d_correct_table": self.d_correct_table, "d_total_table": self.d_total_table, 
                "model_accuracy_table_all": self.model_accuracy_table_all, "averaged_model_accuracy_table_size": self.averaged_model_accuracy_table_size,
                "epochs_accuracy": self.epochs_accuracy}
