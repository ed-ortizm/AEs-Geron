import matplotlib
import matplotlib.pyplot as plt
import numpy as np
################################################################################
class Outlier:
    """
    Class for dealing with the outliers based on a generative model trained with
    tensorflow.keras
    """
    ############################################################################
    def __init__(self, metric:'str'='mse', p:'float'=0.25):
        """
        Init fucntion

        Args:

            metric: (str) the name of the metric used to compute the outlier score
                using the observed spectrum and its reconstruction. Possible

            p: (float > 0) in case the metric is the lp metric, p needs to be a
                non null possitive float [Aggarwal 2001]
        """

        self.metric = metric

        if metric=='lp':
            self.p = p
        else:
            self.p = False
    ############################################################################
    def score(self, O, R, percentages):
        """
        Computes the outlier score according to the metric used to instantiate
        the class.

        Args:
            O: (2D np.array) with the original objects where index 0 indicates
            the object and index 1 the features of the object.

            R: (2D np.array) with the reconstructed objects where index 0
            indicates the object and index 1 the features of the object.

        Returns:
            A one dimensional numpy array with the outlier scores for objects
            present in O
        """

        # check if I can use a dict or anything to avoid to much typing

        if self.metric == 'mse':
            print(f'Computing the outlier scores')
            return self._mse(O=O, R=R, percentages=percentages)

        elif self.metric == 'chi2':
            return self._chi2(O=O, R=R)

        elif self.metric == 'mad':
            return self._mad(O=O, R=R)

        elif self.metric == 'lp':

            if self.p == 'p' or self.p <= 0:
                print(f'For the {self.metric} metric you need p')
                return None

            return self._lp(O=O, R=R)

        else:
            print(f'The provided metric: {self.metric} is not implemented yet')
            sys.exit()
    ############################################################################
# Mahalanobis, Canberra, Braycurtis, and KL-divergence
    ############################################################################
    def _mse(self, O, R, percentages):
        """
        Computes the mean square error for the reconstruction of the input
        objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the mean square error for objects
            present in O
        """

        outlier_scores = []
        for percentage in percentages:

            # mse = np.square(R - O)
            mse = np.abs(R - O)/np.abs(O)

            number_outlier_fluxes = int(0.01*percentage*mse.shape[1])

            highest_mse = np.argpartition(
                mse, -1 * number_outlier_fluxes,
                axis=1)[:, -1 * number_outlier_fluxes:]

            score = np.empty(highest_mse.shape)

            for n, idx in enumerate(highest_mse):

                score[n, :] = mse[n, idx]

            outlier_scores.append(score.sum(axis=1))

        return outlier_scores
    ############################################################################
    def _chi2(self, O, R):
        """
        Computes the chi square error for the reconstruction of the input
        objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the chi square error for objects
            present in O
        """

        return (np.square(R - O) * (1 / np.abs(O))).sum(axis=1)
    ############################################################################
    def _mad(self, O, R):
        """
        Computes the maximum absolute deviation from the reconstruction of the
        input objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the maximum absolute deviation
            from the objects present in O
        """

        return np.abs(R - O)*(1 / np.abs(O)).sum(axis=1)
    ############################################################################
    def _lp(self, O, R):
        """
        Computes the lp distance from the reconstruction of the input objects

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the lp distance from the objects
            present in O
        """

        return (np.sum((np.abs(R - O))**self.p, axis=1))**(1 / self.p)
# gotta code conditionals to make sure that the user inputs a "good one"
    ############################################################################
    def user_metric(self, custom_metric, O, R):
        """
        Computes the custom metric for the reconstruction of the input objects
        as defined by the user

        Args:
            O: (2D np.array) with the original objects where index 0 denotes
                indicates the objec and index 1 the features of the object.

            R: Reconstruction of O by (tensorflow.keras model) the generative
            model

        Returns:
            A one dimensional numpy array with the score produced by the user
            defiend metric of objects present in O
        """

        return self.custom_metric(O, R)
    ############################################################################
    def top_reconstructions(self, scores, n_top_spectra):
        """
        Selects the most normal and outlying objecs

        Args:
            scores: (1D np.array) outlier scores

            n_top_spectra: (int > 0) this parameter controls the number of
                objects identifiers to return for the top reconstruction,
                that is, the idices for the most oulying and the most normal
                objects.

        Returns:
            most_normal, most_oulying: (1D np.array, 1D np.array) numpy arrays
                with the location indexes of the most normal and outlying
                object in the training (and pred) set.
        """

        spec_idxs = np.argpartition(scores,
            [n_top_spectra, -1 * n_top_spectra])

        most_normal_ids = spec_idxs[: n_top_spectra]
        most_oulying_ids = spec_idxs[-1 * n_top_spectra:]

        return most_normal_ids, most_oulying_ids
################################################################################
# class Outlier:
#     """
#     Class for dealing with the outliers based on a generative model trained with
#     tensorflow.keras
#     """
#     ############################################################################
#     def __init__(self, metric:'str'='mse', p:'float'=0.25,
#         custom:'bool'=False, custom_metric:'function'=None):
#         """
#         Init fucntion
#
#         Args:
#
#             metric: (str) the name of the metric used to compute the outlier score
#                 using the observed spectrum and its reconstruction. Possible
#
#             p: (float > 0) in case the metric is the lp metric, p needs to be a
#                 non null possitive float [Aggarwal 2001]
#         """
#
#         self.metric = metric
#
#         # if metric=='lp' and p>0:
#         #     self.p = p
#         # if:
#         #     print(f'For lp metric p must be positive, instead p={p}')
#         #     print(f'Failed to instantiate class')
#         #     sys.exit()
#
#         self.custom = custom
#         if self.custom:
#             self.custom_metric = custom_metric
#     ############################################################################
#     def score(self, O, R, percentages):
#         """
#         Computes the outlier score according to the metric used to instantiate
#         the class.
#
#         Args:
#             O: (2D np.array) with the original objects where index 0 indicates
#             the object and index 1 the features of the object.
#
#             R: (2D np.array) with the reconstructed objects where index 0
#             indicates the object and index 1 the features of the object.
#
#         Returns:
#             A one dimensional numpy array with the outlier scores for objects
#             present in O
#         """
#
#         # check if I can use a dict or anything to avoid to much typing
#         if self.custom:
#             print(f'Computing the predictions of {model_name}')
#             return self.user_metric(O=O, R=R)
#
#         elif self.metric == 'mse':
#             print(f'Computing the outlier scores')
#             return self._mse(O=O, R=R, percentages=percentages)
#
#         elif self.metric == 'chi2':
#             return self._chi2(O=O, R=R)
#
#         elif self.metric == 'mad':
#             return self._mad(O=O, R=R)
#
#         elif self.metric == 'lp':
#
#             if self.p == 'p' or self.p <= 0:
#                 print(f'For the {self.metric} metric you need p')
#                 return None
#
#             return self._lp(O=O, R=R)
#
#         else:
#             print(f'The provided metric: {self.metric} is not implemented yet')
#             sys.exit()
#     ############################################################################
#     def _coscine_similarity(self, O, R):
#         """
#         Computes the coscine similarity between the reconstruction of the input
#         objects
#
#         Args:
#             O: (2D np.array) with the original objects where index 0 denotes
#                 indicates the objec and index 1 the features of the object.
#
#             R: Reconstruction of O by (tensorflow.keras model) the generative
#             model
#
#         Returns:
#             A one dimensional numpy array with the cosine similarity between
#             objects O and their reconstructiob
#         """
#         pass
#     ############################################################################
#     def _jaccard_index(self, O, R):
#         """
#         Computes the mean square error for the reconstruction of the input
#         objects
#
#         Args:
#             O: (2D np.array) with the original objects where index 0 denotes
#                 indicates the objec and index 1 the features of the object.
#
#             R: Reconstruction of O by (tensorflow.keras model) the generative
#             model
#
#         Returns:
#             A one dimensional numpy array with the mean square error for objects
#             present in O
#         """
#         pass
#     ############################################################################
#     def _sorensen_dice_index(self, O, R):
#         """
#         Computes the mean square error for the reconstruction of the input
#         objects
#
#         Args:
#             O: (2D np.array) with the original objects where index 0 denotes
#                 indicates the objec and index 1 the features of the object.
#
#             R: Reconstruction of O by (tensorflow.keras model) the generative
#             model
#
#         Returns:
#             A one dimensional numpy array with the mean square error for objects
#             present in O
#         """
#         pass
# # Mahalanobis, Canberra, Braycurtis, and KL-divergence
#     ############################################################################
#     def _mse(self, O, R, percentages):
#         """
#         Computes the mean square error for the reconstruction of the input
#         objects
#
#         Args:
#             O: (2D np.array) with the original objects where index 0 denotes
#                 indicates the objec and index 1 the features of the object.
#
#             R: Reconstruction of O by (tensorflow.keras model) the generative
#             model
#
#         Returns:
#             A one dimensional numpy array with the mean square error for objects
#             present in O
#         """
#
#         outlier_scores = []
#         for percentage in percentages:
#
#             mse = np.square(R - O)
#
#             number_outlier_fluxes = int(percentage*mse.shape[1])
#
#             highest_mse = np.argpartition(
#                 mse, -1 * number_outlier_fluxes,
#                 axis=1)[:, -1 * number_outlier_fluxes:]
#
#             score = np.empty(highest_mse.shape)
#
#             for n, idx in enumerate(highest_mse):
#
#                 score[n, :] = mse[n, idx]
#
#             outlier_scores.append(score.sum(axis=1))
#
#         return outlier_scores
#     ############################################################################
#     def _chi2(self, O, R):
#         """
#         Computes the chi square error for the reconstruction of the input
#         objects
#
#         Args:
#             O: (2D np.array) with the original objects where index 0 denotes
#                 indicates the objec and index 1 the features of the object.
#
#             R: Reconstruction of O by (tensorflow.keras model) the generative
#             model
#
#         Returns:
#             A one dimensional numpy array with the chi square error for objects
#             present in O
#         """
#
#         return (np.square(R - O) * (1 / np.abs(R))).sum(axis=1)
#     ############################################################################
#     def _mad(self, O, R):
#         """
#         Computes the maximum absolute deviation from the reconstruction of the
#         input objects
#
#         Args:
#             O: (2D np.array) with the original objects where index 0 denotes
#                 indicates the objec and index 1 the features of the object.
#
#             R: Reconstruction of O by (tensorflow.keras model) the generative
#             model
#
#         Returns:
#             A one dimensional numpy array with the maximum absolute deviation
#             from the objects present in O
#         """
#
#         return np.abs(R - O).mean(axis=1)
#     ############################################################################
#     def _lp(self, O, R):
#         """
#         Computes the lp distance from the reconstruction of the input objects
#
#         Args:
#             O: (2D np.array) with the original objects where index 0 denotes
#                 indicates the objec and index 1 the features of the object.
#
#             R: Reconstruction of O by (tensorflow.keras model) the generative
#             model
#
#         Returns:
#             A one dimensional numpy array with the lp distance from the objects
#             present in O
#         """
#
#         return (np.sum((np.abs(R - O))**self.p, axis=1))**(1 / self.p)
# # gotta code conditionals to make sure that the user inputs a "good one"
#     ############################################################################
#     def user_metric(self, custom_metric, O, R):
#         """
#         Computes the custom metric for the reconstruction of the input objects
#         as defined by the user
#
#         Args:
#             O: (2D np.array) with the original objects where index 0 denotes
#                 indicates the objec and index 1 the features of the object.
#
#             R: Reconstruction of O by (tensorflow.keras model) the generative
#             model
#
#         Returns:
#             A one dimensional numpy array with the score produced by the user
#             defiend metric of objects present in O
#         """
#
#         return self.custom_metric(O, R)
#     ############################################################################
#     def top_reconstructions(self, scores, n_top_spectra):
#         """
#         Selects the most normal and outlying objecs
#
#         Args:
#             scores: (1D np.array) outlier scores
#
#             n_top_spectra: (int > 0) this parameter controls the number of
#                 objects identifiers to return for the top reconstruction,
#                 that is, the idices for the most oulying and the most normal
#                 objects.
#
#         Returns:
#             most_normal, most_oulying: (1D np.array, 1D np.array) numpy arrays
#                 with the location indexes of the most normal and outlying
#                 object in the training (and pred) set.
#         """
#
#         spec_idxs = np.argpartition(scores,
#             [n_top_spectra, -1 * n_top_spectra])
#
#         most_normal_ids = spec_idxs[: n_top_spectra]
#         most_oulying_ids = spec_idxs[-1 * n_top_spectra:]
#
#         return most_normal_ids, most_oulying_ids
# ################################################################################
