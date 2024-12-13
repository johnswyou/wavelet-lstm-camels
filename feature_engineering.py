import pandas as pd
import numpy as np
from utils import load_pickle


class MODWTFeatureEngineer:
    """
    A feature engineering class that computes the Maximal Overlap Discrete Wavelet Transform (MODWT)
    for specified features in a pandas DataFrame and appends the resulting coefficients as new features.

    Parameters:
    -----------
    wavelet : str, optional (default='db1')
        The name of the wavelet to use. Must be one of the wavelets available in PyWavelets.
    
    v_levels : int or list of int, optional (default=[1])
        The decomposition level(s) for the scaling (Vj) coefficients.
    
    w_levels : int or list of int, optional (default=[1])
        The decomposition level(s) for the wavelet (Wj) coefficients.
    
    Attributes:
    -----------
    wavelet : pywt.Wavelet
        The PyWavelet wavelet object.
    
    v_levels : list of int
        The decomposition levels for scaling coefficients.
    
    w_levels : list of int
        The decomposition levels for wavelet coefficients.
    
    max_level : int
        The maximum level among `v_levels` and `w_levels`.
    
    L : int
        The length of the wavelet filter.
    
    L_J_v : dict
        The number of boundary coefficients to remove for each scaling level.
    
    L_J_w : dict
        The number of boundary coefficients to remove for each wavelet level.
    
    scaling_filter : numpy.ndarray
        The scaling (low-pass) filter coefficients.
    
    wavelet_filter : numpy.ndarray
        The wavelet (high-pass) filter coefficients.
    """

    def __init__(self, wavelet='db1', v_levels=1, w_levels=1):
        # Initialize wavelet
        self.scaling_dict = load_pickle("filters/scaling_dict.pkl")
        self.wavelet_dict = load_pickle("filters/wavelet_dict.pkl")

        self.scaling_filter = self.scaling_dict[wavelet]
        self.wavelet_filter = self.wavelet_dict[wavelet]
        
        # Convert v_levels and w_levels to lists if they are integers
        if isinstance(v_levels, int):
            self.v_levels = [v_levels]
        elif isinstance(v_levels, list):
            self.v_levels = v_levels
        else:
            raise TypeError("v_levels must be an integer or a list of integers.")
        
        if isinstance(w_levels, int):
            self.w_levels = [w_levels]
        elif isinstance(w_levels, list):
            self.w_levels = w_levels
        else:
            raise TypeError("w_levels must be an integer or a list of integers.")
        
        # Validate that levels are positive integers
        for lvl in self.v_levels + self.w_levels:
            if not isinstance(lvl, int) or lvl < 1:
                raise ValueError("All levels in v_levels and w_levels must be positive integers.")
        
        # Determine the maximum level needed for decomposition
        self.max_level = max(self.v_levels + self.w_levels)
        
        # Length of the decomposition low-pass filter
        # self.L = len(self.wavelet.rec_lo)
        self.L = len(self.scaling_filter)
        
        # Calculate L_J for scaling and wavelet coefficients
        self.L_J_v = {J: (2**J - 1) * (self.L - 1) + 1 for J in self.v_levels}
        self.L_J_w = {J: (2**J - 1) * (self.L - 1) + 1 for J in self.w_levels}

    def modwt(self, signal):
        """
        Perform MODWT on a single 1D signal up to the specified maximum level using the logic
        from `modwt_level_j`. The method signature and name remain the same, but we now rely on
        the `equivalent_filter` and `modwt_level_j` approach for computing coefficients.
        
        Parameters
        ----------
        signal : numpy.ndarray
            The input signal (1D array) to transform.

        Returns
        -------
        W : dict
            Wavelet coefficients for each level j.
        
        V : dict
            Scaling coefficients for each level j.
        """
        def insert_zeros_between_elements(filter_coeffs, num_zeros):
            """
            Insert `num_zeros` zeros between elements of `filter_coeffs`.
            """
            if num_zeros == 0:
                return np.array(filter_coeffs, dtype=float)

            L = len(filter_coeffs)
            new_length = L + (L - 1)*num_zeros
            new_filter = np.zeros(new_length, dtype=float)

            new_filter[0::num_zeros+1] = filter_coeffs
            return new_filter

        def equivalent_filter(g, h, j):
            """
            Construct the equivalent filters g_tilde_j, h_tilde_j for level j.
            """
            g = np.array(g, dtype=float)
            h = np.array(h, dtype=float)

            # eq_filter starts as g for k=1
            eq_filter = g.copy()

            # For k=2,...,j-1 convolve eq_filter with g_k
            for k in range(2, j):
                num_zeros = 2**(k-2)
                g_k = insert_zeros_between_elements(g, num_zeros)
                eq_filter = np.convolve(eq_filter, g_k)

            # For level j use h with 2^(j-1)-1 zeros
            num_zeros = 2**(j-1) - 1
            g_j = insert_zeros_between_elements(g, num_zeros)
            h_j = insert_zeros_between_elements(h, num_zeros)
            g_tilde_j = np.convolve(eq_filter, g_j)
            h_tilde_j = np.convolve(eq_filter, h_j)

            return g_tilde_j, h_tilde_j

        def modwt_level_j(X, g, h, j, equivalent_filter):
            """
            Compute MODWT coefficients at level j using causal boundary handling.
            """
            N = len(X)
            g_tilde_j, h_tilde_j = equivalent_filter(g, h, j)
            L_j = len(h_tilde_j)

            # Normalize the filters by 2^(j/2) for MODWT
            g_tilde_j = g_tilde_j / (2**(j/2))
            h_tilde_j = h_tilde_j / (2**(j/2))

            W_j = np.zeros(N)
            V_j = np.zeros(N)

            # Causal boundary handling: for t < L_j - 1, coefficients are NaN
            for t in range(N):
                if t < L_j - 1:
                    W_j[t] = np.nan
                    V_j[t] = np.nan
                else:
                    w_sum = 0.0
                    v_sum = 0.0
                    for l in range(L_j):
                        idx = t - l
                        w_sum += h_tilde_j[l] * X[idx]
                        v_sum += g_tilde_j[l] * X[idx]
                    W_j[t] = w_sum
                    V_j[t] = v_sum

            return W_j, V_j

        # Now we use modwt_level_j for each level j
        W = {}
        V = {}

        # Initial signal is the original
        X = signal.copy()

        for j in range(1, self.max_level + 1):
            W_j, V_j = modwt_level_j(X, self.scaling_filter, self.wavelet_filter, j, equivalent_filter)
            W[j] = W_j
            V[j] = V_j

        return W, V

    def transform(self, df, feature_columns):
        """
        Apply MODWT to specified feature columns and append the coefficients to the DataFrame.

        Parameters:
        -----------
        df : pandas.DataFrame
            The input DataFrame containing the features.
        
        feature_columns : list of str
            List of column names on which to perform MODWT.

        Returns:
        --------
        pandas.DataFrame
            The DataFrame with appended MODWT features.
        """
        df = df.copy()
        
        for col in feature_columns:
            # Extract the signal
            signal = df[col].values
            
            # Perform MODWT
            W, V = self.modwt(signal)
            
            # Process wavelet coefficients
            for J in self.w_levels:
                Wj = W[J]
                # L_J = self.L_J_w[J]
                
                # # Remove the first L_J boundary coefficients
                # Wj_trimmed = Wj[L_J:]
                
                # # To maintain the original DataFrame length, pad the removed coefficients with NaN
                # if len(Wj_trimmed) < len(Wj):
                #     Wj_padded = np.concatenate((np.full(L_J, np.nan), Wj_trimmed))
                # else:
                #     Wj_padded = Wj_trimmed  # In case L_J is 0
                
                # # Ensure the padded array has the same length as the original signal
                # if len(Wj_padded) < len(signal):
                #     padding_length = len(signal) - len(Wj_padded)
                #     Wj_padded = np.concatenate((np.full(padding_length, np.nan), Wj_padded))
                # elif len(Wj_padded) > len(signal):
                #     Wj_padded = Wj_padded[:len(signal)]

                assert len(Wj) == len(signal)
                
                # Append the coefficients as a new column
                # df[f'{col}_W{J}'] = Wj_padded
                df[f'{col}_W{J}'] = Wj
            
            # Process scaling coefficients
            for J in self.v_levels:
                Vj = V[J]
                # L_J = self.L_J_v[J]
                
                # # Remove the first L_J boundary coefficients
                # Vj_trimmed = Vj[L_J:]
                
                # # To maintain the original DataFrame length, pad the removed coefficients with NaN
                # if len(Vj_trimmed) < len(Vj):
                #     Vj_padded = np.concatenate((np.full(L_J, np.nan), Vj_trimmed))
                # else:
                #     Vj_padded = Vj_trimmed  # In case L_J is 0
                
                # # Ensure the padded array has the same length as the original signal
                # if len(Vj_padded) < len(signal):
                #     padding_length = len(signal) - len(Vj_padded)
                #     Vj_padded = np.concatenate((np.full(padding_length, np.nan), Vj_padded))
                # elif len(Vj_padded) > len(signal):
                #     Vj_padded = Vj_padded[:len(signal)]

                assert len(Vj) == len(signal)
                
                # Append the coefficients as a new column
                # df[f'{col}_V{J}'] = Vj_padded
                df[f'{col}_V{J}'] = Vj
        
        return df

    # @staticmethod
    def available_wavelets(self):
        """
        Retrieve a list of available wavelet names from PyWavelets.

        Returns:
        --------
        list of str
            Available wavelet names.
        """
        # return pywt.wavelist()
        return list(self.scaling_dict.keys())


# def insert_zeros_between_elements(filter_coeffs, num_zeros):
#     """
#     Given a list/array of coefficients and a number of zeros to insert between each element,
#     return a new array with the zeros inserted.
#     """
#     # If no zeros to insert, just return the original.
#     if num_zeros == 0:
#         return np.array(filter_coeffs, dtype=float)

#     # Length of the original filter
#     L = len(filter_coeffs)
#     # The new filter length after zero insertion
#     new_length = L + (L - 1)*num_zeros
#     new_filter = np.zeros(new_length, dtype=float)

#     # Place original elements in the correct positions
#     new_filter[0::num_zeros+1] = filter_coeffs
#     return new_filter


# def equivalent_filter(g, h, j):
#     """
#     Given scaling filter g, wavelet filter h (both length L), and level j,
#     construct the equivalent filter according to the specified procedure.
#     """
#     g = np.array(g, dtype=float)
#     h = np.array(h, dtype=float)

#     # Start with the first filter (k=1) which is just g as is (no zeros inserted)
#     eq_filter = g.copy()

#     # For levels 2 through j-1, convolve with g, inserting increasing zeros
#     for k in range(2, j):
#         num_zeros = 2**(k-2)  # zeros to insert between elements for filter k
#         g_k = insert_zeros_between_elements(g, num_zeros)
#         eq_filter = np.convolve(eq_filter, g_k)

#     # For level j, convolve with h, inserting 2^(j-1)-1 zeros between elements
#     num_zeros = 2**(j-1) - 1
#     g_j = insert_zeros_between_elements(g, num_zeros)
#     h_j = insert_zeros_between_elements(h, num_zeros)
#     g_tilde_j = np.convolve(eq_filter, g_j)
#     h_tilde_j = np.convolve(eq_filter, h_j)

#     return g_tilde_j, h_tilde_j


# def modwt_level_j(X, g, h, j, equivalent_filter):
#     """
#     Compute the MODWT wavelet (W_tilde_j) and scaling (V_tilde_j) coefficients 
#     for a given decomposition level j.

#     Parameters
#     ----------
#     X : numpy.ndarray
#         1D array containing the time series data of length N.
#     g : numpy.ndarray
#         1D array containing the scaling filter (father wavelet) at the base level.
#     h : numpy.ndarray
#         1D array containing the wavelet filter (mother wavelet) at the base level.
#     j : int
#         The decomposition level for which to compute the coefficients.
#     equivalent_filter : function
#         A function with signature equivalent_filter(g, h, j) that returns 
#         (g_tilde_j, h_tilde_j) for the j-th level:
        
#         g_tilde_j: numpy.ndarray
#             The j-th level equivalent scaling filter.
#         h_tilde_j: numpy.ndarray
#             The j-th level equivalent wavelet filter.

#     Returns
#     -------
#     W_j : numpy.ndarray
#         Wavelet coefficients array at level j, length N.
#     V_j : numpy.ndarray
#         Scaling coefficients array at level j, length N.
#     """

#     N = len(X)
#     # Get equivalent filters at level j
#     g_tilde_j, h_tilde_j = equivalent_filter(g, h, j)

#     L_j = len(h_tilde_j)  # should be equal to len(g_tilde_j)

#     # Initialize output arrays
#     W_j = np.zeros(N)
#     V_j = np.zeros(N)

#     # The following implements circular shifting

#     # Compute MODWT coefficients
#     # Note: indices are taken modulo N to implement circular shifting
#     # for t in range(N):
#     #     # Compute W_j[t]
#     #     w_sum = 0.0
#     #     v_sum = 0.0
#     #     for l in range(L_j):
#     #         idx = (t - l) % N
#     #         w_sum += h_tilde_j[l] * X[idx]
#     #         v_sum += g_tilde_j[l] * X[idx]

#     #     W_j[t] = w_sum
#     #     V_j[t] = v_sum

#     # The following implements causal boundary handling

#     for t in range(N):
#         # If we don't have a full set of L_j values available (i.e., t < L_j - 1)
#         # we assign missing values
#         if t < L_j - 1:
#             W_j[t] = np.nan
#             V_j[t] = np.nan
#         else:
#             # Compute W_j[t] and V_j[t] using the L_j samples ending at index t
#             w_sum = 0.0
#             v_sum = 0.0
#             for l in range(L_j):
#                 idx = t - l
#                 w_sum += h_tilde_j[l] * X[idx]
#                 v_sum += g_tilde_j[l] * X[idx]
            
#             W_j[t] = w_sum
#             V_j[t] = v_sum

#     return W_j, V_j


