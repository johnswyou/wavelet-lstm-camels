import pandas as pd
import numpy as np
# import pywt
from utils import load_pickle

# class MODWTFeatureEngineer:
#     """
#     A feature engineering class that computes the Maximal Overlap Discrete Wavelet Transform (MODWT)
#     for specified features in a pandas DataFrame and appends the resulting coefficients as new features.

#     Parameters:
#     -----------
#     wavelet : str, optional (default='db1')
#         The name of the wavelet to use. Must be one of the wavelets available in PyWavelets.
    
#     level : int, optional (default=1)
#         The decomposition level for the MODWT.
    
#     Attributes:
#     -----------
#     wavelet : pywt.Wavelet
#         The PyWavelet wavelet object.
    
#     level : int
#         The decomposition level.
    
#     L : int
#         The length of the wavelet filter.
    
#     L_J : int
#         The number of boundary coefficients to remove based on the equation:
#         \( L_J = (2^J - 1)(L - 1) + 1 \)
#     """

#     def __init__(self, wavelet='db1', level=1):
#         # Initialize wavelet and level
#         self.wavelet = pywt.Wavelet(wavelet)
#         self.level = level

#         # Length of the decomposition low-pass filter
#         self.L = len(self.wavelet.dec_lo)
        
#         # Calculate L_J based on the provided equation
#         self.L_J = (2**self.level - 1) * (self.L - 1) + 1
        
#         # Extract scaling (low-pass) and wavelet (high-pass) filters
#         self.scaling_filter = np.array(self.wavelet.dec_lo)
#         self.wavelet_filter = np.array(self.wavelet.dec_hi)
    
#     def modwt(self, signal):
#         """
#         Perform MODWT on a single 1D signal up to the specified level.

#         Parameters:
#         -----------
#         signal : numpy.ndarray
#             The input signal (1D array) to transform.

#         Returns:
#         --------
#         W : dict
#             Wavelet coefficients for each level.
        
#         V : dict
#             Scaling coefficients for each level.
#         """
#         W = {}
#         V = {}
#         current_signal = signal.copy()
        
#         for j in range(1, self.level + 1):
#             # Convolve with scaling filter (low-pass)
#             conv_low = np.convolve(current_signal, self.scaling_filter, mode='same')
            
#             # Convolve with wavelet filter (high-pass)
#             conv_high = np.convolve(current_signal, self.wavelet_filter, mode='same')
            
#             # Store coefficients
#             V[j] = conv_low
#             W[j] = conv_high
            
#             # Update the signal for the next level
#             current_signal = V[j]
        
#         return W, V
    
#     def transform(self, df, feature_columns):
#         """
#         Apply MODWT to specified feature columns and append the coefficients to the DataFrame.

#         Parameters:
#         -----------
#         df : pandas.DataFrame
#             The input DataFrame containing the features.
        
#         feature_columns : list of str
#             List of column names on which to perform MODWT.

#         Returns:
#         --------
#         pandas.DataFrame
#             The DataFrame with appended MODWT features.
#         """
#         for col in feature_columns:
#             # Extract the signal
#             signal = df[col].values
            
#             # Perform MODWT
#             W, V = self.modwt(signal)
            
#             # Extract coefficients at the specified level
#             Wj = W[self.level]
#             Vj = V[self.level]
            
#             # Remove the first L_J boundary coefficients
#             Wj_trimmed = Wj[self.L_J:]
#             Vj_trimmed = Vj[self.L_J:]
            
#             # To maintain the original DataFrame length, pad the removed coefficients with NaN
#             Wj_padded = np.concatenate((np.full(self.L_J, np.nan), Wj_trimmed))
#             Vj_padded = np.concatenate((np.full(self.L_J, np.nan), Vj_trimmed))
            
#             # Append the coefficients as new columns
#             df[f'{col}_W{self.level}'] = Wj_padded
#             df[f'{col}_V{self.level}'] = Vj_padded
        
#         return df

#     @staticmethod
#     def available_wavelets():
#         """
#         Retrieve a list of available wavelet names from PyWavelets.

#         Returns:
#         --------
#         list of str
#             Available wavelet names.
#         """
#         return pywt.wavelist()


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
        # self.wavelet = pywt.Wavelet(wavelet)
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
        
        # Extract scaling (low-pass) and wavelet (high-pass) filters
        # self.scaling_filter = np.array(self.wavelet.rec_lo)
        # self.wavelet_filter = np.array(self.wavelet.rec_hi)

    def modwt(self, signal):
        """
        Perform MODWT on a single 1D signal up to the specified maximum level.

        Parameters:
        -----------
        signal : numpy.ndarray
            The input signal (1D array) to transform.

        Returns:
        --------
        W : dict
            Wavelet coefficients for each level.
        
        V : dict
            Scaling coefficients for each level.
        """
        W = {}
        V = {}
        current_signal = signal.copy()
        
        for j in range(1, self.max_level + 1):
            # Convolve with scaling filter (low-pass)
            conv_low = np.convolve(current_signal, self.scaling_filter, mode='same')
            
            # Convolve with wavelet filter (high-pass)
            conv_high = np.convolve(current_signal, self.wavelet_filter, mode='same')
            
            # Store coefficients
            V[j] = conv_low
            W[j] = conv_high
            
            # Update the signal for the next level
            current_signal = V[j]
        
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
        for col in feature_columns:
            # Extract the signal
            signal = df[col].values
            
            # Perform MODWT
            W, V = self.modwt(signal)
            
            # Process wavelet coefficients
            for J in self.w_levels:
                Wj = W[J]
                L_J = self.L_J_w[J]
                
                # Remove the first L_J boundary coefficients
                Wj_trimmed = Wj[L_J:]
                
                # To maintain the original DataFrame length, pad the removed coefficients with NaN
                if len(Wj_trimmed) < len(Wj):
                    Wj_padded = np.concatenate((np.full(L_J, np.nan), Wj_trimmed))
                else:
                    Wj_padded = Wj_trimmed  # In case L_J is 0
                
                # Ensure the padded array has the same length as the original signal
                if len(Wj_padded) < len(signal):
                    padding_length = len(signal) - len(Wj_padded)
                    Wj_padded = np.concatenate((np.full(padding_length, np.nan), Wj_padded))
                elif len(Wj_padded) > len(signal):
                    Wj_padded = Wj_padded[:len(signal)]
                
                # Append the coefficients as a new column
                df[f'{col}_W{J}'] = Wj_padded
            
            # Process scaling coefficients
            for J in self.v_levels:
                Vj = V[J]
                L_J = self.L_J_v[J]
                
                # Remove the first L_J boundary coefficients
                Vj_trimmed = Vj[L_J:]
                
                # To maintain the original DataFrame length, pad the removed coefficients with NaN
                if len(Vj_trimmed) < len(Vj):
                    Vj_padded = np.concatenate((np.full(L_J, np.nan), Vj_trimmed))
                else:
                    Vj_padded = Vj_trimmed  # In case L_J is 0
                
                # Ensure the padded array has the same length as the original signal
                if len(Vj_padded) < len(signal):
                    padding_length = len(signal) - len(Vj_padded)
                    Vj_padded = np.concatenate((np.full(padding_length, np.nan), Vj_padded))
                elif len(Vj_padded) > len(signal):
                    Vj_padded = Vj_padded[:len(signal)]
                
                # Append the coefficients as a new column
                df[f'{col}_V{J}'] = Vj_padded
        
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
