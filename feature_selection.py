import numpy as np
import pandas as pd
from scipy.linalg import det, eigh
from itertools import combinations
import time

class EAcmiFrameworkTol:
    def __init__(self, thresh=0.01, silent=False):
        """
        Initialize the EAcmiFrameworkTol class.

        Parameters:
        - thresh (float): Threshold for stopping criterion based on CMI/MI ratio.
        - silent (bool): If True, suppresses the print statements.
        """
        self.thresh = thresh
        self.silent = silent

    def standardize_X(self, X):
        """
        Standardize the input matrix X to have zero mean and unit variance.

        Columns with any NaNs are replaced with zeros.

        Parameters:
        - X (numpy.ndarray): Input data matrix.

        Returns:
        - Xs (numpy.ndarray): Standardized data matrix.
        """
        X = np.asarray(X)
        X_mean = np.nanmean(X, axis=0)
        X_sd = np.nanstd(X, axis=0, ddof=1)
        
        # Avoid division by zero
        X_sd_replaced = np.where(X_sd == 0, 1, X_sd)
        Xs = (X - X_mean) / X_sd_replaced

        # Replace columns with any NaNs with zeros
        cols_with_nan = np.any(np.isnan(Xs), axis=0)
        if np.any(cols_with_nan):
            Xs[:, cols_with_nan] = 0

        return Xs

    def MAD(self, Z):
        """
        Compute the Median Absolute Deviation (MAD) based Z-score.

        Parameters:
        - Z (numpy.ndarray): Input array.

        Returns:
        - Z_score (numpy.ndarray): MAD-based Z-scores.
        """
        median_Z = np.nanmedian(Z)
        MAD_Z = np.abs(Z - median_Z)
        S = 1.4826 * np.nanmedian(MAD_Z)
        Z_score = MAD_Z / S
        return Z_score

    def whiten_E0(self, X):
        """
        Whiten the dataset X by centering it to have zero mean.

        Columns with any NaNs are replaced with zeros.

        Parameters:
        - X (numpy.ndarray): Input data matrix.

        Returns:
        - Xw (numpy.ndarray): Centered data matrix.
        """
        X = np.asarray(X)
        X_mean = np.nanmean(X, axis=0)
        Xw = X - X_mean

        # Replace columns with any NaNs with zeros
        cols_with_nan = np.any(np.isnan(Xw), axis=0)
        if np.any(cols_with_nan):
            Xw[:, cols_with_nan] = 0

        return Xw

    def std_X(self, X):
        """
        Compute the standard deviation of each column in X.

        Parameters:
        - X (numpy.ndarray): Input data matrix.

        Returns:
        - Xsd (numpy.ndarray): Standard deviations of columns.
        """
        Xsd = np.nanstd(X, axis=0, ddof=1)
        return Xsd

    def H_whiten(self, s):
        """
        Compute the entropy scaling term based on standard deviations.

        Parameters:
        - s (numpy.ndarray): Standard deviations of variables.

        Returns:
        - H_w (float): Entropy scaling term.
        """
        H_w = np.log(np.prod(s))
        return H_w

    def H_normal(self, X):
        """
        Compute the Shannon entropy of a multivariate normal distribution
        with the covariance of X.

        Parameters:
        - X (numpy.ndarray): Input data matrix.

        Returns:
        - H_w (float): Shannon entropy.
        """
        X = np.atleast_2d(X)  # Ensure X is at least 2-D
        print(f"H_normal input shape: {X.shape}")  # Debugging statement
        d = X.shape[1]
        cov_X = np.cov(X, rowvar=False)
        print(f"cov_X shape: {cov_X.shape}")  # Debugging statement

        # Ensure covariance matrix is at least 2-D
        if cov_X.ndim == 0:
            cov_X = np.array([[cov_X]])
            print(f"Adjusted cov_X shape: {cov_X.shape}")  # Debugging statement

        # Handle cases where covariance matrix might not be positive definite
        try:
            det_cov = det(cov_X)
            if det_cov <= 0:
                # Handle non-positive definite covariance matrices
                eigvals = eigh(cov_X, eigvals_only=True)
                positive_eigvals = eigvals[eigvals > 0]
                if positive_eigvals.size == 0:
                    det_cov = 1e-12  # Assign a very small positive number to prevent log(0)
                else:
                    det_cov = np.prod(positive_eigvals)
        except np.linalg.LinAlgError:
            # In case of numerical issues, use eigenvalues
            eigvals = eigh(cov_X, eigvals_only=True)
            positive_eigvals = eigvals[eigvals > 0]
            if positive_eigvals.size == 0:
                det_cov = 1e-12  # Assign a very small positive number to prevent log(0)
            else:
                det_cov = np.prod(positive_eigvals)
        
        # Prevent log(0) by ensuring det_cov is positive and non-zero
        det_cov = det_cov if det_cov > 0 else 1e-12

        H_w = 0.5 * np.log(det_cov) + (d / 2) * np.log(2 * np.pi) + (d / 2)
        return H_w

    def Edgeworth_t1_t2_t3(self, X):
        """
        Compute the Edgeworth expansion terms t1, t2, t3.

        Parameters:
        - X (numpy.ndarray): Input data matrix.

        Returns:
        - terms (dict): Dictionary containing t1, t2, t3.
        """
        X = np.asarray(X)
        d = X.shape[1]

        # t1
        t1 = 0
        for i in range(d):
            kappa_iii = np.nanmean(X[:, i] ** 3)
            t1 += kappa_iii ** 2

        # t2
        t2 = 0
        if d > 1:
            for i, j in combinations(range(d), 2):
                kappa_iij = np.nanmean((X[:, i] ** 2) * X[:, j])
                t2 += kappa_iij ** 2
            t2 *= 6  # 3 * 2 * combination count

        # t3
        t3 = 0
        if d > 2:
            for i, j, k in combinations(range(d), 3):
                kappa_ijk = np.nanmean(X[:, i] * X[:, j] * X[:, k])
                t3 += kappa_ijk ** 2
            t3 /= 6

        return {'t1': t1, 't2': t2, 't3': t3}

    def H_EdgeworthApprox(self, X):
        """
        Compute the Edgeworth Approximation-based differential Shannon entropy.

        Parameters:
        - X (numpy.ndarray): Input data matrix.

        Returns:
        - H_ea (float): Edgeworth Approximation-based entropy.
        """
        X = np.atleast_2d(X)  # Ensure X is at least 2-D
        print(f"H_EdgeworthApprox input shape: {X.shape}")  # Debugging statement
        Xw = self.whiten_E0(X)
        s = self.std_X(Xw)
        Y = self.standardize_X(X)
        H_w = self.H_whiten(s)
        H_n = self.H_normal(Y)
        ea = self.Edgeworth_t1_t2_t3(Y)
        H_ea = (H_n - (ea['t1'] + ea['t2'] + ea['t3']) / 12) + H_w
        return H_ea

    def MI_EdgeworthApprox(self, Y, X):
        """
        Compute the Edgeworth Approximation-based Mutual Information between Y and X.

        Parameters:
        - Y (numpy.ndarray): Target variable array.
        - X (numpy.ndarray): Input variable array or matrix.

        Returns:
        - MI_ea (float): Mutual Information.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        H_Y = self.H_EdgeworthApprox(Y.reshape(-1, 1))
        H_X = self.H_EdgeworthApprox(X)
        H_YX = self.H_EdgeworthApprox(np.hstack((Y.reshape(-1,1), X)))
        MI_ea = H_Y + H_X - H_YX
        MI_ea = max(0, np.real(MI_ea))
        return MI_ea

    def CMI_EdgeworthApprox(self, Y, X, Z):
        """
        Compute the Edgeworth Approximation-based Conditional Mutual Information I(Y; X | Z).

        Parameters:
        - Y (numpy.ndarray): Target variable array.
        - X (numpy.ndarray): Input variable array.
        - Z (numpy.ndarray): Conditioning variable matrix.

        Returns:
        - CMI_ea (float): Conditional Mutual Information.
        """
        if Z.size == 0:
            # If Z is empty, CMI reduces to MI
            return self.MI_EdgeworthApprox(Y, X)
        
        H_YZ = self.H_EdgeworthApprox(np.hstack((Y.reshape(-1,1), Z)))
        H_XZ = self.H_EdgeworthApprox(np.hstack((X.reshape(-1,1), Z)))
        H_Z = self.H_EdgeworthApprox(Z)
        H_YXZ = self.H_EdgeworthApprox(np.hstack((Y.reshape(-1,1), X.reshape(-1,1), Z)))
        CMI_ea = H_YZ + H_XZ - H_Z - H_YXZ
        CMI_ea = max(0, np.real(CMI_ea))
        return CMI_ea

    def select_features(self, x, y):
        """
        Perform feature selection using the Edgeworth Approximation-based CMI framework.

        Parameters:
        - x (numpy.ndarray or pandas.DataFrame): Input features matrix.
        - y (numpy.ndarray or pandas.Series): Target variable array.

        Returns:
        - scores (pandas.DataFrame): DataFrame containing selected features and their scores.
        """
        start_time = time.time()
        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(y, pd.Series):
            y = y.values.reshape(-1, 1)
        else:
            y = y.reshape(-1, 1)
        
        n_inputs = x.shape[1]
        n_data = x.shape[0]
        # Generate input names starting from X1 to match R's 1-based indexing
        inp_names = [f'X{i+1}' for i in range(n_inputs)] if not isinstance(x, pd.DataFrame) else list(x.columns)

        # Initialize
        nfevals_2 = 0  # Number of CMI evaluations
        y_stand = self.standardize_X(y)
        x_stand = self.standardize_X(x)

        input_tracker = list(range(n_inputs))  # 0-based indices
        n_selected = 0
        z_in = np.array([]).reshape(n_data, 0)
        scores = pd.DataFrame()
        max_iter = x_stand.shape[1] + 1

        for iter_1 in range(1, max_iter +1):
            print(f"\nIteration {iter_1}:")
            print(f"Remaining features: {[inp_names[idx] for idx in input_tracker]}")
            if n_selected > 0:
                # Compute CMI for each remaining input
                CMI = []
                current_inputs = input_tracker.copy()
                for idx in current_inputs:
                    z = x_stand[:, idx].reshape(-1, 1)
                    if z_in.size == 0:
                        cmi = self.CMI_EdgeworthApprox(y_stand, z, z_in)
                    else:
                        cmi = self.CMI_EdgeworthApprox(y_stand, z, z_in)
                    CMI.append(cmi)
                    nfevals_2 +=1
                CMI = np.array(CMI)
                print(f"Computed CMI: {CMI}")

                if len(CMI) ==0:
                    print("No remaining features to evaluate. Exiting.")
                    break

                best_idx = np.argmax(CMI)
                best_CMI = CMI[best_idx]
                tag = best_idx

                if best_CMI <=0:
                    print(f"Best CMI ({best_CMI}) <= 0. Stopping selection.")
                    break

                selected_input = input_tracker[tag]
                if not self.silent:
                    print(f"\nSelect input {inp_names[selected_input]} ....")

                # Update z_in
                z_in = np.hstack((z_in, x_stand[:, selected_input].reshape(-1,1)))
                n_selected +=1

                # Compute MI_aug and CMI_MI_ratio
                MI_aug = self.MI_EdgeworthApprox(y_stand, z_in)
                CMI_MI_ratio = best_CMI / MI_aug if MI_aug !=0 else 0

                # Record scores
                score_entry = {
                    'Input': inp_names[selected_input],
                    'CMI': round(best_CMI, 4),
                    'MI': round(MI_aug, 4),
                    'CMI_MI_ratio': round(CMI_MI_ratio, 4),
                    'CMIevals': nfevals_2,
                    'CPUtime': round(time.time() - start_time, 4),
                    'ElapsedTime': round(time.time() - start_time, 4)
                }
                scores = pd.concat([scores, pd.DataFrame([score_entry])], ignore_index=True)

                if not self.silent:
                    print(scores.tail(1).to_string(index=False))

                # Check stopping conditions
                if iter_1 >2 and (CMI_MI_ratio <= self.thresh):
                    print(f"CMI_MI_ratio ({CMI_MI_ratio}) <= thresh ({self.thresh}). Stopping selection.")
                    break

                if iter_1 >3:
                    if (scores['CMI_MI_ratio'].iloc[-1] > scores['CMI_MI_ratio'].iloc[-2] > scores['CMI_MI_ratio'].iloc[-3]):
                        print("CMI_MI_ratio has been increasing for the last three iterations. Stopping selection.")
                        break

                # Remove selected input from tracker
                input_tracker.pop(tag)
                n_inputs -=1

            else:
                # Compute MI for each input
                MI = []
                current_inputs = input_tracker.copy()
                for idx in current_inputs:
                    z = x_stand[:, idx].reshape(-1,1)
                    mi = self.MI_EdgeworthApprox(y_stand, z)
                    MI.append(mi)
                    nfevals_2 +=1
                MI = np.array(MI)
                print(f"Computed MI: {MI}")

                if len(MI) ==0:
                    print("No remaining features to evaluate. Exiting.")
                    break

                best_idx = np.argmax(MI)
                best_MI = MI[best_idx]
                tag = best_idx

                if best_MI <=0:
                    print(f"Best MI ({best_MI}) <= 0. Stopping selection.")
                    break

                selected_input = input_tracker[tag]
                if not self.silent:
                    print(f"\nSelect input {inp_names[selected_input]} ....")

                # Update z_in
                z_in = np.hstack((z_in, x_stand[:, selected_input].reshape(-1,1)))
                n_selected +=1

                # Compute MI_aug and CMI_MI_ratio
                MI_aug = self.MI_EdgeworthApprox(y_stand, z_in)
                CMI_MI_ratio = best_MI / MI_aug if MI_aug !=0 else 0

                # Record scores
                score_entry = {
                    'Input': inp_names[selected_input],
                    'CMI': round(best_MI, 4),
                    'MI': round(MI_aug, 4),
                    'CMI_MI_ratio': round(CMI_MI_ratio, 4),
                    'CMIevals': nfevals_2,
                    'CPUtime': round(time.time() - start_time, 4),
                    'ElapsedTime': round(time.time() - start_time, 4)
                }
                scores = pd.concat([scores, pd.DataFrame([score_entry])], ignore_index=True)

                if not self.silent:
                    print(scores.tail(1).to_string(index=False))

                # Check stopping conditions
                if iter_1 >2 and (CMI_MI_ratio <= self.thresh):
                    print(f"CMI_MI_ratio ({CMI_MI_ratio}) <= thresh ({self.thresh}). Stopping selection.")
                    break

                if iter_1 >3:
                    if (scores['CMI_MI_ratio'].iloc[-1] > scores['CMI_MI_ratio'].iloc[-2] > scores['CMI_MI_ratio'].iloc[-3]):
                        print("CMI_MI_ratio has been increasing for the last three iterations. Stopping selection.")
                        break

                # Remove selected input from tracker
                input_tracker.pop(tag)
                n_inputs -=1

        print("\nEA_CMI_TOL ROUTINE COMPLETED\n")
        print(scores.to_string(index=False))

        return scores.iloc[:iter_1-2] if iter_1 >2 else scores

# Example Usage:
# Assuming you have your data in pandas DataFrame `df_x` and Series `df_y`

if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # Generate synthetic data
    np.random.seed(0)
    df_x = pd.DataFrame({
        'Feature1': np.random.randn(100),
        'Feature2': np.random.randn(100),
        'Feature3': np.random.randn(100)
    })
    df_y = pd.Series(np.random.randn(100))

    # Initialize and run the feature selector
    selector = EAcmiFrameworkTol(thresh=0.01, silent=False)
    selected_features = selector.select_features(df_x, df_y)
    print("\nSelected Features:")
    print(selected_features)
