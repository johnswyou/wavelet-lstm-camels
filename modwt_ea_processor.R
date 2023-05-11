# devtools::install_github("johnswyou/fastWavelets")

library(fastWavelets)
library(rlist)
library(hydroIVS)
library(dplyr)

fixed_column_names <- c("date","Q","flag","dayl.s.","prcp.mm.day.","srad.W.m2.","swe.mm.","tmax.C.","tmin.C.","vp.Pa.")

modwt_ea_processor <- function(data_path, filter, forecast_horizon) {

  # Fixed parameters
  J <- 6
  L <- 14
  val_ratio <- 0.15 
  test_ratio <- 0.15
  
  # Read full CSV data set as a data frame
  df <- read.csv(data_path)

  if (!identical(names(df), fixed_column_names)) {
    stop("Serious problem: column names of csv does not adhere to assumed form.")
  }

  # Get the number of rows in total, in the validation set, and in the test set
  N <- nrow(df)
  nval <- round(N*val_ratio)
  ntst <- round(N*test_ratio)

  # Full, unpartitioned data
  X <- df[, -c(1, 2, 3)] # Remove date, streamflow and flag columns
  y <- df[, 2]        # Extract streamflow column

  # Get a list of coefficients. Each list entry is a matrix of coefficients 
  # corresponding to an original input feature.
  start_time <- Sys.time()
  temp <- lapply(X, mo_dwt, filter, J, remove_boundary_coefs=FALSE)
  end_time <- Sys.time()
  elapsed_time <- end_time - start_time
  print(paste("Time for MODWT with ", filter, " and LT ", forecast_horizon, ": ", elapsed_time, " seconds"))

  # Convert temp into a matrix with appropriate column names
  wavelet_suffix <- paste0("W", 1:J)
  scaling_suffix <- paste0("V", J)
  suffixes <- c(wavelet_suffix, scaling_suffix)

  column_names <- c()

  for (i in names(temp)) {
    column_names <- c(column_names, paste0(i, "_", suffixes))
  }

  temp <- rlist::list.cbind(temp)
  colnames(temp) <- column_names

  # NOTE: Up until this point, temp does not contain date, streamflow, and only contains wavelet and scaling coefficients

  # Remove boundary rows
  if (is.null(L)){
    L <- length(r_scaling_filter(filter))
  }
    
  LJ2cut <- ((2^J)-1)*(L-1)
  streamflow <- tail(y, -LJ2cut)
  X <- tail(as.matrix(X), -LJ2cut)
  temp <- tail(temp, -LJ2cut)

  # temp now contains original input features + coefficients + target.
  # Boundary rows have been removed. There is no date column in temp.
  temp <- cbind(X, temp, streamflow)

  # Split temp into train/val/test splits
  temp_test <- tail(temp, ntst)
  temp_val <- tail(head(temp, -ntst), nval)
  temp_train <- head(temp, -(nval+ntst))

  # Perform IVS using the training set and subset temp
  temp_train_X = temp_train[1:(nrow(temp_train)-forecast_horizon), 1:(ncol(temp_train)-1)]
  streamflow = temp_train[(forecast_horizon+1):nrow(temp_train), ncol(temp_train)]

  start_time <- Sys.time()

  ivs_obj <- hydroIVS::ivsIOData(streamflow, temp_train_X, "ea_cmi_tol", 0.05)

  end_time <- Sys.time()

  elapsed_time <- end_time - start_time

  print(paste("Time for EA with ", filter, " and LT ", forecast_horizon, ": ", elapsed_time, " seconds"))

  # temp_X = temp[, 1:(ncol(temp)-1)]
  # streamflow = temp[, ncol(temp)]

  # temp_X <- temp_X[,ivs_obj$names_sel_inputs, drop=FALSE]
  # temp <- cbind(temp_X, streamflow)
  # temp <- as.data.frame(temp)

  # return(temp)

  # **********************************************************************************
  # Note: In the following, temp_train, temp_val and temp_test DO NOT have the target
  # streamflow variable leaded.
  # **********************************************************************************

  # Training set

  temp_train_X <- temp_train[, 1:(ncol(temp_train)-1)]
  streamflow <- temp_train[, ncol(temp_train)]
  temp_train_X <- temp_train_X[,ivs_obj$names_sel_inputs, drop=FALSE]
  temp_train <- cbind(temp_train_X, streamflow)

  # Validation Set

  temp_val_X <- temp_val[, 1:(ncol(temp_val)-1)]
  streamflow <- temp_val[, ncol(temp_val)]
  temp_val_X <- temp_val_X[,ivs_obj$names_sel_inputs, drop=FALSE]
  temp_val <- cbind(temp_val_X, streamflow)

  # Test Set

  temp_test_X <- temp_test[, 1:(ncol(temp_test)-1)]
  streamflow <- temp_test[, ncol(temp_test)]
  temp_test_X <- temp_test_X[,ivs_obj$names_sel_inputs, drop=FALSE]
  temp_test <- cbind(temp_test_X, streamflow)

  # Convert Matrices to Data Frames

  temp_train <- as.data.frame(temp_train)
  temp_val <- as.data.frame(temp_val)
  temp_test <- as.data.frame(temp_test)

  return(list(train=temp_train,
              val=temp_val,
              test=temp_test))

}

ea_processor <- function(data_path, forecast_horizon) {

  # Fixed parameters
  J <- 6
  L <- 14
  val_ratio <- 0.15 
  test_ratio <- 0.15
  
  # Read full CSV data set as a data frame
  df <- read.csv(data_path)

  if (!identical(names(df), fixed_column_names)) {
    stop("Serious problem: column names of csv does not adhere to assumed form.")
  }

  # Get the number of rows in total, in the validation set, and in the test set
  N <- nrow(df)
  nval <- round(N*val_ratio)
  ntst <- round(N*test_ratio)

  # Full, unpartitioned data
  X <- df[, -c(1, 2, 3)] # Remove date, streamflow and flag columns
  y <- df[, 2]        # Extract streamflow column
    
  LJ2cut <- ((2^J)-1)*(L-1)
  streamflow <- tail(y, -LJ2cut)
  X <- tail(as.matrix(X), -LJ2cut)

  # temp now contains original input features + target.
  # Boundary rows have been removed. There is no date column in temp.
  temp <- cbind(X, streamflow)

  # Split temp into train/val/test splits
  temp_test <- tail(temp, ntst)
  temp_val <- tail(head(temp, -ntst), nval)
  temp_train <- head(temp, -(nval+ntst))

  # Perform IVS using the training set and subset temp
  temp_train_X = temp_train[1:(nrow(temp_train)-forecast_horizon), 1:(ncol(temp_train)-1)]
  streamflow = temp_train[(forecast_horizon+1):nrow(temp_train), ncol(temp_train)]

  start_time <- Sys.time()

  ivs_obj <- hydroIVS::ivsIOData(streamflow, temp_train_X, "ea_cmi_tol", 0.05)

  end_time <- Sys.time()

  elapsed_time <- end_time - start_time

  print(paste("Time for EA with LT ", forecast_horizon, ": ", elapsed_time, " seconds"))

  # temp_X = temp[, 1:(ncol(temp)-1)]
  # streamflow = temp[, ncol(temp)]

  # temp_X <- temp_X[,ivs_obj$names_sel_inputs, drop=FALSE]
  # temp <- cbind(temp_X, streamflow)
  # temp <- as.data.frame(temp)

  # return(temp)

  # **********************************************************************************
  # Note: In the following, temp_train, temp_val and temp_test DO NOT have the target
  # streamflow variable leaded.
  # **********************************************************************************

  # Training set

  temp_train_X <- temp_train[, 1:(ncol(temp_train)-1)]
  streamflow <- temp_train[, ncol(temp_train)]
  temp_train_X <- temp_train_X[,ivs_obj$names_sel_inputs, drop=FALSE]
  temp_train <- cbind(temp_train_X, streamflow)

  # Validation Set

  temp_val_X <- temp_val[, 1:(ncol(temp_val)-1)]
  streamflow <- temp_val[, ncol(temp_val)]
  temp_val_X <- temp_val_X[,ivs_obj$names_sel_inputs, drop=FALSE]
  temp_val <- cbind(temp_val_X, streamflow)

  # Test Set

  temp_test_X <- temp_test[, 1:(ncol(temp_test)-1)]
  streamflow <- temp_test[, ncol(temp_test)]
  temp_test_X <- temp_test_X[,ivs_obj$names_sel_inputs, drop=FALSE]
  temp_test <- cbind(temp_test_X, streamflow)

  # Convert Matrices to Data Frames

  temp_train <- as.data.frame(temp_train)
  temp_val <- as.data.frame(temp_val)
  temp_test <- as.data.frame(temp_test)

  return(list(train=temp_train,
              val=temp_val,
              test=temp_test))

}