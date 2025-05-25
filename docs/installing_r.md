I'll show you how to install R version 4.4 on Ubuntu running in WSL.

## Method 1: Install from CRAN Repository (Recommended)

### 1. Update your system
```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Install required dependencies
```bash
sudo apt install -y software-properties-common dirmngr wget
```

### 3. Add the CRAN GPG key
```bash
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
```

### 4. Add the CRAN repository
For Ubuntu 22.04 (which you're using based on your WSL setup):
```bash
sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
```

### 5. Update package list and install R
```bash
sudo apt update
sudo apt install -y r-base r-base-dev
```

### 6. Verify the installation
```bash
R --version
```

This should show R version 4.4.x.

## Method 2: Install Specific Version (if needed)

If you need to ensure you get exactly version 4.4, you can:

### 1. Check available versions
```bash
apt-cache policy r-base
```

### 2. Install specific version (if available)
```bash
sudo apt install r-base=4.4.* r-base-dev=4.4.*
```

## Additional Recommended Packages

Install some commonly needed packages for R development:

```bash
# For compiling R packages
sudo apt install -y build-essential libcurl4-openssl-dev libssl-dev libxml2-dev

# For statistical computing
sudo apt install -y libgsl-dev

# For graphics and plotting
sudo apt install -y libcairo2-dev libxt-dev

# For database connectivity
sudo apt install -y unixodbc-dev
```

## Post-Installation Setup

### 1. Test R installation
```bash
R
```

In the R console, try:
```r
# Check version
version

# Install a test package
install.packages("ggplot2")

# Exit R
q()
```

### 2. Optional: Install RStudio Server (for web-based IDE)
```bash
# Download and install RStudio Server
wget https://download2.rstudio.org/server/jammy/amd64/rstudio-server-2023.12.1-402-amd64.deb
sudo dpkg -i rstudio-server-2023.12.1-402-amd64.deb
sudo apt-get install -f  # Fix any dependency issues

# Start RStudio Server
sudo systemctl start rstudio-server
sudo systemctl enable rstudio-server
```

Then access RStudio at `http://localhost:8787` in your browser.

## Troubleshooting

If you encounter issues:

1. **Permission errors**: Make sure you're using `sudo` for system-wide installation
2. **Package compilation errors**: Install the development packages mentioned above
3. **Repository errors**: Verify your Ubuntu version with `lsb_release -a` and adjust the repository URL accordingly

The installation should give you R version 4.4.x, which is the latest stable release. You can verify this by running `R --version` after installation.
