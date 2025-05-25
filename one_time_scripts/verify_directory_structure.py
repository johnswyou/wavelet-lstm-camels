#!/usr/bin/env python3
"""
Directory Structure Verification Script

This script verifies that each STATION_ID directory under mnt/correct_output
has the same expected structure. It checks for:
1. Presence of leadtime directories (leadtime_1, leadtime_3, leadtime_5)
2. Presence of all expected wavelet directories under each leadtime
3. Presence of all expected files in each wavelet directory

If any deviations are found, they are reported.
"""

import os
from pathlib import Path
from typing import Set, List, Dict, Tuple

# Define the expected structure
EXPECTED_LEADTIMES = {'leadtime_1', 'leadtime_3', 'leadtime_5'}

EXPECTED_WAVELETS = {
    'bl7', 'coif1', 'coif2', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7',
    'fk14', 'fk4', 'fk6', 'fk8', 'han2_3', 'han3_3', 'han4_5', 'han5_5',
    'la10', 'la12', 'la14', 'la8', 'mb10_3', 'mb12_3', 'mb14_3', 'mb4_2',
    'mb8_2', 'mb8_3', 'mb8_4', 'sym4', 'sym5', 'sym6', 'sym7'
}

EXPECTED_FILES = {
    'baseline_feature_scaler.pkl',
    'baseline_history.pkl',
    'baseline_model.keras',
    'baseline_pred_label_df.pkl',
    'baseline_q_scaler.pkl',
    'baseline_test_metrics_dict.pkl',
    'ea_cmi_tol_005_selected_feature_names.pkl',
    'feature_scaler.pkl',
    'history.pkl',
    'model.keras',
    'pred_label_df.pkl',
    'q_scaler.pkl',
    'test_metrics_dict.pkl',
    'timings.pkl'
}

def get_directory_contents(path: Path) -> Set[str]:
    """Get the set of directory names in the given path."""
    if not path.exists() or not path.is_dir():
        return set()
    return {item.name for item in path.iterdir() if item.is_dir()}

def get_file_contents(path: Path) -> Set[str]:
    """Get the set of file names in the given path."""
    if not path.exists() or not path.is_dir():
        return set()
    return {item.name for item in path.iterdir() if item.is_file()}

def verify_station_structure(station_path: Path) -> Dict[str, List[str]]:
    """
    Verify the structure of a single station directory.
    Returns a dictionary of issues found.
    """
    issues = {
        'missing_leadtimes': [],
        'extra_leadtimes': [],
        'missing_wavelets': [],
        'extra_wavelets': [],
        'missing_files': [],
        'extra_files': []
    }
    
    # Check leadtime directories
    actual_leadtimes = get_directory_contents(station_path)
    missing_leadtimes = EXPECTED_LEADTIMES - actual_leadtimes
    extra_leadtimes = actual_leadtimes - EXPECTED_LEADTIMES
    
    if missing_leadtimes:
        issues['missing_leadtimes'] = list(missing_leadtimes)
    if extra_leadtimes:
        issues['extra_leadtimes'] = list(extra_leadtimes)
    
    # Check each leadtime directory
    for leadtime in EXPECTED_LEADTIMES:
        leadtime_path = station_path / leadtime
        if not leadtime_path.exists():
            continue
            
        # Check wavelet directories
        actual_wavelets = get_directory_contents(leadtime_path)
        missing_wavelets = EXPECTED_WAVELETS - actual_wavelets
        extra_wavelets = actual_wavelets - EXPECTED_WAVELETS
        
        if missing_wavelets:
            for wavelet in missing_wavelets:
                issues['missing_wavelets'].append(f"{leadtime}/{wavelet}")
        if extra_wavelets:
            for wavelet in extra_wavelets:
                issues['extra_wavelets'].append(f"{leadtime}/{wavelet}")
        
        # Check files in each wavelet directory
        for wavelet in EXPECTED_WAVELETS:
            wavelet_path = leadtime_path / wavelet
            if not wavelet_path.exists():
                continue
                
            actual_files = get_file_contents(wavelet_path)
            missing_files = EXPECTED_FILES - actual_files
            extra_files = actual_files - EXPECTED_FILES
            
            if missing_files:
                for file in missing_files:
                    issues['missing_files'].append(f"{leadtime}/{wavelet}/{file}")
            if extra_files:
                for file in extra_files:
                    issues['extra_files'].append(f"{leadtime}/{wavelet}/{file}")
    
    return issues

def main():
    """Main verification function."""
    base_path = Path("mnt/correct_output")
    
    if not base_path.exists():
        print(f"Error: Directory {base_path} does not exist!")
        return
    
    # Get all station directories
    station_dirs = [item for item in base_path.iterdir() 
                   if item.is_dir() and item.name.isdigit()]
    
    if not station_dirs:
        print(f"No station directories found in {base_path}")
        return
    
    print(f"Verifying directory structure for {len(station_dirs)} stations...")
    print(f"Expected structure:")
    print(f"  - Leadtimes: {sorted(EXPECTED_LEADTIMES)}")
    print(f"  - Wavelets per leadtime: {len(EXPECTED_WAVELETS)}")
    print(f"  - Files per wavelet: {len(EXPECTED_FILES)}")
    print("-" * 80)
    
    stations_with_issues = 0
    total_issues = 0
    
    for station_dir in sorted(station_dirs, key=lambda x: x.name):
        station_id = station_dir.name
        issues = verify_station_structure(station_dir)
        
        # Check if there are any issues
        has_issues = any(issue_list for issue_list in issues.values())
        
        if has_issues:
            stations_with_issues += 1
            print(f"\n❌ STATION {station_id} - ISSUES FOUND:")
            
            for issue_type, issue_list in issues.items():
                if issue_list:
                    total_issues += len(issue_list)
                    print(f"  {issue_type.replace('_', ' ').title()}:")
                    for issue in sorted(issue_list):
                        print(f"    - {issue}")
        else:
            print(f"✅ STATION {station_id} - OK")
    
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY:")
    print(f"Total stations checked: {len(station_dirs)}")
    print(f"Stations with issues: {stations_with_issues}")
    print(f"Stations without issues: {len(station_dirs) - stations_with_issues}")
    print(f"Total issues found: {total_issues}")
    
    if stations_with_issues == 0:
        print("\n🎉 All stations have the expected directory structure!")
    else:
        print(f"\n⚠️  {stations_with_issues} station(s) have structural deviations.")

if __name__ == "__main__":
    main() 