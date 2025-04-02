import os
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Any


def load_site_rois_from_yaml(yaml_path: str) -> Tuple[List[int], List[List[List[int]]], float]:
    """
    Load site ROIs from a YAML configuration file.
    
    Parameters
    ----------
    yaml_path : str
        Path to the YAML configuration file
        
    Returns
    -------
    Tuple[List[int], List[List[List[int]]], float]
        A tuple containing:
        - atom_roi_xlims: The x limits for the atom ROI
        - site_rois: List of site ROIs, each containing x and y coordinate pairs
        - threshold: The threshold value for atom detection
    """
    yaml_path = Path(yaml_path)
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")
    
    with yaml_path.open('rt') as stream:
        loaded_yaml = yaml.safe_load(stream)
    
    # Extract the site ROIs, atom ROI x limits, and threshold
    site_roi_arr = loaded_yaml['site_rois']
    atom_roi_xlims = loaded_yaml['atom_roi_xlims']
    threshold = loaded_yaml['threshold']
    
    return atom_roi_xlims, site_roi_arr, threshold


class CustomYAMLDumper(yaml.Dumper):
    """Custom YAML dumper to format nested lists in the desired format."""
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def format_site_rois_yaml(site_rois):
    """Format site_rois as a YAML string in the exact format of roi_config.yml."""
    yaml_lines = ["site_rois:"]
    for site in site_rois:
        yaml_lines.append(f"  - [[{site[0][0]}, {site[0][1]}],")
        yaml_lines.append(f"     [{site[1][0]}, {site[1][1]}]]")
    return "\n".join(yaml_lines)


def dump_site_rois_to_yaml(
    site_rois: List[List[List[int]]],
    atom_roi_xlims: List[int],
    threshold: float,
    output_path: str = None
) -> str:
    """
    Dump site ROIs to a YAML file in the same format as roi_config.yml.
    
    Parameters
    ----------
    site_rois : List[List[List[int]]]
        List of site ROIs, each containing x and y coordinate pairs
    atom_roi_xlims : List[int]
        The x limits for the atom ROI
    threshold : float
        The threshold value for atom detection
    output_path : str, optional
        Path to save the YAML file. If None, will save to 'roi_test.yml' 
        in the same directory as the original roi_config.yml
        
    Returns
    -------
    str
        Path to the created YAML file
    """
    # Determine the output path
    if output_path is None:
        # Get the directory of the current script
        current_dir = Path(__file__).parent
        output_path = current_dir / 'roi_test_2.yml'
    else:
        output_path = Path(output_path)
    
    # Format the YAML content manually to match the exact format of roi_config.yml
    yaml_content = "---\n"
    yaml_content += f"threshold: {threshold}\n"
    yaml_content += f"atom_roi_xlims: [{atom_roi_xlims[0]}, {atom_roi_xlims[1]}]\n"
    yaml_content += format_site_rois_yaml(site_rois)
    
    # Write the YAML file
    with output_path.open('w') as stream:
        stream.write(yaml_content)
    
    return str(output_path)


if __name__ == "__main__":
    # Example usage
    current_dir = Path(__file__).parent
    roi_config_path = current_dir / 'roi_test.yml'
    
    # Load the site ROIs from the original config
    atom_roi_xlims, site_rois, threshold = load_site_rois_from_yaml(roi_config_path)
    
    # Print some information
    print(f"Loaded {len(site_rois)} site ROIs")
    print(f"Atom ROI x limits: {atom_roi_xlims}")
    print(f"Threshold: {threshold}")
    
    # Dump the site ROIs to a new YAML file
    output_path = dump_site_rois_to_yaml(site_rois, atom_roi_xlims, threshold)
    print(f"Site ROIs dumped to: {output_path}")