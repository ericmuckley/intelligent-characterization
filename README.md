# Intelligent materials characterization
This repository contains example data poduced by the [automated materials characterization system](https://github.com/ericmuckley/IMES).

## Description of files
* `/images/`: directory with images produced from processing the experimental data
* `/scripts/process_imes_data.py`: used for processing raw output from a IMES experiment into structured JSON data which is ready for plotting
* `/data/`: directory with example data produced by the characterization system
    * `.csv` files: raw data prouced directly from the characterization system
    * `.xlsx` files: data compiled from multiple raw `.csv` files
    * `.opju` files: Origin Pro files used for plotting data from the raw data files
    * `.ini` files: examle initialization files using for configuring experiments in the characterization system