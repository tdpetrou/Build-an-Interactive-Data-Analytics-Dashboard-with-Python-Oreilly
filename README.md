# Build an Interactive Data Analytics Dashboard with Python

Live Course with Teddy Petrou

## Getting setup before the course

It is vital that your machine is setup properly before the start of the course. You will need to complete the following instructions.

1. Verify you have Python 3.9+
1. Download the course material from the GitHub repository
1. Create the virtual environment
1. Launch the dashboard

### Verify you have Python 3.9+

1. Open your terminal/command prompt
1. If you installed Anaconda or Miniconda
    1. You should have a **base** environment
    1. Verify it is active by verifying that **`(base)`** is prepended to your prompt
    1. Run `python --version` to output the version
    1. If you have Python 3.8 or less upgrade by doing the following:
        1. Run `conda create -n py310 python=3.10`
        1. Run `conda deactivate`
        1. Run `conda activate py310`
        1. Now the `python` command is mapped to Python 3.10
1. If you don't use Anaconda or Miniconda you will need to verify you have at least Python 3.9 and complete an upgrade if necessary on your own

### Clone the GitHub repository

1. Navigate to the [course page][1] and click on the green **code** button
1. Click on the **Download ZIP** link from the dropdown menu. If you know git, you can clone the repository
1. Unzip the contents and move the folder to a proper location in your file system (i.e. do not keep it in your downloads folder)

### Create the virtual environment

1. Using the `cd` command
    1. Navigate to the folder you just unzipped and moved from above
    1. Navigate into the `project` directory
1. Run the command `python -m venv dashboard_venv`. This creates a new virtual environment named **dashboard_venv**
1. Deactivate the conda environment with `conda deactivate`
1. Activate the virtual environment with the following command:
    1. Mac/Linux - `source dashboard_venv/bin/activate`
    2. Windows - `dashboard_venv\Scripts\activate.bat`
1. There should be `(dashboard_venv)` prepended to your prompt
1. Run `pip install -U pip` to upgrade pip to the latest version
1. Run `pip install -r requirements.txt` to install all the necessary packages into this environment. This will take some time to complete

## Launch the dashboard

1. Run the command `python dashboard.py`
1. The following text should be printed to the screen - **Dash is running on http://127.0.0.1:8050/**
1. Open your web browser and navigate to 127.0.0.1:8050
1. You should see the coronavirus forecasting dashboard

![2]

[1]: https://github.com/tdpetrou/Build-an-Interactive-Data-Analytics-Dashboard-with-Python-Oreilly
[2]: images/dashboardss.png