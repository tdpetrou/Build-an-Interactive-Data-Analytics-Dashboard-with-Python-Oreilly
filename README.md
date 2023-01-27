# Build an Interactive Data Analytics Dashboard with Python

Live Course with Teddy Petrou

## Getting setup before the course

It is vital that your machine is setup properly before the start of the course. You will need to complete each step of the following instructions.

1. Download the course material from the GitHub repository
1. Create the virtual environment
1. Launch the dashboard

### Clone the GitHub repository

1. Navigate to the [course page][1] and click on the green **code** button
1. Click on the **Download ZIP** link from the dropdown menu. If you know git, you can clone the repository
1. Unzip the contents and move the folder to a proper location in your file system (i.e. do not keep it in your downloads folder)

### Create the virtual environment

1. Open your terminal/command prompt and navigate to the folder you just unzipped and moved from above
1. Navigate into the `project` directory
1. Run the command `python -m venv dashboard_venv`. This creates a new virtual environment named **dashboard_venv**
1. Activate the virtual environment with the following command:
    1. Mac/Linux - `source dashboard_venv/bin/activate`
    2. Windows - `dashboard_venv\Scripts\activate.bat`
1. There should be `(dashboard_venv)` prepended to your prompt
1. Run `pip install -U pip` to upgrade pip to the latest version
1. Run `pip install wheel` to install the wheel package, which helps install the other packages
1. Run `pip install -r requirements.txt` to install all the necessary packages into this environment. This will take some time to complete

[1]: https://github.com/tdpetrou/Build-an-Interactive-Data-Analytics-Dashboard-with-Python-Oreilly