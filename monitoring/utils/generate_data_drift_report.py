import pandas as pd
from evidently.report import Report
from evidently.ui.workspace import Workspace
from evidently.metric_preset import ClassificationPreset
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset



import logging
import src.tools.tools as tools
import os 
from pathlib import Path

logger = logging.getLogger(__name__)

def get_baseline_data() -> pd.DataFrame:
    """
    Get the baseline data.
    Returns:
        pd.DataFrame: Baseline data as a DataFrame.
    """
    filename_sample = os.path.join(tools.DATA_MONITORING_SAMPLE_DIR, "baseline_dataset.csv")
    if os.path.exists(filename_sample):
        df = pd.read_csv(filename_sample, index_col=0)
        df = df.rename(columns={"prdtypecode": "target"})
        return df
    logging.warning(f"Baseline file {filename_sample} does not exist.")
    return None

def get_data_sample_filepaths() -> str:
    """
    Get the file path for the data sample.
    Returns:
        str: Path to the data sample file.
    """
    if not os.path.exists(tools.DATA_MONITORING_SAMPLE_DIR):
        logging.warning(f"Data monitoring sample directory {tools.DATA_MONITORING_SAMPLE_DIR} does not exist.")
        return None
    #X_files = sorted(f for f in os.listdir(tools.DATA_MONITORING_SAMPLE_DIR) if f.contains("X_") and f.endswith(".csv"))
    files = sorted(Path(tools.DATA_MONITORING_SAMPLE_DIR).glob("*stream*csv"))
    filepaths = files #[os.path.join(tools.DATA_MONITORING_SAMPLE_DIR, file) for file in files]
    return filepaths

def add_report_to_workspace(workspace, project_name, project_description, report):
    """
    Adds a report to an existing or new project in a workspace.
    """
    # Check if project already exists
    project = None
    for p in workspace.list_projects():
        if p.name == project_name:
            project = p
            break

    # Create a new project if it doesn't exist
    if project is None:
        project = workspace.create_project(project_name)
        project.description = project_description
        workspace.add_project(project)

    # Add report to the project
    workspace.add_report(project.id, report)
    print(f"New report added to project {project_name}")

def main():
    """
    Main function to generate the Evidently report.
    This function initializes the Evidently workspace, creates a project,
    sets the target variable, and adds daily batches of data.
    """
    WORKSPACE_NAME = "evidently_workspace"
    PROJECT_NAME = "Rakuten Product Classification"
    PROJECT_DESCRIPTION = "Evidently Dashboards"
    baseline_data = get_baseline_data()
    if baseline_data is None:
        logger.error("No baseline data found. The report will not include baseline comparisons.")
        return


    filepaths = get_data_sample_filepaths()
    if not filepaths:
        logger.error("No data samples found. The report will not include data samples.")
        return

    # Initialize the Evidently workspace
    workspace = Workspace(WORKSPACE_NAME)

    # Create or retrieve the project
    project = None
    for p in workspace.list_projects():
        if p.name == PROJECT_NAME:
            project = p
            break

    if project is None:
        project = workspace.create_project(PROJECT_NAME)
        project.description = PROJECT_DESCRIPTION
        workspace.add_project(project)

    print(filepaths)
    filepath = filepaths[-1]
    try:
        curr_data = pd.read_csv(filepath, index_col=0)
        curr_data = curr_data.rename(columns={"prdtypecode": "target"})
    except Exception as e:
        logger.error(f"Could not read file {filepath}: {e}")
        raise

    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
#            ClassificationPreset(probas_threshold=0.5),
   ])

    report.run(
        reference_data=baseline_data,
        current_data=curr_data,
    )

    workspace.add_report(project.id, report)
    logger.info(f"Added report for {filepath.name}")


if __name__ == "__main__":
    # Save the project
    # Set up logging    
    logging.basicConfig(level=logging.INFO)

    main()
