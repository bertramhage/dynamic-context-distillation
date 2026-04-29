# Super simple wrapper around the wandb API to get all runs for a project

import wandb
import pandas as pd

class WandbClient():
    def __init__(self, project):
        self.api = wandb.Api()
        self.entity = "bertram-hage-danmarks-tekniske-universitet-dtu"
        self.project = project

    def _get_runs_for_project(self, filters) -> list:
        return self.api.runs(f"{self.entity}/{self.project}", filters=filters)

    def get_runs(self, filters: dict = {}, return_failed: bool = False):
        runs = self._get_runs_for_project(filters=filters)
        
        run_list = []
        for run in runs:
            if not return_failed and run.state != "finished":
                continue

            data = {
                "name": run.name,
                "id": run.id,
                "state": run.state,
            }
            
            data.update(run.summary._json_dict)
            
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
            data.update(config)
            
            run_list.append(data)

        df = pd.json_normalize(run_list, sep='_') 

        df = df.dropna(axis=1, how='all')
        return df