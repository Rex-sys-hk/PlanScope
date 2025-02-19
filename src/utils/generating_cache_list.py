from pathlib import Path
from typing import Dict, List, Set, cast
import json
import pickle
import logging
logger = logging.getLogger(__name__)

# nuplan-devkit/nuplan/planning/script/config/common/scenario_builder/nuplan_challenge.yaml:4
# data_root: ${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/test/
# nuplan-devkit/nuplan/planning/script/config/common/scenario_builder/nuplan.yaml:4
# data_root: ${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/trainval/

# nuplan-devkit/nuplan/planning/script/builders/scenario_builder.py:67
def get_local_scenario_cache(cache_path: str, feature_names: Set[str]):
    """
    Get a list of cached scenario paths from a local cache.
    :param cache_path: Root path of the local cache dir.
    :param feature_names: Set of required feature names to check when loading scenario paths from the cache.
    :return: List of discovered cached scenario paths.
    """
    cache_dir = Path(cache_path)
    assert cache_dir.exists(), f'Local cache {cache_dir} does not exist!'
    assert any(cache_dir.iterdir()), f'No files found in the local cache {cache_dir}!'

    cache_index = cache_dir/"cache_index.pkl"
    cache_list = cache_dir/"cache_list.pkl"
    if cache_list.exists():
        logger.info(f'Reading local files list from {cache_list}')
        with open(cache_list, "rb") as file:
            scenario_cache_paths = pickle.load(file)
    else:
        logger.info(f'Reading local files from {cache_dir}')
        if cache_index.exists():
            logger.info(f'Reading local files index from {cache_index}')
            with open(cache_index, "rb") as file:
                candidate_scenario_dirs = pickle.load(file)
        else:
            # cache_index.mkdir(parents=False, exist_ok=True)
            candidate_scenario_dirs = {x.parent for x in cache_dir.rglob("*.gz")}
            with open(cache_index, "wb") as file:
                pickle.dump(candidate_scenario_dirs, file)
            logger.info(f'Cache index saved to {cache_index}')
        # Keep only dir paths that contains all required feature names
        logger.info('Got candidate_scenario_dirs.')
        scenario_cache_paths = [
            path
            for path in candidate_scenario_dirs
            if not (feature_names - {feature_name.stem for feature_name in path.iterdir()})
        ]
        with open(cache_list, "wb") as file:
            pickle.dump(scenario_cache_paths, file)
        logger.info(f'Cache index list to {cache_list}')

    return scenario_cache_paths

if __name__ == "__main__":
    cache_path = "/training-pfs-shanghai/private/renxin/PlanScope/exp/cache_pluto_1M"
    logger.info(get_local_scenario_cache(cache_path, {'trajectory', 'feature'})[:20])