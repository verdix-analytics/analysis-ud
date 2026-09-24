import json
import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

_JSON_PATH = Path(__file__).resolve().parents[1] / "Utils_data" / "window_size.json"

def _load_pattern_windows(path: Path, key: str) -> dict[str, int]:
    """
    Function to load the json file that stores all the different window sizes
    for different pattern detection algorithms
    """
    try:
        json_raw = json.loads(Path(path).read_text(encoding = "utf-8"))

        pattern_sizes = {}
        errors = []
        patterns = json_raw.get(key, {})
        
        for key, value in patterns.items():
            if not isinstance(value, int) or value < 1:
                errors.append(f"{key}: expected a positive integer for window size but got: {value}")
            else:
                pattern_sizes[key] = value
        
        if errors:
            raise ValueError("Errors found while loading the data: "+ "\n".join(errors))
        return pattern_sizes
     
    except FileNotFoundError as fe:
        logger.error(f"{path} does not contain window sizes")
    except json.JSONDecodeError as de:
        logger.error(f"Unable to decode the json in the file: {path}")
    except Exception as e:
        logger.error(f"Failed to load the json file due to : {e}")
    return {}
    
class WindowRegistry:
    """
    Class that loads the windows for each patterns on request
    This allows for complete modularity between developing the algorithm and 
    worry about the window sizes and handling such details by the algorithm.
    """
    def __init__(self, path: Path = _JSON_PATH):
        self._path = path
        self._pattern_sizes: dict[str, int] = {}
        self._pretrend_sizes: dict[str, int] = {}
        self._load()
        
    def _load(self) -> None:
        '''
        Function to load all the pattern windows from the json file
        '''
        self._pattern_sizes = _load_pattern_windows(self._path, "patterns")
        self._pretrend_sizes = _load_pattern_windows(self._path, "pretrend_windows")
    
    def reload(self) -> None:
        '''
        Reload the pattern window if the data is lost from memory
        '''
        self._load()
        logger.info("Window Size Register: reloaded window from disk")
    
    def get(self, pattern) -> int:
        """
        Get the pattern window size when a pattern is provided
        """
        size: int = 0
        try:
            if pattern in self._pattern_sizes:
                size = self._pattern_sizes[pattern]
                logger.info("Returning the window size for the requested pattern: {pattern} - {size}")
                return size
            else:
                raise ValueError("Pattern not found in registry: {pattern}")
        except Exception as e:
            logger.error(f"Failed to load pattern {pattern} due to : {e}")
        return size
    
    def get_patterns(self) -> set[str]:
        """
        Get all the pattern window sizes available
        """
        if self._pattern_sizes:
            return set(self._pattern_sizes.keys())

    def get_pretrend(self, pattern: str, default: int = 20) -> int:
        """
        Get the configured pre-trend window for a pattern.
        """
        return self._pretrend_sizes.get(pattern, default)
    
    def __iter__(self) -> Iterator[tuple[str, int]]:
        """
        Iterate over all the pattern sizes available.
        """
        return iter(self._pattern_sizes.items())
    
registry = WindowRegistry()
    
