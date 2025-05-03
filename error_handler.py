import traceback
import streamlit as st
from typing import Any, Callable, Dict, TypeVar, cast

T = TypeVar('T')

def safe_execute(func: Callable[..., T], default_return: Any = None, **kwargs) -> T:
    """
    Safely executes a function with exception handling and returns a default value if it fails.
    
    Args:
        func: The function to execute
        default_return: The default value to return if the function fails
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function or the default value if it fails
    """
    try:
        return func(**kwargs)
    except Exception as e:
        st.error(f"Error: {str(e)}")
        traceback.print_exc()
        return default_return

def validate_list(data: Any, element_validator: Callable[[Any], bool] = None) -> list:
    """
    Validates that data is a list and optionally validates each element.
    
    Args:
        data: The data to validate
        element_validator: Optional function to validate each element
        
    Returns:
        The data if it's a valid list, otherwise an empty list
    """
    if not isinstance(data, list):
        return []
    
    if element_validator is None:
        return data
    
    return [item for item in data if element_validator(item)]

def validate_dict(data: Any, required_keys: list = None) -> dict:
    """
    Validates that data is a dictionary and optionally checks for required keys.
    
    Args:
        data: The data to validate
        required_keys: Optional list of keys that must exist in the dictionary
        
    Returns:
        The data if it's a valid dictionary, otherwise an empty dictionary
    """
    if not isinstance(data, dict):
        return {}
    
    if required_keys is None:
        return data
    
    # Check if all required keys exist
    if all(key in data for key in required_keys):
        return data
    else:
        return {}

def is_valid_subtask(item: Any) -> bool:
    """
    Validates that an item is a valid subtask dictionary.
    
    Args:
        item: The item to validate
        
    Returns:
        True if the item is a valid subtask, False otherwise
    """
    return (
        isinstance(item, dict) and
        "id" in item and
        "description" in item
    )

def get_nested_value(data: Dict, keys: list, default: Any = None) -> Any:
    """
    Safely gets a nested value from a dictionary using a list of keys.
    
    Args:
        data: The dictionary to get the value from
        keys: A list of keys to traverse the dictionary
        default: The default value to return if the key doesn't exist
        
    Returns:
        The value at the specified path or the default value
    """
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current