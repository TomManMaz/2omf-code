from __future__ import annotations

import logging

def configure_logging(level=logging.INFO):                                                                                                                                                  
    """Call once at application startup (in main.py)"""                                                                                                                                     
    f = '%(asctime)s|%(levelname)s|%(name)s|%(message)s'                                                                                                                                    
    logging.basicConfig(level=level, format=f)                                                                                                                                              
    logging.getLogger("gurobipy").propagate = False # Disable propagation of the "gurobipy" logger

def get_logger(logger_name: str) -> logging.Logger:
    """Get the logger by name"""
    f = '%(asctime)s|%(levelname)s|%(name)s|%(message)s'
    logging.basicConfig(level=logging.INFO, format=f)
    return logging.getLogger(logger_name)