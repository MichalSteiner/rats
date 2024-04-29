#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 13:24:03 2022

@author: chamaeleontis

Contains utility functions and decorators to use in other script.

"""
#%% Importing libraries
import dill as pickle
import time
import os
from functools import wraps
import inspect
import logging
logger = logging.getLogger(__name__)

def addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present 

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        pass
    #    raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
        pass
    #    raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        pass
    #    raise AttributeError('{} already defined in logger class'.format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)
    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)

def default_logger_format(logger_update):
    addLoggingLevel('PRINT', 25, methodName=None)
    
    LOG_LEVEL = logging.INFO
    # LOG_LEVEL = logging.DEBUG
    LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
    from colorlog import ColoredFormatter
    logging.root.setLevel(LOG_LEVEL)
    formatter = ColoredFormatter(LOGFORMAT)
    formatter.log_colors['WARNING'] = "cyan"
    formatter.log_colors['PRINT'] = "yellow"
    stream = logging.StreamHandler()
    stream.setLevel(LOG_LEVEL)
    stream.setFormatter(formatter)
    logger_update.setLevel(LOG_LEVEL)
    logger_update.addHandler(stream)
    return logger_update


logger = default_logger_format(logger)

#%% List of decorators
# Basic set to time a function, try loading and saving and to track progress
# from chaos.utilities import time_function, save_and_load, progress_tracker, disable_func, skip_function,save_figure, todo_function
# =============================================================================
# @time_function
# @save_and_load
# @progress_tracker
# =============================================================================

# =============================================================================
# Skips a function for saving time in case output is loaded later on
# =============================================================================
# @skip_function
# =============================================================================
# Disables function to check on breakability of code
# =============================================================================
# @disable_func
# =============================================================================
#%% time_function
def time_function(func):
    '''
    Decorator function for timing the function

    Parameters
    ----------
    func : function
        Function to decorate.

    Returns
    -------
    wrapper : TYPE
        DESCRIPTION.

    '''
    @wraps(func)
    def wrapper(*args,**kwargs):
        '''
        Time the wrapped function and print it.

        Parameters
        ----------
        *args : tuple
            Arguments.
        **kwargs : tuple
            Key arguments.

        Returns
        -------
        output : output
            Output of the wrapped function.

        '''
        t1 = time.time()
        output = func(*args,**kwargs)
        t2 = time.time()-t1
        logger.info(f'{func.__name__} ran in {t2} seconds')
        return output
    return wrapper


#%% save_and_load
def save_and_load(func):
    '''
    Decorator function for saving and loading the function

    Parameters
    ----------
    func : function
        Function to decorate.

    Returns
    -------
    wrapper : function
        Decorated function.

    '''
    signature = inspect.signature(func)
    default_kwargs = {
        kw: value.default for kw, value in signature.parameters.items() if value.default != inspect.Signature.empty
    }
    @wraps(func)
    def wrapper(*args, **kwargs,):
        '''
        Loads and saves the output of the function into pickle file. The functionality is as follows
        force_skip = True:
            Skip the function entirely
        force_load = True and force_skip = False:
            Try to load the function instead of running
            If failed, run the function instead
        force_load = False and force_skip = False:
            Run the function normally

        Parameters
        ----------
        *args : tuple
            Arguments.
        **kwargs : tuple
            Key arguments.

        Returns
        -------
        output : output
            Output of the wrapped function.

        '''
        
        if not(os.path.exists(os.getcwd() + '/saved_data/')):
            os.mkdir(os.getcwd() + '/saved_data/')
        
        try:
            file =  os.getcwd() + '/saved_data/' + kwargs['pkl_name']
        except:
            file =  os.getcwd() + '/saved_data/' + func.__name__ + '.pkl'
        
        kwargs = default_kwargs | kwargs
        try: # Exception so force_skip does not need to be defined
            kwargs['force_skip']
        except:
            kwargs['force_skip'] = False
        try: # Exception so force_load does not need to be defined
            kwargs['force_load']
        except:
            kwargs['force_load'] = False
            
        
        if kwargs['force_skip'] == True:
            logger.info('Currently skipping working on: '+f'{func.__name__}')
            return None
        elif kwargs['force_load'] == True:
            try:
                with open(file, 'rb') as input_file:
                    output =  pickle.load(input_file)
                return output
            except:
                logger.info('Error opening input file: %s'%file)
                pass
        output = func(*args,**kwargs)
        with open(file, 'wb') as output_file:
            logger.info('Saved progress in %s'%file)
            pickle.dump(output, output_file)
        return output
    return wrapper

#%% progress_tracker
def progress_tracker(func):
    '''
    Decorator function for progress tracking of the function

    Parameters
    ----------
    func : function
        Function to decorate.

    Returns
    -------
    wrapper : function
        Decorated function.

    '''
    @wraps(func)
    def wrapper(*args,**kwargs):
        '''
        Prints the name of the function currently running

        Parameters
        ----------
        *args : tuple
            Arguments.
        **kwargs : tuple
            Key arguments.

        Returns
        -------
        output : output
            Output of the wrapped function.

        '''
        
        logger.info('Currently working on:')
        logger.info(f'    {func.__name__}')
        output = func(*args,**kwargs)
        return output
    
    return wrapper

#%% disable_func
def disable_func(func):
    '''
    This decorator disables a function. Use this to check breakability of code by disabling (for removing legacy code pieces).

    Parameters
    ----------
    func : function
        Function to decorate.

    Returns
    -------
    wrapper : function
        Decorated function.

    '''
    @wraps(func)
    def wrapper(*args,**kwargs):
        logger.error('Function: '+f'{func.__name__}'+' has been called while disabled!')
        pass
    return wrapper
#%% skip_function
def skip_function(func):
    '''
    This decorator disables a function based on a provided keyword. Use this to of code by disabling for time saving.

    Parameters
    ----------
    func : function
        Function to decorate.

    Returns
    -------
    wrapper : function
        Decorated function.

    '''
    signature = inspect.signature(func)
    default_kwargs = {
        kw: value.default for kw, value in signature.parameters.items() if value.default != inspect.Signature.empty
    }
    @wraps(func)
    def wrapper(*args,**kwargs):
        kwargs = default_kwargs | kwargs
        
        if kwargs['force_skip']:
            logger.info('Currently skipping working on: \n'+f'    {func.__name__}')
            pass
        else:
            output = func(*args,**kwargs)
            return output
    return wrapper

#%%
def rename_aliens_mac(path:str):
    """
    Removes aliens from the filename going from Mac filesystem to FAT filesystem.

    Parameters
    ----------
    path : str
        Path to the directory.

    Returns
    -------
    None.

    """
    for item in os.listdir(path):
        os.rename(path + item,
                  path + item.replace('\uf022', '_'))
    return

#%% skip_function
def skip_function(func):
    '''
    This decorator disables a function based on a provided keyword. Use this to of code by disabling for time saving.

    Parameters
    ----------
    func : function
        Function to decorate.

    Returns
    -------
    wrapper : function
        Decorated function.

    '''
    signature = inspect.signature(func)
    default_kwargs = {
        kw: value.default for kw, value in signature.parameters.items() if value.default != inspect.Signature.empty
    }
    @wraps(func)
    def wrapper(*args,**kwargs):
        kwargs = default_kwargs | kwargs
        
        if kwargs['force_skip']:
            logger.info('Currently skipping working on: '+f'{func.__name__}')
            pass
        else:
            output = func(*args,**kwargs)
            return output
    return wrapper


def print_acknowledgements():
    logger.print('Please acknowledge following codes, if used in analysis:')
    logger.print('    RATS: (this pipeline)')
    logger.print('        Steiner et al. 2023')
    logger.print('        Steiner et al. 2024')
    
    logger.print('    petitRADTRANS: (CCF templates)')
    logger.print('        Molliere et al. 2019')
    
    logger.print('    LDCU: (Limb darkening coefficients for batman light-curves)')
    # TODO
    logger.print('        TODO:')
    
    logger.print('    batman: (Modeling of light curves, also used in some weightings)')
    logger.print('        TODO:')
    
    
    logger.print('    StarRotator: (Modeling of RM + CLV effects)')
    logger.print('        TODO:')
    
    
    raise NotImplementedError