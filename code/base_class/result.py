'''
Base evaluate class for all evaluation metrics and methods
'''

# Copyright (c) 2025 Yian Chen <cya187508866962021@163.com>


import abc

class result:
    """
    ResultModule: Abstract Class
    Entries: 
    """
    
    data = None
    
    result_name = None
    result_description = None
    
    result_destination_folder_path = None
    result_destination_file_name = None
    
    # initialization function
    def __init__(self, rName=None, rType=None):
        self.result_name = rName
        self.result_description = rType

    @abc.abstractmethod
    def save(self):
        return
 
    @abc.abstractmethod
    def load(self):
        return
