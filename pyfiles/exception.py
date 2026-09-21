import sys
import logging
from click import File
import sys 
sys.path.append(r'D:\LLMOps\pyfiles')
import logger 


def error_message_detail(error, error_detail: sys) -> str:
    """
    Extracts detailed error information including:
    - File name where error occurred
    - Line number of the error
    - Error message
    """
    _, _, exc_tb = error_detail.exc_info() 

    if exc_tb is None:
        return str(error)     

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = (
        f"\n"
        f"  Script     : [{file_name}]\n"
        f"  Line No    : [{line_number}]\n"
        f"  Error Msg  : [{str(error)}]\n"
    )
    return error_message


class CustomException(Exception):
    """
    Custom Exception class for KalamMinds MLProject.
    Captures file name, line number, and error message automatically. 
    
    Usage:
        try:
            ...
        except Exception as e:
            raise CustomException(e, sys)
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
        # auto log the error when exception is raised
        logger.error(self.error_message)

    def __str__(self):
        return self.error_message

class DataIngestionException(CustomException):
    """Raised when data loading or ingestion fails."""
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message, error_detail)
        self.error_message = f"[DataIngestion Error] {self.error_message}"

class DataPreparationException(CustomException):
    """Raised when data preprocessing or transformation fails."""
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message, error_detail)
        self.error_message = f"[DataTransformation Error] {self.error_message}"

class EmbedModelLoaderException(CustomException):
    """Raised when prediction/inference fails."""
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message, error_detail)   
        self.error_message = f"[Error during loading the 'EmbedModel'] {self.error_message}"
 
class ModelLoaderException(CustomException):
    """Raised when model was getting load."""
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message, error_detail)
        self.error_message = f"[Error during loading the Model] {self.error_message}"

class RetrievialSystemException(CustomException):
    """Raised when prediction/inference fails."""
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message, error_detail)
        self.error_message = f"[RetrievalSystem Error] {self.error_message}"
        
