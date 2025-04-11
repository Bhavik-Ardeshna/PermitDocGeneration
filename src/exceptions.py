"""
This module contains custom exception classes for handling different types of errors in the rubric generation pipeline.

Classes:
- BaseCustomException: Base class for custom exceptions.
- LimitExceedError: Raised when a limit is exceeded.
- UnexpectedError: Custom exception for unexpected errors.
- InvalidDocumentTypeError: Raised when extraction tool not in Document ai and Unstructured
- InvalidChunkStrategy: Raised raised when chunking strategy is not in by_title and by_elements
- GoogleApiError: Raised raised when exception is occured when interaction with Google APIs
- MilvusDBError: Raised when GRPC Error occurs in while interaction with milvus db
"""


class BaseCustomException(Exception):
 
    '''
    Base class for custom exceptions
    '''
 
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class UnexpectedError(BaseCustomException):
    #used in ai_infer.py
    """
    Custom exception for unexpected errors.
    """
    def __init__(self, message="Unexpected error occurred"):
        super().__init__(self.message)

class InvalidExtractionTool(BaseCustomException):
    #used in main.py
    """
    Exception raised when extraction tool not in Document ai and Unstructured
    """
    def __init__(self):
        self.message = "Invalid Extraction Tool. Extraction Tool should be either Document AI or Unstructured"
        super().__init__(self.message)

        
class InvalidChunkStrategy(BaseCustomException):
    #used in main.py
    """
    Exception raised when chunking strategy is not in by_title and by_elements
    """
    def __init__(self):
        self.message = "Invalid Chunking Strategy. Chunking strategy should be either by_title or by_elements"
        super().__init__(self.message)

class GoogleApiError(BaseCustomException):
    #used in document_ai_extractor
    """
    Exception raised when exception is occured when interaction with Google APIs
    """
    def __init__(self):
        self.message = "Google server side error occured"
        super().__init__(self.message)
        
class MilvusDBError(BaseCustomException):
     #used in vecdb/utils
    """
    Exception raised when GRPC Error occurs in while interaction with milvus db
    """
    def __init__(self):
        self.message = "Milvus DB Grpc error occured"
        super().__init__(self.message)

class FileNotFoundInGCS(BaseCustomException):
    
    # Used in gcs.py
    
    """
    Exception raised when doc id is not found in GCS
    """
    def __init__(self):
        self.message = f"File not found in Google Cloud Storage."
        super().__init__(self.message)


